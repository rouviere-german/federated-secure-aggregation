"""The per-silo data contract.

In a federated run the coordinator never sees a silo's rows, so a malformed silo cannot be
spotted downstream: the contract is the only place it can be caught, and it runs inside the
silo before anything is trained or masked. These tests pin what the contract rejects, that it
reports *every* violation at once (lazy validation) rather than the first one, and that the
generated population satisfies it for any seed.
"""

from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from federated_secure_aggregation.fl.data import make_population
from federated_secure_aggregation.ml.contracts import (
    DATAFRAME_SCOPE,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    DataContractError,
    validate_borrowers,
)

pytestmark = pytest.mark.unit


def failures_for(frame: pd.DataFrame) -> list[dict[str, str | int]]:
    with pytest.raises(DataContractError) as info:
        validate_borrowers(frame)
    return info.value.failures


def checks_in(failures: list[dict[str, str | int]]) -> set[tuple[str, str]]:
    return {(str(item["column"]), str(item["check"])) for item in failures}


@pytest.fixture
def population() -> pd.DataFrame:
    return make_population(400, seed=7)


def test_the_generated_population_satisfies_the_contract(population: pd.DataFrame) -> None:
    assert len(validate_borrowers(population)) == len(population)


@settings(max_examples=15, deadline=None)
@given(
    n_rows=st.integers(min_value=60, max_value=300),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_the_generator_satisfies_the_contract_for_any_seed(n_rows: int, seed: int) -> None:
    """If this ever fails, the simulation is feeding silos data they would reject in production."""
    validate_borrowers(make_population(n_rows, seed=seed))


def test_out_of_range_values_are_rejected_per_column(population: pd.DataFrame) -> None:
    frame = population.copy()
    frame.loc[0, "age_years"] = 7.0
    frame.loc[1, "debt_to_income"] = 99.0
    frame.loc[2, "revolving_utilization"] = -0.5
    assert checks_in(failures_for(frame)) >= {
        ("age_years", "greater_than_or_equal_to(18.0)"),
        ("debt_to_income", "less_than_or_equal_to(5.0)"),
        ("revolving_utilization", "greater_than_or_equal_to(0.0)"),
    }


def test_every_violation_is_reported_not_just_the_first(population: pd.DataFrame) -> None:
    """Lazy validation: a silo operator gets the whole list in one pass."""
    frame = population.copy()
    frame.loc[0, "age_years"] = 7.0
    frame.loc[1, "log_income"] = 1.0
    frame.loc[2, "prior_default"] = 3.0
    reported = checks_in(failures_for(frame))
    assert {"age_years", "log_income", "prior_default"} <= {column for column, _ in reported}


def test_the_label_must_be_binary(population: pd.DataFrame) -> None:
    frame = population.copy()
    frame.loc[0, LABEL_COLUMN] = 5
    assert any(str(item["column"]) == LABEL_COLUMN for item in failures_for(frame))


def test_a_single_class_silo_is_rejected(population: pd.DataFrame) -> None:
    """A silo with one class contributes no usable gradient, so it must not join a round."""
    frame = population.copy()
    frame[LABEL_COLUMN] = 0
    assert checks_in(failures_for(frame)) == {(DATAFRAME_SCOPE, "both_classes_present")}


def test_credit_history_cannot_predate_adulthood(population: pd.DataFrame) -> None:
    """A cross-field rule that no per-column range check would catch."""
    frame = population.copy()
    ages = frame["age_years"].astype(float)
    frame.at[0, "credit_age_years"] = ages.iloc[0]
    frame.at[5, "credit_age_years"] = ages.iloc[5]
    failures = failures_for(frame)
    assert checks_in(failures) == {(DATAFRAME_SCOPE, "credit_age_fits_age")}
    # Two bad rows must be reported as two, not as one per column of each row.
    assert failures[0]["count"] == 2


def test_duplicate_borrower_ids_are_rejected(population: pd.DataFrame) -> None:
    frame = pd.concat([population, population.iloc[[0]]], ignore_index=True)
    assert any(str(item["column"]) == "borrower_id" for item in failures_for(frame))


def test_malformed_borrower_ids_are_rejected(population: pd.DataFrame) -> None:
    frame = population.copy()
    frame.loc[0, "borrower_id"] = "not-a-borrower-id"
    assert any(str(item["column"]) == "borrower_id" for item in failures_for(frame))


def test_unexpected_columns_are_rejected(population: pd.DataFrame) -> None:
    """``strict`` matters here: an extra column means the silos disagree on the schema."""
    frame = population.copy()
    frame["leaked_ssn"] = "000-00-0000"
    assert ("leaked_ssn", "column_in_schema") in checks_in(failures_for(frame))


def test_a_missing_feature_column_is_rejected(population: pd.DataFrame) -> None:
    frame = population.drop(columns=["log_income"])
    assert failures_for(frame)


def test_the_error_message_counts_rows_and_checks(population: pd.DataFrame) -> None:
    """Two borrowers outside one column's range is 2 failures of 1 check, not 30 across 15.

    ``region_risk`` is used because it takes part in no cross-field rule, so the count is not
    inflated by a second check firing on the same rows.
    """
    frame = population.copy()
    frame.loc[0, "region_risk"] = 5.0
    frame.loc[1, "region_risk"] = 6.0
    with pytest.raises(DataContractError, match=r"2 failure\(s\) across 1 check\(s\)"):
        validate_borrowers(frame)


def test_failures_never_contain_borrower_values(population: pd.DataFrame) -> None:
    """The whole point of the repo: borrower attributes must not escape the silo, not even
    into an exception message that might be logged or forwarded to the coordinator."""
    frame = population.copy()
    frame.loc[0, "borrower_id"] = "b_deadbeef00"
    frame.loc[0, "log_income"] = 1.234567
    frame.loc[0, "age_years"] = 7.5
    failures = failures_for(frame)
    rendered = str(failures) + str(DataContractError(failures))
    assert "b_deadbeef00" not in rendered
    assert "1.234567" not in rendered
    assert "7.5" not in rendered


def test_the_feature_order_is_fixed_and_excludes_the_label() -> None:
    """Every silo must quantize the same coordinates in the same order or the sum is nonsense."""
    assert LABEL_COLUMN not in FEATURE_COLUMNS
    assert "borrower_id" not in FEATURE_COLUMNS
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
