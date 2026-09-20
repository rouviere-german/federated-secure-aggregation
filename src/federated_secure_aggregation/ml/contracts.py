"""pandera contract for a silo's borrower table, with PII-safe failure reporting.

Every silo validates its own partition *before* it is allowed to join a round. Failure summaries
report column, check and count only: the failing values are borrower attributes, which is
exactly the data the protocol exists to keep inside the silo, so they never reach an exception
message, a log line or the coordinator.

The contract is deliberately strict (no coercion, closed column set). A silo whose schema has
drifted is refused rather than silently reinterpreted, because a column that quietly changes
meaning across silos produces a model that is wrong in a way no metric shows.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing.pandas import Series

__all__ = [
    "DATAFRAME_SCOPE",
    "FEATURE_COLUMNS",
    "LABEL_COLUMN",
    "BorrowerSchema",
    "DataContractError",
    "feature_matrix",
    "label_vector",
    "validate_borrowers",
]

FEATURE_COLUMNS: tuple[str, ...] = (
    "age_years",
    "log_income",
    "debt_to_income",
    "revolving_utilization",
    "delinquencies_2y",
    "credit_age_years",
    "loan_to_income",
    "employment_years",
    "inquiries_6m",
    "open_accounts",
    "prior_default",
    "region_risk",
)
LABEL_COLUMN = "defaulted"

# Attributed column for checks that span the whole frame rather than one column.
DATAFRAME_SCOPE = "<dataframe>"


class DataContractError(ValueError):
    """A silo's frame violated the contract. ``failures`` holds (column, check, count) rows."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


class BorrowerSchema(pa.DataFrameModel):
    """One row per borrower in one silo."""

    borrower_id: Series[str] = pa.Field(unique=True, str_matches=r"^b_[0-9a-f]{10}$")
    age_years: Series[float] = pa.Field(ge=18.0, le=100.0)
    log_income: Series[float] = pa.Field(ge=7.0, le=16.0)
    debt_to_income: Series[float] = pa.Field(ge=0.0, le=5.0)
    revolving_utilization: Series[float] = pa.Field(ge=0.0, le=2.0)
    delinquencies_2y: Series[float] = pa.Field(ge=0.0, le=20.0)
    credit_age_years: Series[float] = pa.Field(ge=0.0, le=60.0)
    loan_to_income: Series[float] = pa.Field(ge=0.0, le=10.0)
    employment_years: Series[float] = pa.Field(ge=0.0, le=50.0)
    inquiries_6m: Series[float] = pa.Field(ge=0.0, le=20.0)
    open_accounts: Series[float] = pa.Field(ge=0.0, le=40.0)
    prior_default: Series[float] = pa.Field(isin=[0.0, 1.0])
    region_risk: Series[float] = pa.Field(ge=0.0, le=1.0)
    defaulted: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True
        name = "borrowers"

    @pa.dataframe_check
    @classmethod
    def credit_age_fits_age(cls, frame: pd.DataFrame) -> Series[bool]:
        """A borrower cannot have had credit before turning 18."""
        return Series[bool](frame["credit_age_years"] <= frame["age_years"] - 18.0)

    @pa.dataframe_check
    @classmethod
    def both_classes_present(cls, frame: pd.DataFrame) -> bool:
        """A silo with a single class cannot contribute a useful gradient."""
        return bool(frame[LABEL_COLUMN].nunique() == 2)


def _summarize(cases: pd.DataFrame) -> list[dict[str, str | int]]:
    """Turn pandera's failure cases into one row per distinct problem.

    Column checks group by ``(column, check)`` and count failing rows.

    Dataframe-level checks need separate handling. A check that returns a boolean *Series* fails
    per row, and pandera then emits one failure case for every column of each failing row;
    grouping those the same way would turn two bad rows into fourteen reported columns and make
    the count meaningless. They are collapsed to one entry per check, counted by distinct
    failing row index and attributed to :data:`DATAFRAME_SCOPE`. A check that returns a plain
    ``bool`` fails once for the whole frame and has no row index, so it counts as one.

    Schema-shape violations (an extra or missing column) are not declared checks at all, which
    is exactly how they are told apart: pandera leaves their ``check_number`` empty. Their
    failure case is the offending column name, so it is reported as the column.
    """
    failures: list[dict[str, str | int]] = []
    is_column = cases["schema_context"] == "Column"

    for (column, check), group in cases[is_column].groupby(["column", "check"], sort=False):
        failures.append({"column": str(column), "check": str(check), "count": len(group)})

    frame_level = cases[~is_column]
    shape_problem = frame_level["check_number"].isna()

    for check, group in frame_level[shape_problem].groupby("check", sort=False):
        failures.extend(
            {"column": str(value), "check": str(check), "count": 1}
            for value in sorted(group["failure_case"].astype(str).unique())
        )

    for check, group in frame_level[~shape_problem].groupby("check", sort=False):
        rows = group["index"].dropna()
        failures.append(
            {
                "column": DATAFRAME_SCOPE,
                "check": str(check),
                "count": int(rows.nunique()) if not rows.empty else 1,
            }
        )

    return sorted(failures, key=lambda item: (str(item["column"]), str(item["check"])))


def validate_borrowers(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate lazily (all failures at once) and raise a summarized DataContractError."""
    try:
        return BorrowerSchema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        raise DataContractError(_summarize(exc.failure_cases)) from None


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The feature columns in the fixed order every silo must agree on."""
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise DataContractError(
            [{"column": name, "check": "present", "count": 1} for name in missing]
        )
    return frame.loc[:, list(FEATURE_COLUMNS)]


def label_vector(frame: pd.DataFrame) -> pd.Series[int]:
    if LABEL_COLUMN not in frame.columns:
        raise DataContractError([{"column": LABEL_COLUMN, "check": "present", "count": 1}])
    return frame[LABEL_COLUMN].astype("int64")
