"""Command line entry point: ``federated-secure-aggregation`` or ``python -m ...``.

Subcommands:

``demo``              run the offline end-to-end demo (no network, no API key)
``serve``             run the coordinator HTTP service
``silo``              run one silo against a coordinator (the compose silo containers)
``operate``           open rounds and advance stages (the compose operator container)
``audit verify``      verify a round audit chain, optionally against an expected head
``budget``            report the privacy budget a ledger file has spent
``generate-identity`` print a fresh Ed25519 identity key pair for a silo
``generate-api-key``  print a transport API key and the digest to configure
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:  # pragma: no cover - needs a socket
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from federated_secure_aggregation.core.config import get_settings  # noqa: PLC0415
    from federated_secure_aggregation.core.logging import configure_logging  # noqa: PLC0415
    from federated_secure_aggregation.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )
    from federated_secure_aggregation.service.runtime import build_service_from_env  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = build_service_from_env(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _silo(args: argparse.Namespace) -> int:  # pragma: no cover - needs a socket
    """Run one silo container: load its key material, then join rounds until done."""
    import httpx  # noqa: PLC0415

    from federated_secure_aggregation.core.logging import configure_logging  # noqa: PLC0415
    from federated_secure_aggregation.secagg.client import SecAggClient  # noqa: PLC0415
    from federated_secure_aggregation.service.runtime import (  # noqa: PLC0415
        load_identity,
        load_roster,
    )
    from federated_secure_aggregation.service.silo_runner import (  # noqa: PLC0415
        SiloConfig,
        run_silo,
    )

    configure_logging("INFO", json_output=True, stream=sys.stderr)
    identity = load_identity(args.identity)
    roster = load_roster(args.roster)
    coordinator_key = base64.b64decode(
        json.loads(args.identity.read_text(encoding="utf-8"))["coordinator_public_key_b64"],
        validate=True,
    )
    client = SecAggClient(
        args.silo_id,
        identity,
        peer_identity_keys=roster,
        coordinator_identity_key=coordinator_key,
    )
    config = SiloConfig(
        silo_id=args.silo_id, n_silos=len(roster), rounds=args.rounds, seed=args.seed
    )
    with httpx.Client(base_url=args.coordinator, timeout=60.0) as http:
        summary = run_silo(config, client, http, api_key=_require_api_key())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _operate(args: argparse.Namespace) -> int:  # pragma: no cover - needs a socket
    """Run the operator container: open rounds and ask the coordinator to advance stages."""
    import httpx  # noqa: PLC0415

    from federated_secure_aggregation.core.logging import configure_logging  # noqa: PLC0415
    from federated_secure_aggregation.service.silo_runner import run_operator  # noqa: PLC0415

    configure_logging("INFO", json_output=True, stream=sys.stderr)
    with httpx.Client(base_url=args.coordinator, timeout=60.0) as http:
        outcomes = run_operator(http, api_key=_require_api_key(), rounds=args.rounds)
    print(json.dumps(outcomes, indent=2, sort_keys=True))
    accepted = sum(1 for outcome in outcomes if outcome.get("accepted"))
    print(f"{accepted}/{len(outcomes)} rounds accepted", file=sys.stderr)
    return 0


def _require_api_key() -> str:  # pragma: no cover - needs the environment
    """Read the transport API key from the environment, never from a flag.

    A flag would put the key in the process table and in shell history; compose passes it as an
    environment variable from the operator's own secret.
    """
    key = os.environ.get("APP_API_KEY")
    if not key:
        raise SystemExit("APP_API_KEY is not set; generate one with `generate-api-key`")
    return key


def _demo(workdir: Path) -> int:
    from federated_secure_aggregation.core.logging import configure_logging  # noqa: PLC0415
    from federated_secure_aggregation.demo import run_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    print(json.dumps(run_demo(workdir), indent=2, sort_keys=True))
    return 0


def _budget(path: Path) -> int:
    from federated_secure_aggregation.privacy.ledger import (  # noqa: PLC0415
        BudgetLedger,
        BudgetState,
    )

    state = BudgetState.from_json(path.read_text(encoding="utf-8"))
    ledger = BudgetLedger(
        noise_multiplier=state.noise_multiplier,
        target_epsilon=state.target_epsilon,
        delta=state.delta,
        sample_rate=state.sample_rate,
        path=path,
    )
    print(
        json.dumps(
            {
                "rounds_spent": ledger.steps,
                "epsilon_spent": round(ledger.epsilon_spent(), 6),
                "target_epsilon": ledger.target_epsilon,
                "delta": ledger.delta,
                "noise_multiplier": round(ledger.noise_multiplier, 6),
                "rounds_remaining": ledger.remaining_rounds(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="federated-secure-aggregation")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the coordinator HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    silo = sub.add_parser("silo", help="run one silo against a coordinator")
    silo.add_argument("silo_id")
    silo.add_argument("--coordinator", default="http://coordinator:8000")
    silo.add_argument("--identity", type=Path, default=Path("secrets/identity.json"))
    silo.add_argument("--roster", type=Path, default=Path("secrets/roster.json"))
    silo.add_argument("--rounds", type=int, default=20)
    silo.add_argument("--seed", type=int, default=20260920)

    operate = sub.add_parser("operate", help="open rounds and advance stages")
    operate.add_argument("--coordinator", default="http://coordinator:8000")
    operate.add_argument("--rounds", type=int, default=20)

    audit = sub.add_parser("audit", help="round audit chain tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained round audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    budget = sub.add_parser("budget", help="report a privacy budget ledger")
    budget.add_argument("path", type=Path)

    sub.add_parser("generate-api-key", help="print a transport API key and its SHA-256 digest")
    sub.add_parser("generate-identity", help="print a new Ed25519 identity key pair")

    args = parser.parse_args(argv)
    handler = _HANDLERS[str(args.command)]
    return handler(args)


def _generate_api_key(args: argparse.Namespace) -> int:
    from federated_secure_aggregation.core.security import generate_api_key  # noqa: PLC0415

    del args
    raw, digest = generate_api_key()
    print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
    print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
    return 0


def _generate_identity(args: argparse.Namespace) -> int:
    from federated_secure_aggregation.crypto.keys import IdentityKeyPair  # noqa: PLC0415

    del args
    identity = IdentityKeyPair.generate()
    print(
        json.dumps(
            {
                "private_key_b64": base64.b64encode(identity.private_bytes).decode("ascii"),
                "public_key_b64": base64.b64encode(identity.public_bytes).decode("ascii"),
            },
            indent=2,
        )
    )
    print("Keep private_key_b64 inside the silo; publish only public_key_b64.", file=sys.stderr)
    return 0


def _audit(args: argparse.Namespace) -> int:
    from federated_secure_aggregation.monitoring.chain import main as audit_main  # noqa: PLC0415

    forwarded = ["verify", str(args.path)]
    if args.expect_head:
        forwarded += ["--expect-head", args.expect_head]
    return audit_main(forwarded)


# One entry per subcommand. A table rather than a chain of ifs, so adding a command cannot
# silently fall through to the wrong handler.
_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "serve": lambda args: _serve(args.host, args.port),
    "demo": lambda args: _demo(args.workdir),
    "silo": _silo,
    "operate": _operate,
    "budget": lambda args: _budget(args.path),
    "audit": _audit,
    "generate-api-key": _generate_api_key,
    "generate-identity": _generate_identity,
}


if __name__ == "__main__":
    raise SystemExit(main())
