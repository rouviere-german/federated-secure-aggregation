#!/usr/bin/env python
"""Generate the identity key material for a federation, once, outside the repository.

Writes, into ``--out`` (default ``secrets/``):

    coordinator_identity.json   the coordinator's Ed25519 private key
    roster.json                 silo id -> base64 Ed25519 *public* key
    silo_<id>_identity.json     one private key per silo
    compose.env                 one transport API key per identity, plus the digest lists

The transport keys are per identity on purpose. One shared key would make the operator routes
(open a round, advance a stage) reachable by every silo, and would put all six processes in a
single rate-limit bucket, because the bucket is keyed on the presented key.

Nothing here is ever committed: ``.gitignore`` excludes ``secrets/``, and CI runs gitleaks. In
compose each file is bind-mounted read-only into exactly the container that needs it, so a silo
container holds its own private key and the roster of public keys, and nothing else.

Keys are generated here rather than at service start-up on purpose. A coordinator whose identity
key silently changed would invalidate every silo's stored copy, and silos would have no way to
tell that from an impersonation attempt.

    uv run python scripts/generate_identities.py --silos 5
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from federated_secure_aggregation.core.security import generate_api_key
from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.fl.experiment import silo_ids


def _write(path: Path, payload: dict[str, str]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Best effort on POSIX; Windows ignores the mode bits.
    path.chmod(0o600)


def generate(out: Path, silos: int) -> list[Path]:
    """Create the coordinator identity, the roster and one identity per silo."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    coordinator = IdentityKeyPair.generate()
    coordinator_path = out / "coordinator_identity.json"
    _write(
        coordinator_path,
        {
            "private_key_b64": base64.b64encode(coordinator.private_bytes).decode("ascii"),
            "public_key_b64": base64.b64encode(coordinator.public_bytes).decode("ascii"),
        },
    )
    written.append(coordinator_path)

    roster: dict[str, str] = {}
    for silo_id in silo_ids(silos):
        identity = IdentityKeyPair.generate()
        roster[silo_id] = base64.b64encode(identity.public_bytes).decode("ascii")
        silo_path = out / f"{silo_id}_identity.json"
        _write(
            silo_path,
            {
                "silo_id": silo_id,
                "private_key_b64": base64.b64encode(identity.private_bytes).decode("ascii"),
                "public_key_b64": roster[silo_id],
                "coordinator_public_key_b64": base64.b64encode(coordinator.public_bytes).decode(
                    "ascii"
                ),
            },
        )
        written.append(silo_path)

    roster_path = out / "roster.json"
    roster_path.write_text(json.dumps(roster, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(roster_path)

    written.append(write_compose_env(out, sorted(roster)))
    return written


def write_compose_env(out: Path, ids: list[str]) -> Path:
    """One transport API key per identity, in the form ``docker-compose.yml`` expects."""
    operator_raw, operator_digest = generate_api_key()
    silo_keys = {silo_id: generate_api_key() for silo_id in ids}
    lines = [
        "# Transport API keys, one per identity. Source this before `docker compose up`:",
        "#   set -a && . ./secrets/compose.env && set +a",
        "# Private material: never commit it.",
        f"APP_OPERATOR_API_KEY={operator_raw}",
        f"APP_OPERATOR_API_KEY_SHA256={operator_digest}",
    ]
    lines += [
        f"APP_{silo_id.upper()}_API_KEY={raw}" for silo_id, (raw, _) in sorted(silo_keys.items())
    ]
    lines.append("APP_SILO_API_KEY_SHA256=" + ",".join(silo_keys[key][1] for key in ids))
    path = out / "compose.env"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate federation identity key material.")
    parser.add_argument("--out", type=Path, default=Path("secrets"))
    parser.add_argument("--silos", type=int, default=5)
    args = parser.parse_args(argv)
    if args.silos < 2:
        parser.error("a federation needs at least 2 silos")

    written = generate(args.out, args.silos)
    for path in written:
        print(f"wrote {path}")
    print("\nThese files are private key material. Never commit them.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
