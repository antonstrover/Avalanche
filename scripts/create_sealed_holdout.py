"""Create one encrypted sealed manifest under custodian control."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from avalanche.experiments.protocols import canonical_artifact_bytes
from avalanche.experiments.sealed import instantiate_family, validate_freeze_certificate


def main(argv: list[str] | None = None) -> int:
    """Verify the freeze and encrypt ten shared opaque roots."""
    parser = argparse.ArgumentParser(prog="create_sealed_holdout")
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--recipient", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--age-binary", default="age")
    parser.add_argument("--age-version", required=True)
    parser.add_argument("--age-sha256", required=True)
    args = parser.parse_args(argv)

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    certificate_sha256 = validate_freeze_certificate(certificate)
    _verify_age(args.age_binary, args.age_version, args.age_sha256)
    root_value = json.loads(args.roots.read_text(encoding="utf-8"))
    root_ids = root_value.get("root_ids")
    if not isinstance(root_ids, list) or len(root_ids) != 10:
        raise ValueError("the sealed manifest needs ten opaque roots")
    invalid_roots = any(not isinstance(value, str) for value in root_ids)
    if len(set(root_ids)) != 10 or invalid_roots:
        raise ValueError("the sealed root identities must be unique strings")
    families = ["whiteout-r1", "cascade-r1"]
    manifest = {
        "sealed_manifest_version": 1,
        "certificate_sha256": certificate_sha256,
        "repository_revision": certificate["code_revision"],
        "root_ids": root_ids,
        "families": {
            family: [
                instantiate_family(root, family, certificate=certificate)
                for root in root_ids
            ]
            for family in families
        },
    }
    content = canonical_artifact_bytes(manifest)
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avalanche-sealed-") as directory:
        plaintext = Path(directory) / "sealed-manifest-v1.json"
        plaintext.write_bytes(content)
        result = subprocess.run(
            (
                str(args.age_binary),
                "--encrypt",
                "--recipient",
                args.recipient,
                "--output",
                str(args.output),
                str(plaintext),
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            args.output.unlink(missing_ok=True)
            raise RuntimeError("the sealed manifest encryption failed")
    commitment = {
        "manifest_sha256": manifest_sha256,
        "manifest_ciphertext_sha256": hashlib.sha256(
            args.output.read_bytes()
        ).hexdigest(),
        "family_ids": families,
        "repository_revision": certificate["code_revision"],
        "certificate_sha256": certificate_sha256,
    }
    args.output.with_suffix(args.output.suffix + ".commitment.json").write_bytes(
        canonical_artifact_bytes(commitment)
    )
    return 0


def _verify_age(binary: str, version: str, expected_sha256: str) -> None:
    """Require the exact frozen age executable before encryption."""
    found = shutil.which(binary)
    if found is None:
        raise ValueError("the frozen age binary is unavailable")
    resolved = Path(found).resolve(strict=True)
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("the age binary digest does not match the contract")
    result = subprocess.run(
        (str(resolved), "--version"), capture_output=True, text=True, check=True
    )
    actual = (result.stdout or result.stderr).strip()
    if actual != version:
        raise ValueError("the age version does not match the contract")


if __name__ == "__main__":
    raise SystemExit(main())
