"""Publish complete artifact bytes with atomic replacement."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Flush complete bytes before an atomic replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Encode and publish one UTF-8 text artifact."""
    atomic_write_bytes(path, content.encode("utf-8"))


def fsync_directory(path: Path) -> None:
    """Flush the directory entry when the platform supports it."""
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
