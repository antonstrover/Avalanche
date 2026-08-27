"""Normalize repository-owned logical paths."""

from pathlib import PurePosixPath, PureWindowsPath


def canonical_repository_path(value: str, description: str) -> str:
    """Return one repository-relative POSIX path."""
    text = str(value).replace("\\", "/")
    windows = PureWindowsPath(text)
    path = PurePosixPath(text)
    if not text or path.is_absolute() or windows.drive or not path.parts:
        raise ValueError(f"the {description} path must be repository-relative")
    if ".." in path.parts:
        raise ValueError(f"the {description} path must not traverse a parent")
    return path.as_posix()
