from __future__ import annotations

import os
from typing import Any, Optional


def add_prefix_to_dict(
    d: dict[str, Any], prefix: str, sep: Optional[str] = "/"
) -> dict[str, Any]:
    """Add a prefix to each key of a dictionary.

    Parameters
    ----------
    d : dict[str, Any]
        A dictionary.
    prefix : str
        Prefix to be prepended to each key.
    sep : str, optional
        Separator between `prefix` and key, by default "/".

    Returns
    -------
    dict[str, Any]
        New dictionary with `prefix` prepended to each key of `d`.
    """
    return {f"{prefix}{sep}{k}": v for k, v in d.items()}


def get_project_root_dir() -> str:
    """Get the absolute path to the project root.

    Returns
    -------
    str
        Absolute path to the project root.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def to_abs_path(s: str) -> str:
    """Convert a path string to an absolute path.

    If `s` is already an absolute path, it is returned unchanged.
    Otherwise, it is interpreted as relative to the project root directory.

    Parameters
    ----------
    s : str
        Path string, either absolute or relative to the project root.

    Returns
    -------
    str
        Absolute path corresponding to `s`.
    """
    return s if os.path.isabs(s) else os.path.join(get_project_root_dir(), s)
