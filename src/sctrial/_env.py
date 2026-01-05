from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["ensure_numba_cache_dir", "ensure_matplotlib_config_dir"]


def _ensure_dir(env_var: str, subdir: str) -> str:
    """Ensure an environment variable points to a writable directory."""
    current = os.environ.get(env_var)
    if current:
        Path(current).mkdir(parents=True, exist_ok=True)
        return current

    path = Path(tempfile.gettempdir()) / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[env_var] = str(path)
    return str(path)


def ensure_numba_cache_dir() -> str:
    """Guarantee NUMBA_CACHE_DIR is set to a writable location."""
    return _ensure_dir("NUMBA_CACHE_DIR", "sctrial-numba-cache")


def ensure_matplotlib_config_dir() -> str:
    """Guarantee MPLCONFIGDIR is set to a writable location."""
    return _ensure_dir("MPLCONFIGDIR", "sctrial-mpl-config")
