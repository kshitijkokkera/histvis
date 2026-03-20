"""utils.py - Shared utilities for histvis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_npy(path: Union[str, Path]) -> np.ndarray:
    """Load a .npy file and return it as a float32 array."""
    arr = np.load(str(path))
    logger.debug("Loaded %s  shape=%s  dtype=%s", path, arr.shape, arr.dtype)
    return arr.astype(np.float32)


def load_markers(run_dir: Path) -> List[str]:
    """Load marker names from markers.txt in the given run directory."""
    markers_file = run_dir / "markers.txt"
    if markers_file.exists():
        return [m.strip() for m in markers_file.read_text().splitlines() if m.strip()]
    return []
