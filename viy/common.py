from pathlib import Path
from typing import Any
from typing import Final

import numpy as np
import numpy.typing as npt
from kostyl.utils import load_file


type ImageArray = npt.NDArray[np.uint8]
"""RGB image of shape ``(H, W, 3)`` with ``uint8`` values."""

type ScaleValue = int | float | str
"""Severity of a distortion in the NTIRE ``distortion_scales`` convention."""

REAL_LABEL: Final = 0
FAKE_LABEL: Final = 1
LABELS: Final = frozenset({REAL_LABEL, FAKE_LABEL})


def load_config(path: Path | str) -> dict[str, Any]:
    """
    Read a YAML or JSON config file into a dictionary.

    Example:
        >>> config = BenchmarkConfig.model_validate(load_config("bench.yaml"))
    """
    return load_file(path)
