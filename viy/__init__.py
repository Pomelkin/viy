from importlib.metadata import version

from . import augs
from . import bench
from . import data
from . import datasets
from . import layout
from .common import load_config


__version__ = version("viy")

__all__ = [
    "__version__",
    "augs",
    "bench",
    "data",
    "datasets",
    "layout",
    "load_config",
]
