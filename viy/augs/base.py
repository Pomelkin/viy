import inspect
import json
import math
from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import ClassVar
from typing import override

import numpy as np
from kostyl.utils import setup_logger

from viy.common import ImageArray
from viy.common import ScaleValue
from viy.registry import Registry


logger = setup_logger(fmt="detailed")
PROXY_SUFFIX = "_proxy"


@dataclass(frozen=True, slots=True)
class AppliedOp:
    """
    A distortion applied to an image.

    Attributes:
        name (str): Operation name.
        scale (ScaleValue | None): Sampled severity, or ``None`` for operations
            without a severity scale.
    """

    name: str
    scale: ScaleValue | None = None


def trace_to_json(trace: Sequence[AppliedOp]) -> str:
    """Serialize applied operations as a JSON list of ``{"name", "scale"}`` objects."""
    return json.dumps([{"name": op.name, "scale": op.scale} for op in trace])


OPS: Registry[type["ImageOp"]] = Registry("image operation")


class ImageOp(ABC):
    """
    Deterministic image distortion with discrete severity levels.

    Every call samples one level from ``levels`` with the given random
    generator, so a fixed seed reproduces the distortion exactly. Level values
    follow the ``distortion_scales`` convention of the NTIRE 2026 metadata.
    Operations are registered by ``name`` on class creation.

    A proxy is a fast stand-in for a distortion that needs a neural network or
    per-image optimization, such as a learned codec or an adversarial attack.
    It only imitates the magnitude and rough appearance of the real distortion.
    Proxies set ``proxy_of``, carry the ``_proxy`` name suffix and log a warning
    when created.
    """

    name: ClassVar[str]
    default_levels: ClassVar[tuple[ScaleValue, ...]]
    level_type: ClassVar[type[int] | type[float] | type[str]] = float
    proxy_of: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if cls.name.endswith(PROXY_SUFFIX) != (cls.proxy_of is not None):
            raise TypeError(
                f"Operation '{cls.name}': proxies must set proxy_of and end with '{PROXY_SUFFIX}'."
            )
        OPS.add(cls.name, cls)
        return

    def __init__(self, levels: Sequence[ScaleValue] | None = None) -> None:
        """
        Initialize the operation.

        Args:
            levels (Sequence[ScaleValue] | None): Severity levels to sample from;
                ``None`` uses ``default_levels``.
        """
        chosen = self.default_levels if levels is None else levels
        if len(chosen) == 0:
            raise ValueError(f"Operation '{self.name}' needs at least one level.")
        self.levels: tuple[ScaleValue, ...] = tuple(
            self.level_type(level) for level in chosen
        )
        for level in self.levels:
            self.validate_level(level)
        if self.proxy_of is not None:
            logger.warning_once(
                f"Augmentation '{self.name}' is a proxy of {self.proxy_of}, not a replacement: "
                "it imitates the magnitude and rough appearance of the distortion without "
                "reproducing it. Exclude it or use genuine examples when the distortion matters."
            )
        return

    @property
    def is_proxy(self) -> bool:
        """Whether the operation only approximates the distortion it is named after."""
        return self.proxy_of is not None

    def validate_level(self, level: ScaleValue) -> None:
        """Raise ``ValueError`` if ``level`` is not a valid severity."""
        if isinstance(level, float) and not math.isfinite(level):
            raise ValueError(f"Operation '{self.name}' got a non-finite level {level}.")
        return

    def sample_level(self, rng: np.random.Generator) -> ScaleValue:
        """Draw a severity level."""
        return self.levels[int(rng.integers(len(self.levels)))]

    def __call__(
        self, image: ImageArray, rng: np.random.Generator
    ) -> tuple[ImageArray, AppliedOp]:
        """
        Distort ``image`` with a randomly drawn level.

        Args:
            image (ImageArray): RGB image.
            rng (np.random.Generator): Source of randomness.

        Returns:
            tuple[ImageArray, AppliedOp]: Distorted image and the applied operation.
        """
        level = self.sample_level(rng)
        return self.apply(image, level, rng), AppliedOp(self.name, level)

    @abstractmethod
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        """
        Distort ``image`` with severity ``level``.

        Args:
            image (ImageArray): RGB image; it must not be modified in place.
            level (ScaleValue): Severity.
            rng (np.random.Generator): Source of randomness.

        Returns:
            ImageArray: Distorted RGB image, possibly of a different size.
        """
        raise NotImplementedError


class RangeImageOp(ImageOp, ABC):
    """Operation whose severity is drawn uniformly from ``[low, high]``."""

    @override
    def __init__(self, levels: Sequence[ScaleValue] | None = None) -> None:
        super().__init__(levels)
        if len(self.levels) != 2 or float(self.levels[0]) > float(self.levels[1]):
            raise ValueError(
                f"Operation '{self.name}' expects levels [low, high], got {self.levels}."
            )
        return

    @override
    def sample_level(self, rng: np.random.Generator) -> ScaleValue:
        return float(rng.uniform(float(self.levels[0]), float(self.levels[1])))
