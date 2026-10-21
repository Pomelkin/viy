from typing import Final
from typing import override

import cv2
import numpy as np

from viy.augs.base import ImageOp
from viy.common import ImageArray
from viy.common import ScaleValue

from ._utils import FloatImage
from ._utils import luminance
from ._utils import to_float
from ._utils import to_uint8


CLAHE_CLIP_LIMITS: Final = (1.0, 2.0, 3.0, 4.0, 6.0)


def _brightness(image: FloatImage, factor: float) -> FloatImage:
    return image * factor


def _contrast(image: FloatImage, factor: float) -> FloatImage:
    mean = float(luminance(image).mean())
    return mean + (image - mean) * factor


def _saturation(image: FloatImage, factor: float) -> FloatImage:
    gray = luminance(image)
    return gray + (image - gray) * factor


class Brighten(ImageOp):
    """Scale intensities by ``1 + level``."""

    name = "brighten"
    default_levels = (0.1, 0.2, 0.4, 0.7, 1.1)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return to_uint8(_brightness(to_float(image), 1.0 + float(level)))


class Darken(ImageOp):
    """Scale intensities by ``1 - level``."""

    name = "darken"
    default_levels = (0.05, 0.1, 0.2, 0.4, 0.8)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return to_uint8(_brightness(to_float(image), 1.0 - float(level)))


class LinearContrastChange(ImageOp):
    """Scale deviations from the mean luminance by ``1 + level``."""

    name = "lincontrchange"
    default_levels = (-0.6, -0.4, 0.0, 0.15, 0.3)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return to_uint8(_contrast(to_float(image), 1.0 + float(level)))


class ColorSaturation(ImageOp):
    """Scale deviations from the per-pixel luminance by ``1 + level``."""

    name = "colorsat"
    default_levels = (-0.4, 0.0, 0.1, 0.2, 0.4)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return to_uint8(_saturation(to_float(image), 1.0 + float(level)))


class ColorShift(ImageOp):
    """Misalign the red and blue channels by ``level`` pixels in opposite directions."""

    name = "colorshift"
    default_levels = (1, 3, 6, 8, 12)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        dx, dy = float(level) * np.cos(angle), float(level) * np.sin(angle)
        height, width = image.shape[:2]
        shifted = image.copy()
        for channel, sign in ((0, 1.0), (2, -1.0)):
            matrix = np.array(
                [[1.0, 0.0, sign * dx], [0.0, 1.0, sign * dy]], dtype=np.float32
            )
            moved = cv2.warpAffine(
                image[..., channel],
                matrix,
                (width, height),
                borderMode=cv2.BORDER_REPLICATE,
            )
            shifted[..., channel] = np.asarray(moved, dtype=np.uint8)
        return shifted


class RGBShift(ImageOp):
    """Add a random offset from ``[-level, level]`` to every channel."""

    name = "rgbshift"
    default_levels = (10, 20, 30, 40, 50)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        offsets = rng.integers(-int(level), int(level), size=3, endpoint=True)
        return to_uint8(to_float(image) + offsets.astype(np.float32))


class ColorJitter(ImageOp):
    """Brightness, contrast and saturation factors from ``[1 - level, 1 + level]`` in random order."""

    name = "jitter"
    default_levels = (0.05, 0.1, 0.2, 0.5, 1.0)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        result = to_float(image)
        adjustments = (_brightness, _contrast, _saturation)
        low, high = max(0.0, 1.0 - float(level)), 1.0 + float(level)
        for index in rng.permutation(len(adjustments)):
            result = adjustments[int(index)](result, float(rng.uniform(low, high)))
        return to_uint8(result)


class ColorQuantization(ImageOp):
    """Quantize every channel to ``level`` evenly spaced values."""

    name = "quantization"
    default_levels = (7, 10, 13, 16, 20)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if int(level) < 2:
            raise ValueError(
                f"'{self.name}' needs at least 2 quantization levels, got {level}."
            )
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        steps = int(level) - 1
        return to_uint8(np.rint(to_float(image) / 255.0 * steps) * (255.0 / steps))


class RandomToneCurve(ImageOp):
    """Cubic Bezier tone curve with control points drawn around 0.25 and 0.75 with spread ``level``."""

    name = "randomtonecurve"
    default_levels = (0.05, 0.15, 0.2, 0.3, 0.4)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        low_y = float(np.clip(rng.normal(0.25, float(level)), 0.0, 1.0))
        high_y = float(np.clip(rng.normal(0.75, float(level)), 0.0, 1.0))
        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        curve = 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3
        lut = to_uint8(curve * 255.0)
        return lut[image]


class CLAHE(ImageOp):
    """Contrast limited adaptive histogram equalization of the lightness channel."""

    name = "clahe"
    default_levels = (0, 1, 2, 3, 4)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if not 0 <= int(level) < len(CLAHE_CLIP_LIMITS):
            raise ValueError(
                f"'{self.name}' level must be in [0, {len(CLAHE_CLIP_LIMITS) - 1}]."
            )
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMITS[int(level)], tileGridSize=(8, 8)
        )
        lab = np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2LAB), dtype=np.uint8)
        lab[..., 0] = np.asarray(clahe.apply(lab[..., 0]), dtype=np.uint8)
        return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), dtype=np.uint8)
