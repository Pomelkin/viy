from typing import Final
from typing import override

import cv2
import numpy as np

from viy.augs.base import ImageOp
from viy.common import ImageArray
from viy.common import ScaleValue


PERSPECTIVE_FRACTIONS: Final = (0.02, 0.04, 0.065, 0.09, 0.12)
MAX_ASPECT_CHANGE: Final = 2.0


def _crop(
    image: ImageArray, crop_width: int, crop_height: int, rng: np.random.Generator
) -> ImageArray:
    height, width = image.shape[:2]
    crop_width = int(np.clip(crop_width, 1, width))
    crop_height = int(np.clip(crop_height, 1, height))
    left = int(rng.integers(0, width - crop_width, endpoint=True))
    top = int(rng.integers(0, height - crop_height, endpoint=True))
    return np.ascontiguousarray(
        image[top : top + crop_height, left : left + crop_width]
    )


def _check_area_fraction(name: str, level: ScaleValue) -> None:
    if not 0.0 < float(level) <= 1.0:
        raise ValueError(
            f"'{name}' level is a kept area fraction in (0, 1], got {level}."
        )
    return


class RandomCrop(ImageOp):
    """Crop a random window keeping a ``level`` fraction of the area and the aspect ratio."""

    name = "randomcrop"
    default_levels = (0.4, 0.5, 0.6, 0.7, 0.8)

    @override
    def validate_level(self, level: ScaleValue) -> None:
        _check_area_fraction(self.name, level)
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        side = float(np.sqrt(float(level)))
        height, width = image.shape[:2]
        return _crop(image, round(width * side), round(height * side), rng)


class RandomAspectCrop(ImageOp):
    """Crop a random window keeping a ``level`` fraction of the area with a changed aspect ratio."""

    name = "randomaspectcrop"
    default_levels = (0.4, 0.5, 0.6, 0.7, 0.8)

    @override
    def validate_level(self, level: ScaleValue) -> None:
        _check_area_fraction(self.name, level)
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        aspect = float(
            np.exp(rng.uniform(-np.log(MAX_ASPECT_CHANGE), np.log(MAX_ASPECT_CHANGE)))
        )
        height, width = image.shape[:2]
        crop_width = round(width * np.sqrt(float(level) * aspect))
        crop_height = round(height * np.sqrt(float(level) / aspect))
        return _crop(image, crop_width, crop_height, rng)


class Perspective(ImageOp):
    """Map a random quadrilateral inside the image onto the full frame."""

    name = "perspective"
    default_levels = (0, 1, 2, 3, 4)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if not 0 <= int(level) < len(PERSPECTIVE_FRACTIONS):
            raise ValueError(
                f"'{self.name}' level must be in [0, {len(PERSPECTIVE_FRACTIONS) - 1}]."
            )
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        height, width = image.shape[:2]
        fraction = PERSPECTIVE_FRACTIONS[int(level)]
        corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )
        inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32)
        jitter = rng.uniform(0.0, fraction, size=(4, 2)) * np.array([width, height])
        source = (corners + inward * jitter).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(source, corners)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return np.asarray(warped, dtype=np.uint8)
