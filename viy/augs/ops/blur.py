from typing import override

import cv2
import numpy as np

from viy.augs.base import ImageOp
from viy.common import ImageArray
from viy.common import ScaleValue

from ._utils import filter2d
from ._utils import gaussian_blur


class GaussianBlur(ImageOp):
    """Gaussian blur with standard deviation ``level`` pixels."""

    name = "gausblur"
    default_levels = (0.1, 0.5, 1.0, 2.0, 5.0)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return gaussian_blur(image, float(level))


class LensBlur(ImageOp):
    """Defocus blur with a uniform disk kernel of radius ``level`` pixels."""

    name = "lensblur"
    default_levels = (1, 2, 4, 6, 8)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        radius = int(level)
        coords = np.arange(-radius, radius + 1, dtype=np.float32)
        disk = (coords[None, :] ** 2 + coords[:, None] ** 2 <= radius**2).astype(
            np.float32
        )
        return filter2d(image, disk / disk.sum())


class MotionBlur(ImageOp):
    """Linear motion blur over ``2 * level + 1`` pixels at a random angle."""

    name = "motionblur"
    default_levels = (1, 2, 4, 6, 10)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        radius = int(level)
        size = 2 * radius + 1
        angle = rng.uniform(0.0, np.pi)
        dx, dy = radius * np.cos(angle), radius * np.sin(angle)
        kernel = np.zeros((size, size), dtype=np.float32)
        start = (round(radius - dx), round(radius - dy))
        end = (round(radius + dx), round(radius + dy))
        kernel = np.asarray(
            cv2.line(kernel, start, end, color=1.0, thickness=1), dtype=np.float32
        )
        return filter2d(image, kernel / kernel.sum())


class GlassBlur(ImageOp):
    """Frosted glass: blur, displace every pixel by up to ``level`` pixels, blur again."""

    name = "glassblur"
    default_levels = (1, 2, 3, 4, 6)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        delta = int(level)
        sigma = 0.5 + 0.1 * delta
        height, width = image.shape[:2]
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        offset_x = rng.integers(-delta, delta, size=(height, width), endpoint=True)
        offset_y = rng.integers(-delta, delta, size=(height, width), endpoint=True)
        map_x = np.clip(grid_x + offset_x, 0, width - 1).astype(np.float32)
        map_y = np.clip(grid_y + offset_y, 0, height - 1).astype(np.float32)
        blurred = gaussian_blur(image, sigma)
        displaced = np.asarray(
            cv2.remap(blurred, map_x, map_y, interpolation=cv2.INTER_NEAREST),
            dtype=np.uint8,
        )
        return gaussian_blur(displaced, sigma)
