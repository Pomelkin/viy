from typing import Final
from typing import override

import cv2
import numpy as np

from viy.augs.base import ImageOp
from viy.common import ImageArray
from viy.common import ScaleValue

from ._utils import to_float
from ._utils import to_uint8


ISO_NOISE_PARAMS: Final = (
    (0.01, 0.1),
    (0.02, 0.2),
    (0.03, 0.3),
    (0.04, 0.4),
    (0.05, 0.5),
)
SHOT_NOISE_PHOTONS: Final = (60.0, 25.0, 12.0, 5.0, 3.0)


class WhiteNoise(ImageOp):
    """Additive Gaussian noise with variance ``level`` on the ``[0, 1]`` intensity scale."""

    name = "whitenoise"
    default_levels = (0.001, 0.002, 0.0025, 0.003, 0.005)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        noise = rng.normal(0.0, np.sqrt(float(level)) * 255.0, size=image.shape)
        return to_uint8(to_float(image) + noise)


class ImpulseNoise(ImageOp):
    """Salt and pepper noise replacing a ``level`` fraction of pixels."""

    name = "impulsenoise"
    default_levels = (0.001, 0.005, 0.01, 0.015, 0.02)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        amount = float(level)
        draws = rng.random(image.shape[:2])
        noisy = image.copy()
        noisy[draws < amount / 2] = 0
        noisy[(draws >= amount / 2) & (draws < amount)] = 255
        return noisy


class MultiplicativeNoise(ImageOp):
    """Speckle noise ``x * (1 + n)`` with ``n`` of variance ``level``."""

    name = "multnoise"
    default_levels = (0.001, 0.005, 0.01, 0.015, 0.035)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        noise = rng.normal(0.0, np.sqrt(float(level)), size=image.shape)
        return to_uint8(to_float(image) * (1.0 + noise))


class ISONoise(ImageOp):
    """Camera sensor noise: Poisson luminance noise and Gaussian hue noise in HLS space."""

    name = "isonoise"
    default_levels = (0, 1, 2, 3, 4)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if not 0 <= int(level) < len(ISO_NOISE_PARAMS):
            raise ValueError(
                f"'{self.name}' level must be in [0, {len(ISO_NOISE_PARAMS) - 1}]."
            )
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        color_shift, intensity = ISO_NOISE_PARAMS[int(level)]
        hls = np.asarray(
            cv2.cvtColor(to_float(image) / 255.0, cv2.COLOR_RGB2HLS), dtype=np.float32
        )
        lightness_std = float(hls[..., 1].std())
        luminance_noise = rng.poisson(
            lightness_std * intensity * 255.0, size=hls.shape[:2]
        )
        hue_noise = rng.normal(0.0, color_shift * 360.0 * intensity, size=hls.shape[:2])
        hls[..., 0] = np.mod(hls[..., 0] + hue_noise, 360.0)
        hls[..., 1] += luminance_noise / 255.0 * (1.0 - hls[..., 1])
        rgb = np.asarray(
            cv2.cvtColor(np.clip(hls, 0.0, 360.0), cv2.COLOR_HLS2RGB), dtype=np.float32
        )
        return to_uint8(rgb * 255.0)


class ShotNoise(ImageOp):
    """Poisson photon noise with the ImageNet-C photon counts per severity."""

    name = "shotnoise"
    default_levels = (0, 1, 2, 3, 4)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if not 0 <= int(level) < len(SHOT_NOISE_PHOTONS):
            raise ValueError(
                f"'{self.name}' level must be in [0, {len(SHOT_NOISE_PHOTONS) - 1}]."
            )
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        photons = SHOT_NOISE_PHOTONS[int(level)]
        noisy = rng.poisson(to_float(image) / 255.0 * photons) / photons
        return to_uint8(noisy * 255.0)
