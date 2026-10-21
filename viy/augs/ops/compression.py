from typing import Final
from typing import override

import cv2
import numpy as np

from viy.augs.base import ImageOp
from viy.augs.base import RangeImageOp
from viy.common import ImageArray
from viy.common import ScaleValue

from ._utils import jpeg
from ._utils import jpeg2000
from ._utils import resize
from ._utils import webp


RECOMPRESSION_QUALITY_RANGE: Final = (30, 90)
DOUBLE_JPEG_QUALITIES: Final = ((90, 70), (85, 60), (80, 50), (75, 40), (70, 30))
JPEG2000_RATES: Final = (16.0, 32.0, 45.0, 120.0, 170.0)
JPEG_AI_PROXY_QUALITIES: Final = (18, 28, 42, 58)
CHENG2020_PROXY_PARAMS: Final = (
    (0.5, 20),
    (0.61, 30),
    (0.72, 40),
    (0.82, 50),
    (0.92, 62),
)
MIN_SIDE: Final = 8


def _random_quality(rng: np.random.Generator) -> int:
    low, high = RECOMPRESSION_QUALITY_RANGE
    return int(rng.integers(low, high, endpoint=True))


def _check_index(name: str, level: ScaleValue, size: int, offset: int = 0) -> None:
    if not offset <= int(level) < size + offset:
        raise ValueError(
            f"'{name}' level must be in [{offset}, {size + offset - 1}], got {level}."
        )
    return


def _jpeg_ai_proxy(image: ImageArray, level: int) -> ImageArray:
    return webp(image, JPEG_AI_PROXY_QUALITIES[level])


class JPEG(ImageOp):
    """JPEG compression with quality ``level``."""

    name = "jpeg"
    default_levels = (7, 12, 28, 38, 45)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return jpeg(image, int(level))


class JPEG2000(ImageOp):
    """JPEG 2000 compression with compression ratio ``level``."""

    name = "jpeg2000"
    default_levels = JPEG2000_RATES

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return jpeg2000(image, float(level))


class JPEGRecompression(ImageOp):
    """``level`` consecutive JPEG compressions with random qualities."""

    name = "jpeg_recompression_1"
    default_levels = (2, 3, 4, 5)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        for _ in range(int(level)):
            image = jpeg(image, _random_quality(rng))
        return image


class MisalignedDoubleJPEG(ImageOp):
    """Two JPEG compressions with a random block grid shift in between; stronger with ``level``."""

    name = "jpeg_recompression_2"
    default_levels = (0, 1, 2, 3, 4)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        _check_index(self.name, level, len(DOUBLE_JPEG_QUALITIES))
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        first, second = DOUBLE_JPEG_QUALITIES[int(level)]
        compressed = jpeg(image, first)
        dy, dx = (int(value) for value in rng.integers(1, 8, size=2))
        if min(compressed.shape[0] - dy, compressed.shape[1] - dx) >= MIN_SIDE:
            compressed = np.ascontiguousarray(compressed[dy:, dx:])
        return jpeg(compressed, second)


class JPEGJPEG2000Recompression(ImageOp):
    """``level`` compressions alternating JPEG and JPEG 2000."""

    name = "jpeg_recompression_comb"
    default_levels = (2, 3, 4, 5)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        for index in range(int(level)):
            if index % 2:
                image = jpeg2000(
                    image, JPEG2000_RATES[int(rng.integers(len(JPEG2000_RATES)))]
                )
            else:
                image = jpeg(image, _random_quality(rng))
        return image


class JPEGJPEGAIRecompressionProxy(ImageOp):
    """``level`` compressions alternating JPEG and the JPEG AI proxy."""

    name = "jpeg_recompression_comb_jpegai_proxy"
    proxy_of = "JPEG recompression combined with JPEG AI"
    default_levels = (2, 3, 4, 5)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        for index in range(int(level)):
            if index % 2:
                image = _jpeg_ai_proxy(
                    image, int(rng.integers(len(JPEG_AI_PROXY_QUALITIES)))
                )
            else:
                image = jpeg(image, _random_quality(rng))
        return image


class JPEGAIProxy(ImageOp):
    """
    WebP stand-in for the JPEG AI learned codec.

    Levels follow NTIRE: ``0`` is the lowest bitrate of 0.12 bpp and ``3`` the highest of 0.75 bpp.
    """

    name = "jpeg_ai_proxy"
    proxy_of = "the JPEG AI learned codec"
    default_levels = (0, 1, 2, 3)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        _check_index(self.name, level, len(JPEG_AI_PROXY_QUALITIES))
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        return _jpeg_ai_proxy(image, int(level))


class Cheng2020Proxy(ImageOp):
    """
    Resampling plus WebP stand-in for the Cheng2020 learned codec.

    Levels follow the CompressAI quality index: ``1`` is the lowest bitrate.
    """

    name = "cheng2020_proxy"
    proxy_of = "the Cheng2020 learned codec"
    default_levels = (1, 2, 3, 4, 5)
    level_type = int

    @override
    def validate_level(self, level: ScaleValue) -> None:
        _check_index(self.name, level, len(CHENG2020_PROXY_PARAMS), offset=1)
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        scale, quality = CHENG2020_PROXY_PARAMS[int(level) - 1]
        height, width = image.shape[:2]
        small = resize(
            image, max(MIN_SIDE, int(width * scale)), max(MIN_SIDE, int(height * scale))
        )
        return webp(resize(small, width, height), quality)


class Pixelate(ImageOp):
    """Downsample by ``1 - level`` with area averaging and upsample back with nearest neighbours."""

    name = "pixelate"
    default_levels = (0.01, 0.05, 0.1, 0.2, 0.5)

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if not 0.0 <= float(level) < 1.0:
            raise ValueError(f"'{self.name}' level must be in [0, 1), got {level}.")
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        height, width = image.shape[:2]
        factor = 1.0 - float(level)
        small_size = (max(1, round(width * factor)), max(1, round(height * factor)))
        small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        return np.asarray(restored, dtype=np.uint8)


class Downscale(RangeImageOp):
    """Reduce the resolution by a factor drawn from ``[low, high]``; the output is smaller."""

    name = "downscale"
    default_levels = (0.3, 0.8)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        height, width = image.shape[:2]
        factor = float(level)
        return resize(
            image,
            max(MIN_SIDE, round(width * factor)),
            max(MIN_SIDE, round(height * factor)),
        )
