from collections.abc import Callable
from types import MappingProxyType
from typing import Final
from typing import override

import numpy as np
import numpy.typing as npt

from viy.augs.base import ImageOp
from viy.common import ImageArray
from viy.common import ScaleValue

from ._utils import edge_sign
from ._utils import low_frequency_field
from ._utils import to_float
from ._utils import to_uint8


type Carrier = Callable[
    [npt.NDArray[np.float32], npt.NDArray[np.float32], np.random.Generator],
    npt.NDArray[np.float32],
]

WATERMARK_AMPLITUDE: Final = (2.0, 7.0)
CROSS_CHANNEL_LEAK: Final = 0.35


def _sparse_keyed(
    xx: npt.NDArray[np.float32], yy: npt.NDArray[np.float32], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    del yy
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=xx.shape)


def _block_masked(
    xx: npt.NDArray[np.float32], yy: npt.NDArray[np.float32], rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    blocks = rng.random((int(yy.max()) // 16 + 1, int(xx.max()) // 16 + 1)) < 0.5
    mask = blocks[(yy // 16).astype(int), (xx // 16).astype(int)]
    return _sparse_keyed(xx, yy, rng) * mask


def _periodic(
    pattern: Callable[
        [npt.NDArray[np.float32], npt.NDArray[np.float32]], npt.NDArray[np.float32]
    ],
) -> Carrier:
    def carrier(
        xx: npt.NDArray[np.float32],
        yy: npt.NDArray[np.float32],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float32]:
        del rng
        return pattern(xx, yy).astype(np.float32)

    return carrier


def _radial(
    xx: npt.NDArray[np.float32], yy: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    radius = np.sqrt((xx - xx.mean()) ** 2 + (yy - yy.mean()) ** 2)
    return np.sin(2 * np.pi * radius / 11.0)


WATERMARK_CARRIERS: Final[MappingProxyType[str, Carrier]] = MappingProxyType(
    {
        "pixelseal": _sparse_keyed,
        "maskwm": _block_masked,
        "dct_marker": _periodic(
            lambda x, y: np.cos(2 * np.pi * x / 8) * np.cos(2 * np.pi * y / 8)
        ),
        "mbrs": _periodic(
            lambda x, y: (
                0.6 * np.sin(2 * np.pi * x / 13) + 0.4 * np.sin(2 * np.pi * y / 19)
            )
        ),
        "pimog": _periodic(lambda x, y: np.sin(2 * np.pi * (x + y) / 17)),
        "sshidden": _periodic(lambda x, y: np.sin(2 * np.pi * (x - y) / 23)),
        "stega_stamp": _periodic(lambda x, y: ((x // 4 + y // 4) % 2) * 2.0 - 1.0),
        "syncseal": _periodic(
            lambda x, y: 0.5 * (np.cos(2 * np.pi * x / 16) + np.cos(2 * np.pi * y / 16))
        ),
        "fin": _periodic(
            lambda x, y: np.sin(2 * np.pi * x / 7) * np.sin(2 * np.pi * y / 7)
        ),
        "arwgan": _periodic(_radial),
    }
)


class AdvEmbedCLIPProxy(ImageOp):
    """Random low-frequency sign noise of L-inf radius ``level`` in place of a CLIP embedding attack."""

    name = "adv_embed_clip_proxy"
    proxy_of = "an adversarial attack on CLIP image embeddings"
    default_levels = (4, 6, 8, 10, 12)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        height, width = image.shape[:2]
        field = np.sign(low_frequency_field(rng, height, width, cells=12))
        return to_uint8(to_float(image) + float(level) * field)


class AdvEmbedResNetProxy(ImageOp):
    """Edge-aligned sign noise of L-inf radius ``level`` in place of a ResNet embedding attack."""

    name = "adv_embed_resnet_proxy"
    proxy_of = "an adversarial attack on ResNet embeddings"
    default_levels = (4, 6, 8, 10, 12)
    level_type = int

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        channel_signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=3)
        return to_uint8(
            to_float(image) + float(level) * edge_sign(image) * channel_signs
        )


class WMForgerProxy(ImageOp):
    """Low-frequency plus edge residual in place of the WMForger removal attack with learning rate ``level``."""

    name = "wmforger_proxy"
    proxy_of = "the WMForger watermark removal attack"
    default_levels = (0.05, 0.1, 0.2, 0.5, 1.0)

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        height, width = image.shape[:2]
        epsilon = 2.0 + 7.0 * float(level)
        residual = 0.7 * low_frequency_field(
            rng, height, width, cells=8
        ) + 0.3 * edge_sign(image)
        return to_uint8(to_float(image) + epsilon * residual)


class WatermarkProxy(ImageOp):
    """Carrier pattern named by ``level`` added to one channel in place of an invisible watermark."""

    name = "watermark_proxy"
    proxy_of = "neural invisible watermarking methods"
    default_levels = (
        "arwgan",
        "dct_marker",
        "mbrs",
        "pimog",
        "pixelseal",
        "stega_stamp",
    )
    level_type = str

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if level not in WATERMARK_CARRIERS:
            raise ValueError(
                f"Unknown watermark '{level}'. Available: {', '.join(sorted(WATERMARK_CARRIERS))}."
            )
        return

    @override
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        pattern = WATERMARK_CARRIERS[str(level)](xx, yy, rng)
        amplitude = float(rng.uniform(*WATERMARK_AMPLITUDE))
        channel = int(rng.integers(3))
        result = to_float(image)
        result[..., channel] += amplitude * pattern
        result[..., (channel + 1) % 3] -= CROSS_CHANNEL_LEAK * amplitude * pattern
        return to_uint8(result)
