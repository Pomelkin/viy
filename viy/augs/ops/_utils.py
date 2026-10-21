import io
from typing import Any
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from viy.common import ImageArray


type FloatImage = npt.NDArray[np.float32]

LUMA_WEIGHTS: Final = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_float(image: ImageArray) -> FloatImage:
    """Convert to ``float32`` keeping the ``[0, 255]`` range."""
    return image.astype(np.float32)


def to_uint8(array: npt.NDArray[np.floating[Any]]) -> ImageArray:
    """Round, clip to ``[0, 255]`` and convert to ``uint8``."""
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def luminance(image: FloatImage) -> FloatImage:
    """Luma of an RGB image with a trailing channel axis of size one."""
    return (image @ LUMA_WEIGHTS)[..., None]


def encode_decode(image: ImageArray, format_name: str, **params: Any) -> ImageArray:
    """Round-trip ``image`` through an in-memory codec supported by Pillow."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format=format_name, **params)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()


def jpeg(image: ImageArray, quality: int) -> ImageArray:
    """JPEG round trip."""
    return encode_decode(image, "JPEG", quality=quality)


def webp(image: ImageArray, quality: int) -> ImageArray:
    """WebP round trip."""
    return encode_decode(image, "WEBP", quality=quality, method=0)


def jpeg2000(image: ImageArray, rate: float) -> ImageArray:
    """JPEG 2000 round trip at compression ratio ``rate``."""
    return encode_decode(
        image,
        "JPEG2000",
        quality_mode="rates",
        quality_layers=[rate],
        irreversible=True,
    )


def resize(image: ImageArray, width: int, height: int) -> ImageArray:
    """Antialiased bicubic resize."""
    resized = Image.fromarray(image).resize(
        (max(1, width), max(1, height)), Image.Resampling.BICUBIC
    )
    return np.asarray(resized, dtype=np.uint8).copy()


def filter2d(image: ImageArray, kernel: npt.NDArray[np.float32]) -> ImageArray:
    """Convolve every channel with ``kernel`` using reflected borders."""
    filtered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    return np.asarray(filtered, dtype=np.uint8)


def gaussian_blur(image: ImageArray, sigma: float) -> ImageArray:
    """Gaussian blur with standard deviation ``sigma`` in pixels."""
    blurred = cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101
    )
    return np.asarray(blurred, dtype=np.uint8)


def low_frequency_field(
    rng: np.random.Generator, height: int, width: int, cells: int
) -> npt.NDArray[np.float32]:
    """Smooth random field in ``[-1, 1]`` of shape ``(height, width, 3)``."""
    coarse = rng.normal(0.0, 1.0, size=(cells, cells, 3)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    field = np.asarray(field, dtype=np.float32).reshape(height, width, 3)
    return field / max(float(np.abs(field).max()), 1e-6)


def edge_sign(image: ImageArray) -> npt.NDArray[np.float32]:
    """Sign of the centred edge magnitude, shape ``(H, W, 1)``."""
    gray = luminance(to_float(image))[..., 0]
    edges = np.abs(np.asarray(cv2.Laplacian(gray, cv2.CV_32F), dtype=np.float32))
    return np.sign(edges - edges.mean())[..., None].astype(np.float32)
