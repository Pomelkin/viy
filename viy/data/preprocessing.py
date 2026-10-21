from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
import torch
from kostyl import StrictPydanticBaseModel
from PIL import Image
from pydantic import Field

from viy.common import ImageArray


type RGB = tuple[float, float, float]

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)


class ResizeMode(StrEnum):
    """How an image is brought to a square of ``image_size``."""

    RESIZE = "resize"
    RESIZE_CENTER_CROP = "resize_center_crop"
    SHORTER_SIDE_CENTER_CROP = "shorter_side_center_crop"


class Interpolation(StrEnum):
    """Resampling filter; every filter except ``nearest`` is antialiased."""

    BICUBIC = "bicubic"
    BILINEAR = "bilinear"
    LANCZOS = "lanczos"
    BOX = "box"
    NEAREST = "nearest"


_RESAMPLING: Final = MappingProxyType(
    {
        Interpolation.BICUBIC: Image.Resampling.BICUBIC,
        Interpolation.BILINEAR: Image.Resampling.BILINEAR,
        Interpolation.LANCZOS: Image.Resampling.LANCZOS,
        Interpolation.BOX: Image.Resampling.BOX,
        Interpolation.NEAREST: Image.Resampling.NEAREST,
    }
)


class PreprocessingConfig(StrictPydanticBaseModel):
    """
    Conversion of a ``uint8`` RGB image into a normalized model input.

    Attributes:
        image_size (int): Side of the square output.
        resize_mode (ResizeMode): ``resize`` squashes to the output size;
            ``resize_center_crop`` squashes to ``image_size + crop_padding`` and
            crops the centre; ``shorter_side_center_crop`` keeps the aspect
            ratio, scales the shorter side to ``image_size + crop_padding`` and
            crops the centre.
        crop_padding (int): Margin removed by the centre crop.
        interpolation (Interpolation): Resampling filter.
        mean (tuple[float, float, float]): Per-channel mean on the ``[0, 1]`` scale.
        std (tuple[float, float, float]): Per-channel standard deviation on the ``[0, 1]`` scale.
    """

    image_size: int = Field(gt=0)
    resize_mode: ResizeMode = ResizeMode.RESIZE
    crop_padding: int = Field(default=32, ge=0)
    interpolation: Interpolation = Interpolation.BICUBIC
    mean: RGB = IMAGENET_MEAN
    std: RGB = IMAGENET_STD


class Preprocessor:
    """Resize, crop, scale to ``[0, 1]`` and normalize an RGB image into a ``(3, H, W)`` tensor."""

    def __init__(
        self,
        image_size: int,
        resize_mode: ResizeMode | str = ResizeMode.RESIZE,
        crop_padding: int = 32,
        interpolation: Interpolation | str = Interpolation.BICUBIC,
        mean: RGB = IMAGENET_MEAN,
        std: RGB = IMAGENET_STD,
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            image_size (int): Side of the square output.
            resize_mode (ResizeMode | str): How the image is brought to the output size.
            crop_padding (int): Margin removed by the centre crop.
            interpolation (Interpolation | str): Resampling filter.
            mean (RGB): Per-channel mean on the ``[0, 1]`` scale.
            std (RGB): Per-channel standard deviation on the ``[0, 1]`` scale.
        """
        if any(value <= 0 for value in std):
            raise ValueError(f"Standard deviations must be positive, got {std}.")
        self.image_size = image_size
        self.resize_mode = ResizeMode(resize_mode)
        self.crop_padding = crop_padding
        self.resample = _RESAMPLING[Interpolation(interpolation)]
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        return

    @classmethod
    def from_pydantic_model(cls, config: PreprocessingConfig) -> "Preprocessor":
        """Build the preprocessor from its configuration."""
        return cls(
            image_size=config.image_size,
            resize_mode=config.resize_mode,
            crop_padding=config.crop_padding,
            interpolation=config.interpolation,
            mean=config.mean,
            std=config.std,
        )

    def __call__(self, image: ImageArray | Image.Image) -> torch.Tensor:
        """
        Convert ``image`` into a normalized float tensor.

        Args:
            image (ImageArray | Image.Image): RGB image.

        Returns:
            torch.Tensor: Tensor of shape ``(3, image_size, image_size)``.
        """
        pil_image = (
            image.convert("RGB")
            if isinstance(image, Image.Image)
            else Image.fromarray(image)
        )
        array = np.asarray(self._resize(pil_image), dtype=np.uint8)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self.mean) / self.std

    def _resize(self, image: Image.Image) -> Image.Image:
        size = self.image_size
        padded = size + self.crop_padding
        match self.resize_mode:
            case ResizeMode.RESIZE:
                return image.resize((size, size), self.resample)
            case ResizeMode.RESIZE_CENTER_CROP:
                return _center_crop(image.resize((padded, padded), self.resample), size)
            case ResizeMode.SHORTER_SIDE_CENTER_CROP:
                scale = padded / min(image.size)
                width, height = (max(size, round(side * scale)) for side in image.size)
                return _center_crop(image.resize((width, height), self.resample), size)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))
