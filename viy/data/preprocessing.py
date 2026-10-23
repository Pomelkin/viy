from collections.abc import Callable
from enum import StrEnum
from types import MappingProxyType
from typing import Final
from typing import Self

import numpy as np
import torch
from kostyl import StrictPydanticBaseModel
from PIL import Image
from pydantic import Field
from pydantic import model_validator
from transformers import AutoImageProcessor
from transformers import BaseImageProcessor

from viy.common import ImageArray


type RGB = tuple[float, float, float]

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)
DEFAULT_CROP_PADDING: Final = 32


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
    Conversion of an image into a model input: custom settings or a transformers image processor.

    Set either ``image_size``, optionally with the other custom settings, or
    ``image_processor``, never both.

    Attributes:
        image_size (int | None): Side of the square output.
        resize_mode (ResizeMode | None): ``resize`` squashes to the output size;
            ``resize_center_crop`` squashes to ``image_size + crop_padding`` and
            crops the centre; ``shorter_side_center_crop`` keeps the aspect
            ratio, scales the shorter side to ``image_size + crop_padding`` and
            crops the centre. Defaults to ``resize``.
        crop_padding (int | None): Margin removed by the centre crop; defaults to 32.
        interpolation (Interpolation | None): Resampling filter; defaults to ``bicubic``.
        mean (tuple[float, float, float] | None): Per-channel mean on the ``[0, 1]``
            scale; defaults to ImageNet.
        std (tuple[float, float, float] | None): Per-channel standard deviation on
            the ``[0, 1]`` scale; defaults to ImageNet.
        image_processor (str | None): Hub name or local path of a transformers
            image processor, loaded with ``AutoImageProcessor.from_pretrained``.
    """

    image_size: int | None = Field(default=None, gt=0)
    resize_mode: ResizeMode | None = None
    crop_padding: int | None = Field(default=None, ge=0)
    interpolation: Interpolation | None = None
    mean: RGB | None = None
    std: RGB | None = None
    image_processor: str | None = None

    @model_validator(mode="after")
    def _validate_choice(self) -> Self:
        custom = (
            self.resize_mode,
            self.crop_padding,
            self.interpolation,
            self.mean,
            self.std,
        )
        if self.image_processor is not None and (
            self.image_size is not None or any(value is not None for value in custom)
        ):
            raise ValueError(
                "Set either custom preprocessing settings or image_processor, not both."
            )
        if self.image_processor is None and self.image_size is None:
            raise ValueError(
                "Set image_size for custom preprocessing, or image_processor."
            )
        return self


class Preprocessor:
    """
    Converts an RGB image into a model input tensor.

    Configure it either with custom settings, which resize, crop, scale to
    ``[0, 1]`` and normalize, or with a transformers image processor, which
    then defines the whole transformation. Combining both is an error.

    Example:
        >>> Preprocessor(image_size=256, resize_mode="resize_center_crop")
        >>> Preprocessor(image_processor="facebook/dinov2-small")
    """

    def __init__(
        self,
        image_size: int | None = None,
        resize_mode: ResizeMode | str | None = None,
        crop_padding: int | None = None,
        interpolation: Interpolation | str | None = None,
        mean: RGB | None = None,
        std: RGB | None = None,
        image_processor: BaseImageProcessor | str | None = None,
    ) -> None:
        """
        Initialize the preprocessor.

        Args:
            image_size (int | None): Side of the square output of custom preprocessing.
            resize_mode (ResizeMode | str | None): How the image is brought to the
                output size; defaults to ``resize``.
            crop_padding (int | None): Margin removed by the centre crop; defaults to 32.
            interpolation (Interpolation | str | None): Resampling filter; defaults to ``bicubic``.
            mean (RGB | None): Per-channel mean on the ``[0, 1]`` scale; defaults to ImageNet.
            std (RGB | None): Per-channel standard deviation on the ``[0, 1]`` scale;
                defaults to ImageNet.
            image_processor (BaseImageProcessor | str | None): Transformers image
                processor used instead of all custom settings, or its Hub name or
                local path, loaded with ``AutoImageProcessor.from_pretrained``.
        """
        custom = (resize_mode, crop_padding, interpolation, mean, std)
        self._transform: Callable[[Image.Image], torch.Tensor]
        if image_processor is not None:
            if image_size is not None or any(value is not None for value in custom):
                raise ValueError(
                    "Pass either custom preprocessing settings or image_processor, not both."
                )
            self._transform = _ImageProcessorTransform(_load_processor(image_processor))
        elif image_size is None:
            raise ValueError(
                "Pass image_size for custom preprocessing, or image_processor."
            )
        else:
            self._transform = _CustomTransform(
                image_size=image_size,
                resize_mode=ResizeMode(resize_mode or ResizeMode.RESIZE),
                crop_padding=DEFAULT_CROP_PADDING
                if crop_padding is None
                else crop_padding,
                interpolation=Interpolation(interpolation or Interpolation.BICUBIC),
                mean=IMAGENET_MEAN if mean is None else mean,
                std=IMAGENET_STD if std is None else std,
            )
        return

    @classmethod
    def from_pydantic_model(cls, config: PreprocessingConfig) -> "Preprocessor":
        """Build the preprocessor from its configuration."""
        if config.image_processor is not None:
            return cls(image_processor=config.image_processor)
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
        Convert ``image`` into a model input.

        Args:
            image (ImageArray | Image.Image): RGB image.

        Returns:
            torch.Tensor: Float tensor of shape ``(3, H, W)``.
        """
        pil_image = (
            image.convert("RGB")
            if isinstance(image, Image.Image)
            else Image.fromarray(image)
        )
        return self._transform(pil_image)


class _CustomTransform:
    def __init__(
        self,
        image_size: int,
        resize_mode: ResizeMode,
        crop_padding: int,
        interpolation: Interpolation,
        mean: RGB,
        std: RGB,
    ) -> None:
        if any(value <= 0 for value in std):
            raise ValueError(f"Standard deviations must be positive, got {std}.")
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.crop_padding = crop_padding
        self.resample = _RESAMPLING[interpolation]
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        return

    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(self._resize(image), dtype=np.uint8)
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


class _ImageProcessorTransform:
    def __init__(self, processor: BaseImageProcessor) -> None:
        self.processor = processor
        return

    def __call__(self, image: Image.Image) -> torch.Tensor:
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise TypeError(
                f"{type(self.processor).__name__} must return pixel_values of shape (1, C, H, W)."
            )
        return pixel_values[0]


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _load_processor(processor: BaseImageProcessor | str) -> BaseImageProcessor:
    if isinstance(processor, BaseImageProcessor):
        return processor
    loaded = AutoImageProcessor.from_pretrained(processor)
    if not isinstance(loaded, BaseImageProcessor):
        raise TypeError(f"'{processor}' is not an image processor.")
    return loaded
