from typing import Self

from kostyl import StrictPydanticBaseModel
from pydantic import Field
from pydantic import model_validator

from viy.augs import AugmentationConfig
from viy.datasets import DatasetSourceConfig
from viy.datasets import StorageConfig

from .preprocessing import PreprocessingConfig


class TrainDatasetConfig(DatasetSourceConfig):
    """
    A dataset used for training and validation.

    Attributes:
        train_splits (list[str] | None): Splits or split groups used for training;
            ``None`` uses the dataset defaults and ``[]`` excludes the dataset from training.
        val_splits (list[str] | None): Splits used for validation; ``None`` uses
            the dataset defaults and ``[]`` excludes the dataset from validation.
        train_limit (int | None): Maximum number of images taken from every
            training split, drawn with class stratification.
        val_limit (int | None): Maximum number of images taken from every validation split.
    """

    train_splits: list[str] | None = None
    val_splits: list[str] | None = None
    train_limit: int | None = Field(default=None, gt=0)
    val_limit: int | None = Field(default=None, gt=0)


class TrainDataConfig(StrictPydanticBaseModel):
    """
    Everything needed to build training and validation datasets.

    Attributes:
        storage (StorageConfig): Canonical layout location and preparation settings.
        datasets (list[TrainDatasetConfig]): Datasets of the supported pool.
        preprocessing (PreprocessingConfig): Preprocessing of training images.
        val_preprocessing (PreprocessingConfig | None): Preprocessing of validation
            images; ``None`` reuses ``preprocessing``.
        train_augmentation (AugmentationConfig | None): Augmentation of training images.
        val_augmentation (AugmentationConfig | None): Augmentation of validation
            images, applied deterministically per image.
        seed (int): Seed for subsampling and augmentation.
    """

    storage: StorageConfig
    datasets: list[TrainDatasetConfig] = Field(min_length=1)
    preprocessing: PreprocessingConfig
    val_preprocessing: PreprocessingConfig | None = None
    train_augmentation: AugmentationConfig | None = None
    val_augmentation: AugmentationConfig | None = None
    seed: int = 0

    @model_validator(mode="after")
    def _validate_unique_datasets(self) -> Self:
        names = [dataset.name for dataset in self.datasets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Datasets listed more than once: {duplicates}.")
        return self
