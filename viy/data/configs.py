from enum import StrEnum

from kostyl import StrictPydanticBaseModel
from pydantic import Field

from viy.augs import AugmentationConfig
from viy.datasets import DatasetSourceConfig
from viy.datasets import StorageConfig

from .preprocessing import PreprocessingConfig


class Stage(StrEnum):
    """What a dataset is built for; decides default splits and augmentation seeding."""

    TRAIN = "train"
    VAL = "val"


class DatasetSpecConfig(DatasetSourceConfig):
    """
    A dataset of the supported pool and the part of it to use.

    Attributes:
        splits (list[str] | None): Splits or split groups; ``None`` uses the
            dataset defaults for the stage.
        limit (int | None): Maximum number of images taken from every split,
            drawn with class stratification.
    """

    splits: list[str] | None = None
    limit: int | None = Field(default=None, gt=0)


class DetectionDatasetConfig(StrictPydanticBaseModel):
    """
    Configuration of a :class:`viy.data.DetectionDataset`.

    Attributes:
        datasets (list[DatasetSpecConfig]): Datasets to combine.
        stage (Stage): ``train`` or ``val``.
        preprocessing (PreprocessingConfig): Conversion of images into model inputs.
        augmentation (AugmentationConfig | None): Augmentation applied before preprocessing.
        storage (StorageConfig): Canonical layout location and preparation settings.
        seed (int): Seed for subsampling and augmentation.
    """

    datasets: list[DatasetSpecConfig] = Field(min_length=1)
    stage: Stage
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig | None = None
    storage: StorageConfig = Field(default_factory=StorageConfig)
    seed: int = 0
