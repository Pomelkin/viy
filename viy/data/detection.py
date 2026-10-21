from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import torch
from kostyl.utils import setup_logger
from torch.utils.data import ConcatDataset

from viy.augs import AugmentationPipeline
from viy.common import ImageArray
from viy.datasets import DATASETS
from viy.datasets import StorageConfig

from .configs import DatasetSpecConfig
from .configs import DetectionDatasetConfig
from .configs import Stage
from .dataset import DetectionSample
from .dataset import SplitDataset
from .preprocessing import Preprocessor
from .sampling import subsample


logger = setup_logger(fmt="detailed")


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """
    A dataset of the supported pool and the part of it to use.

    Attributes:
        name (str): Registered dataset name, e.g. ``ntire``.
        splits (Sequence[str] | None): Splits or split groups; ``None`` uses the
            dataset defaults for the stage.
        limit (int | None): Maximum number of images taken from every split,
            drawn with class stratification.
        raw_path (Path | str | None): Local copy of the raw dataset; ``None``
            downloads it from the Hugging Face Hub.
    """

    name: str
    splits: Sequence[str] | None = None
    limit: int | None = None
    raw_path: Path | str | None = None

    @classmethod
    def from_pydantic_model(cls, config: DatasetSpecConfig) -> "DatasetSpec":
        """Build the specification from its configuration."""
        return cls(
            name=config.name,
            splits=config.splits,
            limit=config.limit,
            raw_path=config.raw_path,
        )


class DetectionDataset(ConcatDataset[DetectionSample]):
    """
    Images of one or several datasets for one stage, ready for a DataLoader.

    ``stage`` picks the default splits of every dataset and how augmentation is
    seeded: randomly per worker for training, fixed per image for validation.
    A dataset without splits for the stage is skipped with a warning, and an
    error is raised when no dataset has any. Use :meth:`by_source` to get one
    dataset, and so one DataLoader, per split, and :func:`check_disjoint` to
    make sure training and validation images do not overlap.

    Example:
        >>> train = DetectionDataset(
        ...     [DatasetSpec("ntire", splits=["train"]), DatasetSpec("mllm_generated", limit=5000)],
        ...     stage="train",
        ...     preprocessor=Preprocessor(image_size=256),
        ...     augmentation=AugmentationPipeline([A.HorizontalFlip(p=0.5), NTIREPresetStep("aggressive")]),
        ... )
        >>> val = DetectionDataset(
        ...     [DatasetSpec("ntire")], stage="val", preprocessor=Preprocessor(image_size=256)
        ... )
        >>> check_disjoint(train, val)
    """

    def __init__(
        self,
        datasets: Sequence[DatasetSpec],
        stage: Stage | str,
        preprocessor: Callable[[ImageArray], torch.Tensor],
        augmentation: AugmentationPipeline | None = None,
        storage: StorageConfig | None = None,
        seed: int = 0,
    ) -> None:
        """
        Prepare, load and check every split of the requested datasets.

        Args:
            datasets (Sequence[DatasetSpec]): Datasets to combine.
            stage (Stage | str): ``train`` or ``val``.
            preprocessor (Callable[[ImageArray], torch.Tensor]): Converts an augmented
                ``uint8`` RGB image into a model input, e.g. a :class:`Preprocessor`.
            augmentation (AugmentationPipeline | None): Augmentation applied before preprocessing.
            storage (StorageConfig | None): Canonical layout location and preparation
                settings; ``None`` uses the defaults.
            seed (int): Seed for subsampling and augmentation.
        """
        self.stage = Stage(stage)
        self.storage = StorageConfig() if storage is None else storage
        self.parts = [
            part
            for spec in datasets
            for part in self._parts(spec, preprocessor, augmentation, seed)
        ]
        if not self.parts:
            raise ValueError(
                f"None of the datasets {[spec.name for spec in datasets]} has {self.stage} splits."
            )
        duplicates = sorted(
            {source for source in self.sources if self.sources.count(source) > 1}
        )
        if duplicates:
            raise ValueError(f"Splits listed more than once: {duplicates}.")
        super().__init__(self.parts)
        logger.info(
            f"Built {self.stage} dataset of {len(self)} images from {self.sources}"
        )
        return

    @classmethod
    def from_pydantic_model(cls, config: DetectionDatasetConfig) -> "DetectionDataset":
        """Build the dataset from its configuration."""
        augmentation = (
            None
            if config.augmentation is None
            else AugmentationPipeline.from_pydantic_model(config.augmentation)
        )
        return cls(
            datasets=[
                DatasetSpec.from_pydantic_model(spec) for spec in config.datasets
            ],
            stage=config.stage,
            preprocessor=Preprocessor.from_pydantic_model(config.preprocessing),
            augmentation=augmentation,
            storage=config.storage,
            seed=config.seed,
        )

    @property
    def sources(self) -> list[str]:
        """``<dataset>/<split>`` names of the parts in order."""
        return [part.source for part in self.parts]

    @property
    def labels(self) -> list[int]:
        """Label of every sample in order."""
        return [label for part in self.parts for label in part.labels]

    def by_source(self) -> dict[str, SplitDataset]:
        """One dataset per split, keyed by ``<dataset>/<split>``."""
        return {part.source: part for part in self.parts}

    def _parts(
        self,
        spec: DatasetSpec,
        preprocessor: Callable[[ImageArray], torch.Tensor],
        augmentation: AugmentationPipeline | None,
        seed: int,
    ) -> list[SplitDataset]:
        builder = DATASETS.get(spec.name)(raw_path=spec.raw_path, storage=self.storage)
        defaults = (
            builder.default_train_splits
            if self.stage is Stage.TRAIN
            else builder.default_val_splits
        )
        splits = builder.expand_splits(defaults if spec.splits is None else spec.splits)
        if not splits:
            logger.warning(
                f"Dataset '{spec.name}' has no {self.stage} splits and is skipped."
            )
        return [
            SplitDataset(
                records=subsample(builder.load(split), spec.limit, seed),
                dataset_dir=builder.layout.dir,
                source=f"{spec.name}/{split}",
                preprocessor=preprocessor,
                augmentation=augmentation,
                seed=seed,
                deterministic=self.stage is Stage.VAL,
            )
            for split in splits
        ]


def check_disjoint(*datasets: DetectionDataset) -> None:
    """
    Make sure no image belongs to more than one of ``datasets``.

    Images are compared by resolved file path, so the same raw file reached
    through different splits or datasets is caught as well.

    Args:
        *datasets (DetectionDataset): Datasets to compare, e.g. training and validation.

    Raises:
        ValueError: If two datasets share images.
    """
    images = [_resolved_images(dataset) for dataset in datasets]
    for (first, first_images), (second, second_images) in combinations(
        zip(datasets, images, strict=True), 2
    ):
        overlap = first_images.keys() & second_images.keys()
        if overlap:
            examples = [
                f"{first_images[path]} and {second_images[path]}: {path}"
                for path in sorted(overlap)[:3]
            ]
            raise ValueError(
                f"{len(overlap)} images appear in both the {first.stage} and the "
                f"{second.stage} dataset, e.g. {examples}."
            )
    return


def _resolved_images(dataset: DetectionDataset) -> dict[Path, str]:
    return {
        (part.dataset_dir / record.image).resolve(): part.source
        for part in dataset.parts
        for record in part.records
    }
