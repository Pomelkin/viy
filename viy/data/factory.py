from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from kostyl.utils import setup_logger

from viy.augs import AugmentationConfig
from viy.augs import AugmentationPipeline
from viy.datasets import DatasetBuilder
from viy.layout import SampleRecord

from .configs import TrainDataConfig
from .configs import TrainDatasetConfig
from .dataset import ConcatDetectionDataset
from .dataset import DetectionDataset
from .preprocessing import Preprocessor


logger = setup_logger(fmt="detailed")


def subsample(
    records: Sequence[SampleRecord], limit: int | None, seed: int
) -> list[SampleRecord]:
    """
    Draw at most ``limit`` records keeping the class ratio and the original order.

    Args:
        records (Sequence[SampleRecord]): Records of a split.
        limit (int | None): Maximum number of records; ``None`` keeps all.
        seed (int): Seed of the draw.

    Returns:
        list[SampleRecord]: Selected records.
    """
    if limit is None or limit >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted({record.label for record in records}):
        indices = [
            index for index, record in enumerate(records) if record.label == label
        ]
        quota = round(limit * len(indices) / len(records))
        selected.extend(int(index) for index in rng.permutation(indices)[:quota])
    return [records[index] for index in sorted(selected)]


@dataclass(frozen=True, slots=True)
class TrainValDatasets:
    """
    Training and validation datasets ready for a DataLoader.

    Attributes:
        train (ConcatDetectionDataset): Augmented training images of all datasets.
        val (ConcatDetectionDataset): Validation images; ``val.part(name)`` gives
            one split for per-split metrics.
    """

    train: ConcatDetectionDataset
    val: ConcatDetectionDataset

    @classmethod
    def from_pydantic_model(cls, config: TrainDataConfig) -> "TrainValDatasets":
        """
        Prepare, load and check every configured split.

        A dataset without validation splits is used for training only, with a
        warning; an error is raised if no dataset has validation splits.

        Args:
            config (TrainDataConfig): Data configuration.

        Returns:
            TrainValDatasets: Training and validation datasets.
        """
        return _TrainValBuilder(config).build()


class _TrainValBuilder:
    def __init__(self, config: TrainDataConfig) -> None:
        self.config = config
        self.train_preprocessor = Preprocessor.from_pydantic_model(config.preprocessing)
        self.val_preprocessor = Preprocessor.from_pydantic_model(
            config.val_preprocessing or config.preprocessing
        )
        self.train_augmentation = _pipeline(config.train_augmentation)
        self.val_augmentation = _pipeline(config.val_augmentation)
        return

    def build(self) -> TrainValDatasets:
        train_parts: list[DetectionDataset] = []
        val_parts: list[DetectionDataset] = []
        for dataset_config in self.config.datasets:
            builder = DatasetBuilder.from_pydantic_model(
                dataset_config, self.config.storage
            )
            train_parts.extend(self._parts(builder, dataset_config, train=True))
            val_parts.extend(self._parts(builder, dataset_config, train=False))
        if not train_parts:
            raise ValueError("None of the configured datasets has training splits.")
        if not val_parts:
            raise ValueError(
                "None of the configured datasets has validation splits; "
                "add a dataset with validation splits or set val_splits explicitly."
            )
        _check_disjoint(train_parts, val_parts)
        logger.info(
            f"Training on {sum(len(part) for part in train_parts)} images from {[part.source for part in train_parts]}, "
            f"validating on {sum(len(part) for part in val_parts)} images from {[part.source for part in val_parts]}"
        )
        return TrainValDatasets(
            train=ConcatDetectionDataset(train_parts),
            val=ConcatDetectionDataset(val_parts),
        )

    def _parts(
        self, builder: DatasetBuilder, dataset_config: TrainDatasetConfig, train: bool
    ) -> list[DetectionDataset]:
        requested = dataset_config.train_splits if train else dataset_config.val_splits
        defaults = builder.default_train_splits if train else builder.default_val_splits
        splits = builder.expand_splits(defaults if requested is None else requested)
        if not splits and not train:
            logger.warning(
                f"Dataset '{builder.name}' has no validation splits and is used for training only."
            )
        limit = dataset_config.train_limit if train else dataset_config.val_limit
        return [
            DetectionDataset(
                records=subsample(builder.load(split), limit, self.config.seed),
                dataset_dir=builder.layout.dir,
                source=f"{builder.name}/{split}",
                preprocessor=self.train_preprocessor
                if train
                else self.val_preprocessor,
                augmentation=self.train_augmentation
                if train
                else self.val_augmentation,
                seed=self.config.seed,
                deterministic=not train,
            )
            for split in splits
        ]


def _pipeline(config: AugmentationConfig | None) -> AugmentationPipeline | None:
    return None if config is None else AugmentationPipeline.from_pydantic_model(config)


def _resolved_images(parts: Sequence[DetectionDataset]) -> dict[Path, str]:
    return {
        (part.dataset_dir / record.image).resolve(): part.source
        for part in parts
        for record in part.records
    }


def _check_disjoint(
    train_parts: Sequence[DetectionDataset], val_parts: Sequence[DetectionDataset]
) -> None:
    train_images = _resolved_images(train_parts)
    val_images = _resolved_images(val_parts)
    overlap = train_images.keys() & val_images.keys()
    if overlap:
        examples = [
            f"{train_images[path]} and {val_images[path]}: {path}"
            for path in sorted(overlap)[:3]
        ]
        raise ValueError(
            f"{len(overlap)} images are used for both training and validation, e.g. {examples}."
        )
    return
