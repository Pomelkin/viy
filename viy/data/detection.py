from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from kostyl.utils import setup_logger
from PIL import Image
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from viy.augs import AlbumentationsTransform
from viy.augs import AugmentationConfig
from viy.augs import AugmentationPipeline
from viy.augs import Step
from viy.datasets import DATASETS
from viy.datasets import StorageConfig
from viy.layout import SampleRecord
from viy.progress import track

from .collate import DetectionCollator
from .configs import DatasetSpecConfig
from .configs import DetectionDatasetConfig
from .configs import Stage
from .dataset import DetectionSample
from .dataset import SplitDataset
from .preprocessing import Preprocessor
from .sampling import subsample


logger = setup_logger(fmt="detailed")

type Augmentation = (
    AugmentationPipeline | AugmentationConfig | Sequence[Step | AlbumentationsTransform]
)
"""A pipeline, its config, or a list of steps and albumentations transforms."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """
    A dataset of the supported pool and the part of it to use.

    Attributes:
        name (str): Registered dataset name, e.g. ``ntire``.
        splits (str | Sequence[str] | None): Split, split group or a list of
            them; ``None`` uses the dataset defaults for the stage.
        limit (int | None): Maximum number of images taken from every split,
            drawn with class stratification.
        raw_path (Path | str | None): Local copy of the raw dataset; ``None``
            downloads it from the Hugging Face Hub.
    """

    name: str
    splits: str | Sequence[str] | None = None
    limit: int | None = None
    raw_path: Path | str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.splits, str):
            object.__setattr__(self, "splits", (self.splits,))
        return

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
        datasets: str | DatasetSpec | Sequence[str | DatasetSpec],
        stage: Stage | str,
        preprocessor: Callable[[Image.Image], torch.Tensor],
        augmentation: Augmentation | None = None,
        storage: StorageConfig | None = None,
        seed: int = 0,
    ) -> None:
        """
        Prepare, load and check every split of the requested datasets.

        Args:
            datasets (str | DatasetSpec | Sequence[str | DatasetSpec]): Datasets to
                combine; a name stands for the dataset with its default splits.
            stage (Stage | str): ``train`` or ``val``.
            preprocessor (Callable[[Image.Image], torch.Tensor]): Converts an augmented
                RGB image into a model input, e.g. a :class:`Preprocessor` or the
                ``preprocess`` method of a benchmarked ``DetectorModel``.
            augmentation (Augmentation | None): Augmentation applied before
                preprocessing: a pipeline, its config, or a list of steps and
                albumentations transforms.
            storage (StorageConfig | None): Canonical layout location and preparation
                settings; ``None`` uses the defaults.
            seed (int): Seed of subsampling, augmentation and of the shuffling of
                :meth:`dataloader`.
        """
        self.stage = Stage(stage)
        self.storage = StorageConfig() if storage is None else storage
        self.seed = seed
        specs = [
            DatasetSpec(spec) if isinstance(spec, str) else spec
            for spec in (
                [datasets] if isinstance(datasets, str | DatasetSpec) else datasets
            )
        ]
        pipeline = _pipeline(augmentation)
        self.parts = [
            part
            for spec in specs
            for part in self._parts(spec, preprocessor, pipeline, seed)
        ]
        if not self.parts:
            raise ValueError(
                f"None of the datasets {[spec.name for spec in specs]} has {self.stage} splits; "
                "list the splits to use with DatasetSpec(name, splits=[...])."
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
            preprocessor=Preprocessor.from_pydantic_model(config.preprocessor),
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

    def record(self, source: str, index: int) -> SampleRecord:
        """Record of a sample from its ``source`` and ``index``, e.g. taken from a batch."""
        return self.by_source()[source].records[index]

    def dataloader(
        self, batch_size: int, shuffle: bool | None = None, **kwargs: Any
    ) -> DataLoader[DetectionSample]:
        """
        A DataLoader over the whole dataset that yields :class:`DetectionBatch` items.

        Training data is shuffled by default, with a generator seeded from
        ``seed``, so ``seed`` alone reproduces the order and the augmentation.

        Args:
            batch_size (int): Batch size.
            shuffle (bool | None): Shuffle the samples; ``None`` shuffles at the
                training stage only.
            **kwargs (Any): Other DataLoader arguments, e.g. ``num_workers``.

        Returns:
            DataLoader[DetectionSample]: The loader.
        """
        if shuffle is None:
            shuffle = self.stage is Stage.TRAIN
        return _loader(self, batch_size, shuffle=shuffle, seed=self.seed, kwargs=kwargs)

    def dataloaders(
        self, batch_size: int, **kwargs: Any
    ) -> dict[str, DataLoader[DetectionSample]]:
        """
        One unshuffled DataLoader per split, keyed by ``<dataset>/<split>``, for validation.

        Args:
            batch_size (int): Batch size.
            **kwargs (Any): Other DataLoader arguments, e.g. ``num_workers``.

        Returns:
            dict[str, DataLoader[DetectionSample]]: Loaders in the order of :attr:`sources`.
        """
        return {
            part.source: _loader(
                part, batch_size, shuffle=False, seed=self.seed, kwargs=kwargs
            )
            for part in self.parts
        }

    def _parts(
        self,
        spec: DatasetSpec,
        preprocessor: Callable[[Image.Image], torch.Tensor],
        augmentation: AugmentationPipeline | None,
        seed: int,
    ) -> list[SplitDataset]:
        builder_cls = DATASETS.get(spec.name)
        if builder_cls.benchmark_only:
            raise ValueError(
                f"Dataset '{spec.name}' is for benchmarking only; evaluate it with viy.bench "
                "instead of training or validating on it."
            )
        builder = builder_cls(raw_path=spec.raw_path, storage=self.storage)
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
    images = [
        (part.dataset_dir / record.image, part.source)
        for part in dataset.parts
        for record in part.records
    ]
    return {
        path.resolve(): source
        for path, source in track(
            images, f"Resolving {dataset.stage} images", transient=True
        )
    }


def _pipeline(augmentation: Augmentation | None) -> AugmentationPipeline | None:
    match augmentation:
        case None | AugmentationPipeline():
            return augmentation
        case AugmentationConfig():
            return AugmentationPipeline.from_pydantic_model(augmentation)
        case _:
            return AugmentationPipeline(augmentation)


def _loader(
    dataset: Dataset[DetectionSample],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    kwargs: dict[str, Any],
) -> DataLoader[DetectionSample]:
    if shuffle and "generator" not in kwargs:
        kwargs = {**kwargs, "generator": torch.Generator().manual_seed(seed)}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DetectionCollator(),
        **kwargs,
    )
