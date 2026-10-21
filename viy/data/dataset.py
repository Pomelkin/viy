import os
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from typing import override

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

from viy.augs import AppliedOp
from viy.augs import AugmentationPipeline
from viy.augs import trace_to_json
from viy.common import ImageArray
from viy.layout import SampleRecord


class DetectionSample(TypedDict):
    """
    One item of a detection dataset.

    Attributes:
        image (torch.Tensor): Preprocessed image of shape ``(3, H, W)``.
        label (int): ``0`` for real images, ``1`` for generated ones.
        source (str): ``<dataset>/<split>`` the image comes from.
        index (int): Row of the image in its split.
        augmentations (str): JSON list of the applied augmentations with their levels.
    """

    image: torch.Tensor
    label: int
    source: str
    index: int
    augmentations: str


class DetectionDataset(Dataset[DetectionSample]):
    """
    Images of one split with optional augmentation and preprocessing.

    Augmentation randomness comes from a generator owned by each worker
    process and seeded from ``torch.initial_seed()``, so a seeded DataLoader
    reproduces the whole run and every epoch sees new augmentations. With
    ``deterministic=True`` the generator is derived from ``(seed, index)``
    instead, so every image is always augmented the same way.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        dataset_dir: Path,
        source: str,
        preprocessor: Callable[[ImageArray], torch.Tensor],
        augmentation: AugmentationPipeline | None = None,
        seed: int = 0,
        deterministic: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            records (Sequence[SampleRecord]): Records of the split.
            dataset_dir (Path): Directory the record image paths are relative to.
            source (str): ``<dataset>/<split>`` name reported in every sample.
            preprocessor (Callable[[ImageArray], torch.Tensor]): Converts an augmented image into a tensor.
            augmentation (AugmentationPipeline | None): Augmentation applied before preprocessing.
            seed (int): Seed mixed into every augmentation generator.
            deterministic (bool): Derive the generator from ``(seed, index)``.
        """
        self.records = list(records)
        self.dataset_dir = Path(dataset_dir)
        self.source = source
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.seed = seed
        self.deterministic = deterministic
        self._worker_rng: np.random.Generator | None = None
        self._worker_pid: int | None = None
        return

    @property
    def labels(self) -> list[int]:
        """Label of every record."""
        return [record.label for record in self.records]

    def image_path(self, index: int) -> Path:
        """Path of the image of record ``index``."""
        return self.dataset_dir / self.records[index].image

    def read_image(self, index: int) -> ImageArray:
        """Decode the image of record ``index`` as RGB."""
        with Image.open(self.image_path(index)) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.records)

    @override
    def __getitem__(self, index: int) -> DetectionSample:
        image = self.read_image(index)
        trace: list[AppliedOp] = []
        if self.augmentation is not None:
            image, trace = self.augmentation(image, self._rng(index))
        return DetectionSample(
            image=self.preprocessor(image),
            label=self.records[index].label,
            source=self.source,
            index=index,
            augmentations=trace_to_json(trace),
        )

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng((self.seed, index))
        if self._worker_rng is None or self._worker_pid != os.getpid():
            self._worker_rng = np.random.default_rng((self.seed, torch.initial_seed()))
            self._worker_pid = os.getpid()
        return self._worker_rng


class ConcatDetectionDataset(ConcatDataset[DetectionSample]):
    """Concatenation of detection datasets that keeps access to its parts."""

    def __init__(self, datasets: Sequence[DetectionDataset]) -> None:
        """
        Initialize the concatenation.

        Args:
            datasets (Sequence[DetectionDataset]): Parts in order.
        """
        super().__init__(list(datasets))
        self.parts = list(datasets)
        return

    @property
    def sources(self) -> list[str]:
        """``<dataset>/<split>`` names of the parts."""
        return [part.source for part in self.parts]

    @property
    def labels(self) -> list[int]:
        """Label of every sample in order."""
        return [label for part in self.parts for label in part.labels]

    def part(self, source: str) -> DetectionDataset:
        """Return the part named ``source``."""
        for part in self.parts:
            if part.source == source:
                return part
        raise KeyError(f"No part '{source}'. Available: {', '.join(self.sources)}.")
