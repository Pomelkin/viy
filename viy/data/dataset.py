import os
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from typing import override

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

from viy.augs import AppliedOp
from viy.augs import AugmentationPipeline
from viy.augs import image_rng
from viy.common import ImageArray
from viy.layout import SampleRecord


class DetectionSample(TypedDict):
    """
    One item of a detection dataset.

    Attributes:
        image (torch.Tensor): Preprocessed image of shape ``(3, H, W)``.
        label (int): ``0`` for real images, ``1`` for generated ones.
        source (str): ``<dataset>/<split>`` the image comes from.
        index (int): Position of the record in ``SplitDataset.records``; with a
            ``limit`` it differs from the row of the split CSV.
        path (str): Image path relative to the dataset directory, as in the split
            CSV and in benchmark predictions.
        augmentations (tuple[AppliedOp, ...]): Applied augmentations with their
            levels in order; ``viy.augs.trace_to_json`` serializes them.
    """

    image: torch.Tensor
    label: int
    source: str
    index: int
    path: str
    augmentations: tuple[AppliedOp, ...]


def decode_image(path: Path) -> ImageArray:
    """Decode an image file as an RGB ``uint8`` array."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


class SplitDataset(Dataset[DetectionSample]):
    """
    Images of one split with optional augmentation and preprocessing.

    Augmentation randomness comes from a generator owned by each process and
    seeded from ``seed`` and the distributed rank, plus, in a DataLoader
    worker, the worker seed that the DataLoader derives from its generator. A
    seeded DataLoader therefore reproduces the whole run, every epoch sees new
    augmentations, and ranks never repeat each other's augmentations. With ``deterministic=True`` the generator is derived
    from ``seed`` and the image path instead, so every image is always
    augmented the same way, whatever subset of the split is used.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        dataset_dir: Path,
        source: str,
        preprocessor: Callable[[Image.Image], torch.Tensor],
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
            preprocessor (Callable[[Image.Image], torch.Tensor]): Converts an augmented
                RGB image into a tensor, e.g. a ``Preprocessor`` or ``DetectorModel.preprocess``.
            augmentation (AugmentationPipeline | None): Augmentation applied before preprocessing.
            seed (int): Seed mixed into every augmentation generator.
            deterministic (bool): Derive the generator from ``seed`` and the image path.
        """
        self.records = list(records)
        self.dataset_dir = Path(dataset_dir)
        self.source = source
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.seed = seed
        self.deterministic = deterministic
        self.rank = _process_rank()
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
        return decode_image(self.image_path(index))

    def __len__(self) -> int:
        return len(self.records)

    @override
    def __getitem__(self, index: int) -> DetectionSample:
        image = self.read_image(index)
        trace: list[AppliedOp] = []
        if self.augmentation is not None:
            image, trace = self.augmentation(image, self._rng(index))
        return DetectionSample(
            image=self.preprocessor(Image.fromarray(image)),
            label=self.records[index].label,
            source=self.source,
            index=index,
            path=self.records[index].image,
            augmentations=tuple(trace),
        )

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return image_rng(self.seed, self.records[index].image)
        if self._worker_rng is None or self._worker_pid != os.getpid():
            worker = get_worker_info()
            self._worker_rng = np.random.default_rng(
                (self.seed, self.rank)
                if worker is None
                else (self.seed, self.rank, worker.seed)
            )
            self._worker_pid = os.getpid()
        return self._worker_rng


def _process_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
