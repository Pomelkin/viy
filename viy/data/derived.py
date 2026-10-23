import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from pathlib import PurePosixPath
from typing import Final
from typing import Self
from typing import override

import albumentations as A
import cv2
import numpy as np
import PIL
from kostyl import StrictPydanticBaseModel
from kostyl.utils import setup_logger
from PIL import Image
from pydantic import Field
from pydantic import model_validator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from viy.augs import AugmentationConfig
from viy.augs import AugmentationPipeline
from viy.augs import image_rng
from viy.augs import trace_to_json
from viy.datasets import DatasetBuilder
from viy.layout import DatasetLayout
from viy.layout import LayoutError
from viy.layout import SampleRecord
from viy.progress import track

from .dataset import decode_image
from .sampling import subsample


logger = setup_logger(fmt="detailed")

AUGMENTATIONS_COLUMN: Final = "augmentations"
BASE_IMAGE_COLUMN: Final = "base_image"
DERIVED_IMAGES_DIR: Final = "images/derived"
FINGERPRINT_LENGTH: Final = 12
PNG_COMPRESS_LEVEL: Final = 1


class SplitDerivation(StrictPydanticBaseModel):
    """
    How a new split is built from a prepared one.

    The seed fixes both the drawn subset and every augmentation, so the same
    derivation gives the same images on any machine with the same library
    versions, with or without a cache.

    Attributes:
        seed (int): Seed of subsampling and augmentation.
        limit (int | None): Number of images drawn with class stratification;
            ``None`` keeps the whole split.
        augmentation (AugmentationConfig | None): Augmentation applied once to
            every image; ``None`` keeps the images as they are.
    """

    seed: int = Field(ge=0)
    limit: int | None = Field(default=None, gt=0)
    augmentation: AugmentationConfig | None = None

    @model_validator(mode="after")
    def _validate_effect(self) -> Self:
        if self.limit is None and self.augmentation is None:
            raise ValueError("A derivation needs a limit, an augmentation or both.")
        return self

    def fingerprint(self, split: str) -> str:
        """Short hash of the base split name and every setting of the derivation."""
        payload = json.dumps(
            {"split": split, **self.model_dump(mode="json")}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:FINGERPRINT_LENGTH]

    def split_name(self, split: str) -> str:
        """Name of the split derived from ``split``, e.g. ``train-3f9a1c2b7d10``."""
        return f"{split}-{self.fingerprint(split)}"


class DerivedSplitInfo(StrictPydanticBaseModel):
    """
    Provenance of a derived split, stored as ``splits/<name>.json`` next to its CSV.

    Attributes:
        dataset (str): Dataset name.
        split (str): Base split.
        name (str): Name of the derived split.
        derivation (SplitDerivation): Settings the split was built with.
        versions (dict[str, str]): Versions of the libraries that produced the images.
        created_at (datetime): Build time.
    """

    dataset: str
    split: str
    name: str
    derivation: SplitDerivation
    versions: dict[str, str]
    created_at: datetime


def derive_split(
    builder: DatasetBuilder,
    split: str,
    derivation: SplitDerivation,
    num_workers: int = 0,
) -> str:
    """
    Build a split from ``split`` unless it already exists.

    A derivation with only a ``limit`` writes a CSV over the existing images.
    With an ``augmentation`` every image is augmented once with a generator
    seeded from the derivation seed and the image path, and saved as PNG under
    ``images/derived/<name>/``. The applied operations of every image go to
    the ``augmentations`` column and its base image to the ``base_image``
    column. ``distortions`` of augmented images are unknown, because the
    augmentation history is kept apart from the history of the base image.

    A validation :class:`~viy.data.DetectionDataset` with the same seed, limit
    and augmentation sees exactly the same images.

    Args:
        builder (DatasetBuilder): Builder of the dataset.
        split (str): Concrete base split; it is prepared if needed.
        derivation (SplitDerivation): How to build the new split.
        num_workers (int): Worker processes augmenting and encoding images.

    Returns:
        str: Name of the derived split.
    """
    name = derivation.split_name(split)
    layout = builder.layout
    if layout.has_split(name):
        _warn_on_version_change(layout, name)
        return name
    records = subsample(builder.load(split), derivation.limit, derivation.seed)
    with layout.lock():
        if not layout.has_split(name):
            _build(builder, split, name, records, derivation, num_workers)
    return name


def derived_info_path(layout: DatasetLayout, name: str) -> Path:
    """Location of the provenance JSON of the derived split ``name``, next to its CSV."""
    return layout.split_path(name).with_suffix(".json")


def read_derived_info(layout: DatasetLayout, name: str) -> DerivedSplitInfo:
    """Read the provenance of the derived split ``name``."""
    return DerivedSplitInfo.model_validate_json(
        derived_info_path(layout, name).read_text(encoding="utf-8")
    )


def library_versions() -> dict[str, str]:
    """Versions of the libraries that decide the pixels of augmented images."""
    return {
        "viy": version("viy"),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "albumentations": A.__version__,
        "pillow": PIL.__version__,
    }


def derived_image(name: str, image: str) -> str:
    """Location of the augmented copy of ``image`` in the derived split ``name``."""
    return str(PurePosixPath(DERIVED_IMAGES_DIR, name, image).with_suffix(".png"))


def _build(
    builder: DatasetBuilder,
    split: str,
    name: str,
    records: Sequence[SampleRecord],
    derivation: SplitDerivation,
    num_workers: int,
) -> None:
    logger.info(f"Deriving split '{builder.name}/{name}' from '{split}'")
    derived = list(records)
    if derivation.augmentation is not None:
        derived = _augment(
            builder,
            name,
            records,
            AugmentationPipeline.from_pydantic_model(derivation.augmentation),
            derivation.seed,
            num_workers,
        )
    info = DerivedSplitInfo(
        dataset=builder.name,
        split=split,
        name=name,
        derivation=derivation,
        versions=library_versions(),
        created_at=datetime.now(UTC),
    )
    path = derived_info_path(builder.layout, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")
    builder.layout.write_split(
        name, derived, source=f"{builder.name}/{split} with seed {derivation.seed}"
    )
    logger.info(f"Derived split '{builder.name}/{name}' with {len(derived)} images")
    return


def _augment(
    builder: DatasetBuilder,
    name: str,
    records: Sequence[SampleRecord],
    pipeline: AugmentationPipeline,
    seed: int,
    num_workers: int,
) -> list[SampleRecord]:
    targets = [derived_image(name, record.image) for record in records]
    collisions = [path for path, count in Counter(targets).items() if count > 1]
    if collisions:
        raise LayoutError(
            f"Images of '{builder.name}' differ only in their extension and would "
            f"overwrite each other as PNG, e.g. {collisions[:3]}."
        )
    images = _AugmentedImages(builder.layout.dir, records, targets, pipeline, seed)
    loader = DataLoader(images, batch_size=None, num_workers=num_workers)
    traces = [""] * len(records)
    batches = iter(loader)
    for index, trace in track(
        batches, f"Deriving {builder.name}/{name}", total=len(loader)
    ):
        traces[index] = trace
    return [
        SampleRecord(
            image=target,
            label=record.label,
            generator=record.generator,
            source_id=record.source_id,
            extra={
                **record.extra,
                AUGMENTATIONS_COLUMN: trace,
                BASE_IMAGE_COLUMN: record.image,
            },
        )
        for record, target, trace in zip(records, targets, traces, strict=True)
    ]


class _AugmentedImages(Dataset[tuple[int, str]]):
    def __init__(
        self,
        dataset_dir: Path,
        records: Sequence[SampleRecord],
        targets: Sequence[str],
        pipeline: AugmentationPipeline,
        seed: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.records = list(records)
        self.targets = list(targets)
        self.pipeline = pipeline
        self.seed = seed
        return

    def __len__(self) -> int:
        return len(self.records)

    @override
    def __getitem__(self, index: int) -> tuple[int, str]:
        record = self.records[index]
        image, trace = self.pipeline(
            decode_image(self.dataset_dir / record.image),
            image_rng(self.seed, record.image),
        )
        target = self.dataset_dir / self.targets[index]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.tmp")
        Image.fromarray(image).save(
            temporary, format="PNG", compress_level=PNG_COMPRESS_LEVEL
        )
        temporary.replace(target)
        return index, trace_to_json(trace)


def _warn_on_version_change(layout: DatasetLayout, name: str) -> None:
    built = read_derived_info(layout, name).versions
    current = library_versions()
    changed = {
        library: f"{built.get(library)} -> {current_version}"
        for library, current_version in current.items()
        if built.get(library) != current_version
    }
    if changed:
        logger.warning_once(
            f"Derived split '{layout.dataset}/{name}' was built with other library versions "
            f"({changed}); a fresh build may give different pixels. Delete "
            f"{layout.split_path(name)} to rebuild it."
        )
    return
