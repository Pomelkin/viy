import ast
import csv
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from typing import override

from viy.common import ScaleValue
from viy.layout import SampleRecord

from .base import DatasetBuilder
from .base import RawSample
from .base import SplitRole
from .base import SplitSpec
from .hub import HubArtifact


TRAIN_REPO: Final = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
VAL_REPO: Final = "deepfakesMSU/NTIRE-RobustAIGenDetection-val"
TEST_REPO: Final = "deepfakesMSU/NTIRE-RobustAIGenDetection-test-public"
NUM_OFFICIAL_SHARDS: Final = 6
PRIVATE_TEST_DIR: Final = "test/new_test_images/ds/NTIRE_test_private/v1"
PRIVATE_METADATA: Final = ("clear_metadata.csv", "distorted_metadata.csv")


@dataclass(frozen=True, slots=True)
class _FlatSplit:
    images: str
    labels: str


_FLAT_SPLITS: Final = MappingProxyType(
    {
        "val": _FlatSplit("val/val_images", "val/val_labels.csv"),
        "val_hard": _FlatSplit("val/val_images_hard", "val/val_hard_labels.csv"),
        "test_public": _FlatSplit("test/test_images", "test/test_labels.csv"),
        "test_private": _FlatSplit(
            f"{PRIVATE_TEST_DIR}/test_set/test_images",
            f"{PRIVATE_TEST_DIR}/test_set/labels.csv",
        ),
    }
)


def _shard_spec(index: int) -> SplitSpec:
    return SplitSpec(
        name=f"shard_{index}",
        role=SplitRole.TRAIN,
        artifacts=(
            HubArtifact(TRAIN_REPO, f"shard_{index}.zip", subdir="train", extract=True),
        ),
        description=f"Official training shard {index}; binary labels only.",
    )


def _hub_split(
    repo_id: str, subdir: str, archive: str, labels: str
) -> tuple[HubArtifact, ...]:
    return (
        HubArtifact(repo_id, archive, subdir=subdir, extract=True),
        HubArtifact(repo_id, labels, subdir=subdir),
    )


class NTIRE(DatasetBuilder):
    """
    NTIRE 2026 Robust AI-Generated Image Detection in the Wild.

    The raw layout matches the Hub repositories unpacked side by side::

        train/shard_<i>/{images/, labels.csv}
        val/{val_images/, val_labels.csv, val_images_hard/, val_hard_labels.csv}
        test/{test_images/, test_labels.csv}
        test/new_test_images/ds/NTIRE_test_private/v1/...

    Validation and test splits carry the applied distortion chains; the private
    test split also carries the generator of every fake image.
    """

    name = "ntire"
    description = "NTIRE 2026 Robust AI-Generated Image Detection in the Wild."
    split_specs = (
        *(_shard_spec(index) for index in range(NUM_OFFICIAL_SHARDS)),
        SplitSpec(
            name="shard_6",
            role=SplitRole.TRAIN,
            description="Local shard with genuine JPEG AI and WMForger versions of shard 0-5 images.",
        ),
        SplitSpec(
            name="val",
            role=SplitRole.VAL,
            artifacts=_hub_split(VAL_REPO, "val", "val_images.zip", "val_labels.csv"),
            description="Validation, 9 generators, 19 distortion types.",
        ),
        SplitSpec(
            name="val_hard",
            role=SplitRole.VAL,
            artifacts=_hub_split(
                VAL_REPO, "val", "val_images_hard.zip", "val_hard_labels.csv"
            ),
            description="Hard validation, 7 generators, adds neural codecs and adversarial attacks.",
        ),
        SplitSpec(
            name="test_public",
            role=SplitRole.TEST,
            artifacts=_hub_split(
                TEST_REPO, "test", "test_images.zip", "test_labels.csv"
            ),
            description="Public test, 10 generators, 24 distortion types including watermarks.",
        ),
        SplitSpec(
            name="test_private",
            role=SplitRole.TEST,
            description="Private test with per-image generators; only available locally.",
        ),
    )
    split_groups = MappingProxyType(
        {"train": tuple(f"shard_{index}" for index in range(NUM_OFFICIAL_SHARDS))}
    )
    default_train_splits = ("train",)
    default_val_splits = ("val", "val_hard")

    @override
    def collect(self, spec: SplitSpec, raw_dir: Path) -> Iterator[RawSample]:
        if spec.name in _FLAT_SPLITS:
            generators = (
                _private_generators(raw_dir) if spec.name == "test_private" else None
            )
            yield from _collect_flat(spec.name, raw_dir, generators)
        elif spec.name == "shard_6":
            yield from _collect_shard6(raw_dir)
        else:
            yield from _collect_shard(spec.name, raw_dir)
        return


def _collect_shard(split: str, raw_dir: Path) -> Iterator[RawSample]:
    shard_dir = raw_dir / "train" / split
    for row in _read_csv(shard_dir / "labels.csv"):
        name = row["image_name"]
        yield RawSample(
            source=shard_dir / "images" / name,
            record=SampleRecord(
                image=f"images/{split}/{name}", label=int(row["label"])
            ),
        )
    return


def _collect_shard6(raw_dir: Path) -> Iterator[RawSample]:
    shard_dir = raw_dir / "train" / "shard_6"
    for row in _read_csv(shard_dir / "labels.csv"):
        name = row["image_name"]
        match row["transform"]:
            case "jpeg_ai":
                scale: ScaleValue = int(row["jpeg_ai_level"])
            case "wmforger":
                scale = float(row["wmforger_learning_rate"])
            case other:
                raise ValueError(f"shard_6/{name}: unknown transform '{other}'.")
        yield RawSample(
            source=shard_dir / "images" / name,
            record=SampleRecord(
                image=f"images/shard_6/{name}",
                label=int(row["label"]),
                distortions=(row["transform"],),
                distortion_scales=(scale,),
                source_id=row["source_image_name"],
            ),
        )
    return


def _collect_flat(
    split: str, raw_dir: Path, generators: Mapping[str, str | None] | None
) -> Iterator[RawSample]:
    flat = _FLAT_SPLITS[split]
    for row in _read_csv(raw_dir / flat.labels):
        name = row["image_name"]
        label = int(row["label"])
        distortions, scales = _parse_distortions(row)
        generator = (
            None if generators is None else _lookup_generator(generators, name, label)
        )
        yield RawSample(
            source=raw_dir / flat.images / name,
            record=SampleRecord(
                image=f"images/{split}/{name}",
                label=label,
                generator=generator,
                distortions=distortions,
                distortion_scales=scales,
            ),
        )
    return


def _parse_distortions(
    row: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[ScaleValue, ...]]:
    names = ast.literal_eval(row["distortions"])
    scales = ast.literal_eval(row["distortion_scales"])
    if (
        not isinstance(names, list)
        or not isinstance(scales, list)
        or len(names) != len(scales)
    ):
        raise ValueError(
            f"{row['image_name']}: malformed distortions {row['distortions']!r}."
        )
    if (len(names) > 0) != (row["is_distorted"] == "1"):
        raise ValueError(f"{row['image_name']}: is_distorted contradicts distortions.")
    return tuple(str(name) for name in names), tuple(scales)


def _private_generators(raw_dir: Path) -> dict[str, str | None]:
    generators: dict[str, str | None] = {}
    for metadata in PRIVATE_METADATA:
        for row in _read_csv(raw_dir / PRIVATE_TEST_DIR / metadata):
            generator = row["generator"]
            generators[f"{row['val_hashed_name']}.jpg"] = (
                None if generator == "real" else generator
            )
    return generators


def _lookup_generator(
    generators: Mapping[str, str | None], name: str, label: int
) -> str | None:
    if name not in generators:
        raise ValueError(f"{name}: missing from the private test metadata.")
    generator = generators[name]
    if (generator is None) == bool(label):
        raise ValueError(f"{name}: label {label} contradicts generator '{generator}'.")
    return generator


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"NTIRE raw file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
