import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from viy.augs import AugmentationConfig
from viy.augs import AugmentationPipeline
from viy.augs import image_rng
from viy.augs import trace_from_json
from viy.data import AUGMENTATIONS_COLUMN
from viy.data import BASE_IMAGE_COLUMN
from viy.data import DatasetSpec
from viy.data import DetectionDataset
from viy.data import Preprocessor
from viy.data import SplitDerivation
from viy.data import decode_image
from viy.data import derive_split
from viy.data import read_derived_info
from viy.data import subsample
from viy.datasets import MLLMGenerated
from viy.datasets import StorageConfig


AUGMENTATION = AugmentationConfig.model_validate(
    {
        "steps": [
            {"kind": "op", "op": "jpeg", "p": 0.5},
            {"kind": "ntire_preset", "preset": "val", "p": 0.7},
        ]
    }
)


def _derivation(seed: int = 7, limit: int | None = 8) -> SplitDerivation:
    return SplitDerivation(seed=seed, limit=limit, augmentation=AUGMENTATION)


def test_image_rng_depends_on_seed_and_image() -> None:
    def draw(seed: int, image: str) -> int:
        return int(image_rng(seed, image).integers(2**32))

    assert draw(0, "images/a.png") == draw(0, "images/a.png")
    assert draw(0, "images/a.png") != draw(0, "images/b.png")
    assert draw(0, "images/a.png") != draw(1, "images/a.png")


def test_derivation_name_follows_settings() -> None:
    derivation = _derivation()
    name = derivation.split_name("train")
    assert name.startswith("train-")
    assert name == _derivation().split_name("train")
    assert name != _derivation(seed=8).split_name("train")
    assert name != _derivation(limit=9).split_name("train")
    assert name != derivation.split_name("val")
    with pytest.raises(ValidationError, match="limit, an augmentation or both"):
        SplitDerivation(seed=0)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SplitDerivation(seed=-1, limit=5)


def test_derive_split(mllm_raw: Path, storage: StorageConfig) -> None:
    builder = MLLMGenerated(raw_path=mllm_raw, storage=storage)
    derivation = _derivation()
    name = derive_split(builder, "train", derivation)
    assert name == derivation.split_name("train")

    records = builder.load(name)
    base = subsample(builder.load("train"), derivation.limit, derivation.seed)
    assert len(records) == 8
    assert [record.extra[BASE_IMAGE_COLUMN] for record in records] == [
        record.image for record in base
    ]
    assert all(record.image.startswith(f"images/derived/{name}/") for record in records)
    assert all(record.image.endswith(".png") for record in records)
    assert all(record.distortions is None for record in records)
    assert [record.label for record in records] == [record.label for record in base]

    pipeline = AugmentationPipeline.from_pydantic_model(AUGMENTATION)
    for record, original in zip(records, base, strict=True):
        expected, trace = pipeline(
            decode_image(builder.layout.dir / original.image),
            image_rng(derivation.seed, original.image),
        )
        assert np.array_equal(decode_image(builder.layout.image_path(record)), expected)
        assert trace_from_json(record.extra[AUGMENTATIONS_COLUMN]) == tuple(trace)

    info = read_derived_info(builder.layout, name)
    assert info.split == "train"
    assert info.derivation == derivation
    assert set(info.versions) == {"viy", "numpy", "opencv", "albumentations", "pillow"}
    stored = json.loads(
        builder.layout.split_path(name).with_suffix(".json").read_text()
    )
    assert stored["derivation"]["seed"] == derivation.seed

    csv_mtime = builder.layout.split_path(name).stat().st_mtime_ns
    assert derive_split(builder, "train", derivation) == name
    assert builder.layout.split_path(name).stat().st_mtime_ns == csv_mtime


def test_derive_split_is_reproducible_without_cache(
    mllm_raw: Path, tmp_path: Path
) -> None:
    def build(root: Path, num_workers: int) -> tuple[list[str], list[bytes]]:
        builder = MLLMGenerated(raw_path=mllm_raw, storage=StorageConfig(root=root))
        name = derive_split(builder, "train", _derivation(), num_workers=num_workers)
        records = builder.load(name)
        return (
            [record.extra[AUGMENTATIONS_COLUMN] for record in records],
            [
                decode_image(builder.layout.image_path(record)).tobytes()
                for record in records
            ],
        )

    first = build(tmp_path / "first", num_workers=0)
    second = build(tmp_path / "second", num_workers=2)
    assert first == second
    other = MLLMGenerated(
        raw_path=mllm_raw, storage=StorageConfig(root=tmp_path / "other")
    )
    reseeded = derive_split(other, "train", _derivation(seed=8))
    assert [
        record.extra[AUGMENTATIONS_COLUMN] for record in other.load(reseeded)
    ] != first[0]


def test_limit_only_derivation_reuses_images(
    mllm_raw: Path, storage: StorageConfig
) -> None:
    builder = MLLMGenerated(raw_path=mllm_raw, storage=storage)
    derivation = SplitDerivation(seed=3, limit=6)
    records = builder.load(derive_split(builder, "train", derivation))
    assert records == subsample(builder.load("train"), 6, seed=3)
    assert not (builder.layout.images_dir / "derived").exists()


def test_val_dataset_matches_derived_split(
    mllm_raw: Path, storage: StorageConfig
) -> None:
    builder = MLLMGenerated(raw_path=mllm_raw, storage=storage)
    derivation = _derivation()
    derived = builder.load(derive_split(builder, "train", derivation))
    val = DetectionDataset(
        [DatasetSpec("mllm_generated", splits=["train"], limit=8, raw_path=mllm_raw)],
        stage="val",
        preprocessor=Preprocessor(image_size=16),
        augmentation=AugmentationPipeline.from_pydantic_model(AUGMENTATION),
        storage=storage,
        seed=derivation.seed,
    )
    assert [val[index]["augmentations"] for index in range(len(val))] == [
        trace_from_json(record.extra[AUGMENTATIONS_COLUMN]) for record in derived
    ]
    full = DetectionDataset(
        [DatasetSpec("mllm_generated", splits=["train"], raw_path=mllm_raw)],
        stage="val",
        preprocessor=Preprocessor(image_size=16),
        augmentation=AugmentationPipeline.from_pydantic_model(AUGMENTATION),
        storage=storage,
        seed=derivation.seed,
    )
    by_image = {
        record.image: full[index]["augmentations"]
        for index, record in enumerate(full.parts[0].records)
    }
    assert [by_image[record.extra[BASE_IMAGE_COLUMN]] for record in derived] == [
        trace_from_json(record.extra[AUGMENTATIONS_COLUMN]) for record in derived
    ]


@pytest.mark.parametrize(
    "steps",
    [
        [{"kind": "op", "op": "jpeg", "levels": [70]}],
        [
            {"kind": "op", "op": "brighten", "p": 0.5},
            {"kind": "op", "op": "jpeg", "levels": [80]},
        ],
    ],
)
def test_single_level_jpeg_is_stored_as_jpeg(
    mllm_raw: Path, storage: StorageConfig, steps: list[dict[str, object]]
) -> None:
    builder = MLLMGenerated(raw_path=mllm_raw, storage=storage)
    augmentation = AugmentationConfig.model_validate({"steps": steps})
    derivation = SplitDerivation(seed=5, limit=8, augmentation=augmentation)
    records = builder.load(derive_split(builder, "train", derivation))
    base = subsample(builder.load("train"), derivation.limit, derivation.seed)

    pipeline = AugmentationPipeline.from_pydantic_model(augmentation)
    for record, original in zip(records, base, strict=True):
        path = builder.layout.image_path(record)
        assert record.image.endswith(".jpg")
        with Image.open(path) as stored:
            assert stored.format == "JPEG"
        expected, trace = pipeline(
            decode_image(builder.layout.dir / original.image),
            image_rng(derivation.seed, original.image),
        )
        assert np.array_equal(decode_image(path), expected)
        assert trace_from_json(record.extra[AUGMENTATIONS_COLUMN]) == tuple(trace)


@pytest.mark.parametrize(
    "step",
    [
        {"kind": "op", "op": "jpeg", "levels": [70, 80]},
        {"kind": "op", "op": "jpeg", "p": 0.5, "levels": [70]},
        {"kind": "op", "op": "jpeg"},
    ],
)
def test_random_jpeg_is_stored_as_png(
    mllm_raw: Path, storage: StorageConfig, step: dict[str, object]
) -> None:
    builder = MLLMGenerated(raw_path=mllm_raw, storage=storage)
    derivation = SplitDerivation(
        seed=5,
        limit=4,
        augmentation=AugmentationConfig.model_validate({"steps": [step]}),
    )
    records = builder.load(derive_split(builder, "train", derivation))
    assert all(record.image.endswith(".png") for record in records)
