import json
from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader
from transformers import ViTImageProcessorPil

from viy.augs import AugmentationPipeline
from viy.augs import NTIREPresetStep
from viy.augs import OpStep
from viy.data import DatasetSpec
from viy.data import DetectionDataset
from viy.data import DetectionDatasetConfig
from viy.data import PreprocessingConfig
from viy.data import Preprocessor
from viy.data import ResizeMode
from viy.data import Stage
from viy.data import check_disjoint
from viy.data import subsample
from viy.datasets import StorageConfig
from viy.layout import SampleRecord


def _train(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path, limit: int | None = None
) -> DetectionDataset:
    return DetectionDataset(
        [
            DatasetSpec("ntire", splits=["train"], limit=limit, raw_path=ntire_raw),
            DatasetSpec("mllm_generated", raw_path=mllm_raw),
        ],
        stage="train",
        preprocessor=Preprocessor(image_size=32),
        augmentation=AugmentationPipeline(
            [
                A.HorizontalFlip(p=0.5),
                NTIREPresetStep("aggressive_stage6", distort_probability=1.0),
            ]
        ),
        storage=storage,
        seed=3,
    )


def test_preprocessor_modes() -> None:
    image = np.full((40, 60, 3), 128, dtype=np.uint8)
    for mode in ResizeMode:
        tensor = Preprocessor(image_size=24, resize_mode=mode)(image)
        assert tensor.shape == (3, 24, 24)
        assert tensor.dtype == torch.float32
    assert Preprocessor(image_size=24, resize_mode="resize_center_crop")(
        image
    ).shape == (3, 24, 24)
    plain = Preprocessor(image_size=8, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))(image)
    assert torch.allclose(plain, torch.full_like(plain, 128 / 255))
    with pytest.raises(ValueError, match="squash"):
        Preprocessor(image_size=8, resize_mode="squash")


def test_subsample_is_stratified() -> None:
    records = [
        SampleRecord(image=f"{index}.jpg", label=int(index < 30))
        for index in range(100)
    ]
    selected = subsample(records, 10, seed=0)
    assert len(selected) == 10
    assert sum(record.label for record in selected) == 3
    assert selected == sorted(
        selected, key=lambda record: int(record.image.split(".")[0])
    )
    assert subsample(records, None, seed=0) == records


def test_train_dataset(storage: StorageConfig, ntire_raw: Path, mllm_raw: Path) -> None:
    train = _train(storage, ntire_raw, mllm_raw, limit=2)
    assert train.stage is Stage.TRAIN
    assert train.sources == [
        *(f"ntire/shard_{i}" for i in range(6)),
        "mllm_generated/train",
    ]
    assert len(train) == 6 * 2 + 18
    assert len(train.labels) == len(train)

    sample = train[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["source"] == "ntire/shard_0"
    trace = json.loads(sample["augmentations"])
    assert all(set(op) == {"name", "scale"} for op in trace)
    assert len({train[1]["augmentations"] for _ in range(8)}) > 1

    batch = next(iter(DataLoader(train, batch_size=4, shuffle=True)))
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["label"].shape == (4,)
    assert len(batch["augmentations"]) == 4


def test_val_dataset(storage: StorageConfig, ntire_raw: Path, mllm_raw: Path) -> None:
    val = DetectionDataset(
        [
            DatasetSpec("ntire", raw_path=ntire_raw),
            DatasetSpec("mllm_generated", raw_path=mllm_raw),
        ],
        stage=Stage.VAL,
        preprocessor=Preprocessor(
            image_size=32, resize_mode="resize_center_crop", crop_padding=8
        ),
        augmentation=AugmentationPipeline([OpStep("jpeg", p=0.5)]),
        storage=storage,
    )
    assert val.sources == ["ntire/val", "ntire/val_hard"]
    parts = val.by_source()
    assert list(parts) == ["ntire/val", "ntire/val_hard"]
    first, second = parts["ntire/val"][2], parts["ntire/val"][2]
    assert first["augmentations"] == second["augmentations"]
    assert torch.equal(first["image"], second["image"])
    loaders = {name: DataLoader(part, batch_size=4) for name, part in parts.items()}
    assert list(loaders) == list(parts)
    assert sum(len(part) for part in parts.values()) == 16


def test_from_pydantic_model(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path
) -> None:
    config = DetectionDatasetConfig.model_validate(
        {
            "datasets": [
                {
                    "name": "ntire",
                    "raw_path": str(ntire_raw),
                    "splits": ["shard_0", "shard_6"],
                    "limit": 2,
                },
                {"name": "mllm_generated", "raw_path": str(mllm_raw)},
            ],
            "stage": "train",
            "preprocessing": {"image_size": 32},
            "augmentation": {
                "steps": [
                    {"kind": "albumentations", "transform": "HorizontalFlip", "p": 0.5}
                ]
            },
            "storage": storage.model_dump(),
            "seed": 1,
        }
    )
    dataset = DetectionDataset.from_pydantic_model(config)
    assert dataset.sources == ["ntire/shard_0", "ntire/shard_6", "mllm_generated/train"]
    assert len(dataset) == 2 + 2 + 18


def test_stage_rules(storage: StorageConfig, ntire_raw: Path, mllm_raw: Path) -> None:
    with pytest.raises(ValueError, match="has val splits"):
        DetectionDataset(
            [DatasetSpec("mllm_generated", raw_path=mllm_raw)],
            stage="val",
            preprocessor=Preprocessor(image_size=32),
            storage=storage,
        )
    with pytest.raises(ValueError, match="more than once"):
        DetectionDataset(
            [
                DatasetSpec("ntire", splits=["shard_0"], raw_path=ntire_raw),
                DatasetSpec("ntire", splits=["shard_0", "shard_1"], raw_path=ntire_raw),
            ],
            stage="train",
            preprocessor=Preprocessor(image_size=32),
            storage=storage,
        )
    with pytest.raises(ValueError, match="'test' is not a valid Stage"):
        DetectionDataset(
            [], stage="test", preprocessor=Preprocessor(image_size=32), storage=storage
        )


def test_check_disjoint(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path
) -> None:
    train = _train(storage, ntire_raw, mllm_raw)
    val = DetectionDataset(
        [DatasetSpec("ntire", raw_path=ntire_raw)],
        stage="val",
        preprocessor=Preprocessor(image_size=32),
        storage=storage,
    )
    check_disjoint(train, val)
    leaky = DetectionDataset(
        [DatasetSpec("ntire", splits=["shard_0"], raw_path=ntire_raw)],
        stage="val",
        preprocessor=Preprocessor(image_size=32),
        storage=storage,
    )
    with pytest.raises(
        ValueError, match="appear in both the train and the val dataset"
    ):
        check_disjoint(train, val, leaky)


def _vit_processor() -> ViTImageProcessorPil:
    return ViTImageProcessorPil(
        size={"height": 24, "width": 24}, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )


def test_preprocessor_with_image_processor(tmp_path: Path) -> None:
    image = np.full((40, 60, 3), 255, dtype=np.uint8)
    tensor = Preprocessor(image_processor=_vit_processor())(image)
    assert tensor.shape == (3, 24, 24)
    assert torch.allclose(tensor, torch.ones_like(tensor))
    with pytest.raises(ValueError, match="not both"):
        Preprocessor(image_size=24, image_processor=_vit_processor())
    with pytest.raises(ValueError, match="not both"):
        Preprocessor(mean=(0.0, 0.0, 0.0), image_processor=_vit_processor())
    with pytest.raises(ValueError, match="image_size"):
        Preprocessor()

    _vit_processor().save_pretrained(tmp_path / "processor")
    config = PreprocessingConfig(image_processor=str(tmp_path / "processor"))
    assert Preprocessor.from_pydantic_model(config)(image).shape == (3, 24, 24)
    with pytest.raises(ValidationError, match="not both"):
        PreprocessingConfig(image_size=24, image_processor="facebook/dinov2-small")
    with pytest.raises(ValidationError, match="image_size"):
        PreprocessingConfig()
