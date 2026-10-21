import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from viy.data import Preprocessor
from viy.data import ResizeMode
from viy.data import TrainDataConfig
from viy.data import TrainValDatasets
from viy.data import subsample
from viy.datasets import StorageConfig
from viy.layout import SampleRecord


def _config(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path, **ntire: object
) -> TrainDataConfig:
    return TrainDataConfig.model_validate(
        {
            "storage": storage.model_dump(),
            "datasets": [
                {"name": "ntire", "raw_path": str(ntire_raw), **ntire},
                {"name": "mllm_generated", "raw_path": str(mllm_raw)},
            ],
            "preprocessing": {"image_size": 32},
            "val_preprocessing": {
                "image_size": 32,
                "resize_mode": "resize_center_crop",
                "crop_padding": 8,
            },
            "train_augmentation": {
                "steps": [
                    {"kind": "albumentations", "transform": "HorizontalFlip", "p": 0.5},
                    {
                        "kind": "ntire_preset",
                        "preset": "aggressive",
                        "distort_probability": 1.0,
                    },
                ]
            },
            "val_augmentation": {"steps": [{"kind": "op", "op": "jpeg", "p": 0.5}]},
            "seed": 3,
        }
    )


def test_preprocessor_modes() -> None:
    image = np.full((40, 60, 3), 128, dtype=np.uint8)
    for mode in ResizeMode:
        tensor = Preprocessor(image_size=24, resize_mode=mode)(image)
        assert tensor.shape == (3, 24, 24)
        assert tensor.dtype == torch.float32
    plain = Preprocessor(image_size=8, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))(image)
    assert torch.allclose(plain, torch.full_like(plain, 128 / 255))


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


def test_train_val_datasets(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path
) -> None:
    datasets = TrainValDatasets.from_pydantic_model(
        _config(storage, ntire_raw, mllm_raw, train_limit=2)
    )
    assert datasets.train.sources == [
        *(f"ntire/shard_{i}" for i in range(6)),
        "mllm_generated/train",
    ]
    assert datasets.val.sources == ["ntire/val", "ntire/val_hard"]
    assert len(datasets.train) == 6 * 2 + 18
    assert len(datasets.val) == 16

    sample = datasets.train[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["source"] == "ntire/shard_0"
    trace = json.loads(sample["augmentations"])
    assert all(set(op) == {"name", "scale"} for op in trace)

    val_part = datasets.val.part("ntire/val")
    first, second = val_part[2], val_part[2]
    assert first["augmentations"] == second["augmentations"]
    assert torch.equal(first["image"], second["image"])

    batch = next(iter(DataLoader(datasets.train, batch_size=4, shuffle=True)))
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["label"].shape == (4,)
    assert len(batch["augmentations"]) == 4


def test_training_augmentations_vary(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path
) -> None:
    datasets = TrainValDatasets.from_pydantic_model(
        _config(storage, ntire_raw, mllm_raw)
    )
    traces = {datasets.train[1]["augmentations"] for _ in range(8)}
    assert len(traces) > 1


def test_validation_requirements(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path
) -> None:
    with pytest.raises(
        ValueError, match="None of the configured datasets has validation splits"
    ):
        TrainValDatasets.from_pydantic_model(
            _config(storage, ntire_raw, mllm_raw, val_splits=[])
        )
    with pytest.raises(ValueError, match="both training and validation"):
        TrainValDatasets.from_pydantic_model(
            _config(
                storage,
                ntire_raw,
                mllm_raw,
                train_splits=["shard_0"],
                val_splits=["shard_0"],
            )
        )
    with pytest.raises(ValueError, match="more than once"):
        TrainDataConfig.model_validate(
            {
                "storage": storage.model_dump(),
                "datasets": [{"name": "ntire"}, {"name": "ntire"}],
                "preprocessing": {"image_size": 32},
            }
        )
