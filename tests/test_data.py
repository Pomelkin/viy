from pathlib import Path

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from torch.utils.data import DataLoader
from transformers import ViTImageProcessorPil

from viy.augs import AppliedOp
from viy.augs import AugmentationConfig
from viy.augs import AugmentationPipeline
from viy.augs import NTIREPresetStep
from viy.augs import OpStep
from viy.data import DatasetSpec
from viy.data import DetectionCollator
from viy.data import DetectionDataset
from viy.data import DetectionDatasetConfig
from viy.data import PreprocessingConfig
from viy.data import Preprocessor
from viy.data import ResizeMode
from viy.data import Stage
from viy.data import check_disjoint
from viy.data import subsample
from viy.datasets import NTIRE
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
                NTIREPresetStep("aggressive_stage6", p=1.0),
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
    assert isinstance(sample["augmentations"], tuple)
    assert all(isinstance(op, AppliedOp) for op in sample["augmentations"])
    assert len({train[1]["augmentations"] for _ in range(8)}) > 1

    loader = DataLoader(
        train, batch_size=4, shuffle=True, collate_fn=DetectionCollator()
    )
    batch = next(iter(loader))
    assert batch["images"].shape == (4, 3, 32, 32)
    assert batch["labels"].dtype == torch.int64
    assert batch["labels"].shape == (4,)
    assert batch["indices"].shape == (4,)
    assert len(batch["sources"]) == 4
    assert len(batch["augmentations"]) == 4
    assert all(isinstance(trace, tuple) for trace in batch["augmentations"])


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
    loaders = {
        name: DataLoader(part, batch_size=4, collate_fn=DetectionCollator())
        for name, part in parts.items()
    }
    assert next(iter(loaders["ntire/val"]))["images"].shape == (4, 3, 32, 32)
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
            "preprocessor": {"image_size": 32},
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


def test_collator_rejects_mixed_shapes() -> None:
    sample = {
        "image": torch.zeros(3, 8, 8),
        "label": 0,
        "source": "a/b",
        "index": 0,
        "augmentations": (),
    }
    other = {**sample, "image": torch.zeros(3, 9, 8)}
    with pytest.raises(ValueError, match="equal shapes"):
        DetectionCollator()([sample, other])  # ty: ignore[invalid-argument-type]


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


def test_ranks_draw_different_augmentations(
    storage: StorageConfig,
    ntire_raw: Path,
    mllm_raw: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def traces(rank: int) -> list[tuple[AppliedOp, ...]]:
        monkeypatch.setattr("viy.data.dataset._process_rank", lambda: rank)
        torch.manual_seed(0)
        part = _train(storage, ntire_raw, mllm_raw).parts[-1]
        return [part[index]["augmentations"] for index in range(len(part))]

    assert traces(0) == traces(0)
    assert traces(0) != traces(1)


def test_convenient_inputs_and_loaders(
    storage: StorageConfig, ntire_raw: Path, mllm_raw: Path
) -> None:
    seen: list[type] = []

    def preprocess(image: Image.Image) -> torch.Tensor:
        seen.append(type(image))
        return Preprocessor(image_size=16)(image)

    train = DetectionDataset(
        [DatasetSpec("ntire", splits="shard_0", raw_path=ntire_raw)],
        stage="train",
        preprocessor=preprocess,
        augmentation=[A.HorizontalFlip(p=0.5), OpStep("jpeg", p=0.5)],
        storage=storage,
        seed=5,
    )
    assert train.sources == ["ntire/shard_0"]
    first = [batch["paths"] for batch in train.dataloader(batch_size=2)]
    second = [batch["paths"] for batch in train.dataloader(batch_size=2)]
    assert first == second
    assert sorted(path for paths in first for path in paths) == sorted(
        record.image for record in train.parts[0].records
    )
    assert seen and all(issubclass(kind, Image.Image) for kind in seen)

    for split in ("val", "val_hard"):
        NTIRE(raw_path=ntire_raw, storage=storage).prepare(split)
    val = DetectionDataset(
        "ntire",
        stage="val",
        preprocessor=Preprocessor(image_size=16),
        augmentation=AugmentationConfig.model_validate(
            {"steps": [{"kind": "op", "op": "jpeg"}]}
        ),
        storage=storage,
    )
    loaders = val.dataloaders(batch_size=4)
    assert list(loaders) == ["ntire/val", "ntire/val_hard"]
    batch = next(iter(loaders["ntire/val"]))
    record = val.record(batch["sources"][1], int(batch["indices"][1]))
    assert record.image == batch["paths"][1]


def test_preprocessor_loads_processor_by_name(tmp_path: Path) -> None:
    _vit_processor().save_pretrained(tmp_path / "processor")
    tensor = Preprocessor(image_processor=str(tmp_path / "processor"))(
        np.zeros((8, 8, 3), np.uint8)
    )
    assert tensor.shape == (3, 24, 24)


def test_seed_alone_reproduces_training_batches(
    storage: StorageConfig, ntire_raw: Path
) -> None:
    def batches(torch_seed: int) -> list[list[tuple[AppliedOp, ...]]]:
        torch.manual_seed(torch_seed)
        dataset = DetectionDataset(
            DatasetSpec("ntire", splits="shard_1", raw_path=ntire_raw),
            stage="train",
            preprocessor=Preprocessor(image_size=8),
            augmentation=[OpStep("jpeg", p=0.5), OpStep("brighten", p=0.5)],
            storage=storage,
            seed=11,
        )
        return [batch["augmentations"] for batch in dataset.dataloader(batch_size=2)]

    assert batches(1) == batches(2)
