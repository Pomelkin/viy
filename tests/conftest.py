import csv
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from viy.datasets import StorageConfig


REAL_VALUE = 60
FAKE_VALUE = 190
PRIVATE_DIR = "test/new_test_images/ds/NTIRE_test_private/v1"


def write_image(
    path: Path, label: int, seed: int, size: tuple[int, int] = (48, 40)
) -> None:
    """Write a small noisy image whose brightness encodes the label."""
    rng = np.random.default_rng(seed)
    base = FAKE_VALUE if label else REAL_VALUE
    array = np.clip(rng.normal(base, 20, size=(size[1], size[0], 3)), 0, 255).astype(
        np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def write_csv(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _flat_split(
    raw: Path,
    images_dir: str,
    labels_csv: str,
    prefix: str,
    chains: Sequence[tuple[Sequence[object], Sequence[object]]],
) -> list[str]:
    names = []
    rows = []
    for index, (distortions, scales) in enumerate(chains):
        name = f"{prefix}{index:04d}.jpg"
        label = index % 2
        write_image(raw / images_dir / name, label, seed=_seed(name))
        rows.append(
            [
                index,
                name,
                label,
                repr(distortions),
                repr(scales),
                int(bool(distortions)),
            ]
        )
        names.append(name)
    write_csv(
        raw / labels_csv,
        ["", "image_name", "label", "distortions", "distortion_scales", "is_distorted"],
        rows,
    )
    return names


VAL_CHAINS = [
    ([], []),
    ([], []),
    (["jpeg", "downscale"], [28, 0.61]),
    (["jpeg", "downscale"], [12, 0.45]),
    (["lensblur"], [4]),
    (["lensblur"], [4]),
    (["colorsat", "jpeg"], [0.2, 28]),
    (["colorsat", "jpeg"], [-0.4, 7]),
]
TEST_CHAINS = [
    ([], []),
    ([], []),
    (["watermark", "downscale"], ["pixelseal", 0.5]),
    (["watermark", "downscale"], ["mbrs", 0.7]),
    (["jpeg_ai"], [0]),
    (["jpeg_ai"], [2]),
]
PRIVATE_GENERATORS = [None, "hidream", None, "gpt-image-1.5", None, "hidream"]


@pytest.fixture
def ntire_raw(tmp_path: Path) -> Path:
    """Raw NTIRE layout with every split, including the local ones."""
    raw = tmp_path / "raw_ntire"
    for shard in range(6):
        rows = []
        for index in range(4):
            name = f"s{shard}_{index:03d}.jpg"
            write_image(
                raw / "train" / f"shard_{shard}" / "images" / name,
                index % 2,
                seed=shard * 10 + index,
            )
            rows.append([index, name, index % 2])
        write_csv(
            raw / "train" / f"shard_{shard}" / "labels.csv",
            ["", "image_name", "label"],
            rows,
        )
    shard6_rows = [
        ["s6_real_jpeg_ai_0000.jpg", 0, "jpeg_ai", 2, "", "s0_000.jpg"],
        ["s6_ai_wmforger_0000.jpg", 1, "wmforger", "", 0.5, "s0_001.jpg"],
    ]
    for name, label, *_ in shard6_rows:
        write_image(
            raw / "train" / "shard_6" / "images" / str(name),
            int(label),
            seed=_seed(str(name)),
        )
    write_csv(
        raw / "train" / "shard_6" / "labels.csv",
        [
            "image_name",
            "label",
            "transform",
            "jpeg_ai_level",
            "wmforger_learning_rate",
            "source_image_name",
        ],
        shard6_rows,
    )
    _flat_split(raw, "val/val_images", "val/val_labels.csv", "v", VAL_CHAINS)
    _flat_split(raw, "val/val_images_hard", "val/val_hard_labels.csv", "h", VAL_CHAINS)
    _flat_split(raw, "test/test_images", "test/test_labels.csv", "t", TEST_CHAINS)
    names = _flat_split(
        raw,
        f"{PRIVATE_DIR}/test_set/test_images",
        f"{PRIVATE_DIR}/test_set/labels.csv",
        "p",
        TEST_CHAINS,
    )
    metadata = [
        [name.removesuffix(".jpg"), generator or "real"]
        for name, generator in zip(names, PRIVATE_GENERATORS, strict=True)
    ]
    write_csv(
        raw / PRIVATE_DIR / "clear_metadata.csv",
        ["val_hashed_name", "generator"],
        metadata[:3],
    )
    write_csv(
        raw / PRIVATE_DIR / "distorted_metadata.csv",
        ["val_hashed_name", "generator"],
        metadata[3:],
    )
    return raw


@pytest.fixture
def mllm_raw(tmp_path: Path) -> Path:
    """Raw MLLM-Generated layout with JPEG reals and PNG fakes."""
    raw = tmp_path / "raw_mllm"
    for generator in ("Real", "GPT Image 2", "FLUX.2"):
        for domain in ("Hybrid Images", "Texture Images"):
            for index in range(3):
                suffix = "jpg" if generator == "Real" else "png"
                path = raw / "images" / generator / domain / f"{index:04d}.{suffix}"
                write_image(path, int(generator != "Real"), seed=_seed(str(path)))
    return raw


@pytest.fixture
def storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig(root=tmp_path / "viy_root")
