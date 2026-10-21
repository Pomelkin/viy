import shutil
import zipfile
from pathlib import Path

import pytest

from viy.datasets import NTIRE
from viy.datasets import DatasetBuilder
from viy.datasets import DatasetSourceConfig
from viy.datasets import MLLMGenerated
from viy.datasets import StorageConfig
from viy.datasets import hub
from viy.layout import LayoutError
from viy.layout import SampleRecord
from viy.layout import write_records


def test_ntire_prepares_every_split(ntire_raw: Path, storage: StorageConfig) -> None:
    builder = DatasetBuilder.from_pydantic_model(
        DatasetSourceConfig(name="ntire", raw_path=ntire_raw), storage
    )
    assert isinstance(builder, NTIRE)
    assert builder.expand_splits(["train", "shard_6", "shard_0"]) == [
        *(f"shard_{i}" for i in range(6)),
        "shard_6",
    ]

    shard = builder.load("shard_3")
    assert len(shard) == 4
    assert shard[0].image == "images/shard_3/s3_000.jpg"
    assert shard[0].distortions is None
    linked = builder.layout.image_path(shard[0])
    assert linked.is_symlink()
    assert (
        linked.resolve()
        == (ntire_raw / "train" / "shard_3" / "images" / "s3_000.jpg").resolve()
    )

    val = builder.load("val")
    assert [record.distortions for record in val[:3]] == [(), (), ("jpeg", "downscale")]
    assert val[2].distortion_scales == (28, 0.61)

    shard6 = builder.load("shard_6")
    assert shard6[0].distortions == ("jpeg_ai",)
    assert shard6[0].distortion_scales == (2,)
    assert shard6[1].distortion_scales == (0.5,)
    assert shard6[1].source_id == "s0_001.jpg"

    private = builder.load("test_private")
    assert [record.generator for record in private] == [
        None,
        "hidream",
        None,
        "gpt-image-1.5",
        None,
        "hidream",
    ]
    assert private[2].distortion_scales == ("pixelseal", 0.5)

    manifest = builder.layout.read_manifest()
    assert set(manifest.splits) == {"shard_3", "val", "shard_6", "test_private"}
    assert manifest.splits["val"].num_fake == 4


def test_prepared_split_is_reused(ntire_raw: Path, storage: StorageConfig) -> None:
    builder = NTIRE(raw_path=ntire_raw, storage=storage)
    builder.load("val")
    csv_path = builder.layout.split_path("val")
    stamp = csv_path.stat().st_mtime_ns
    (ntire_raw / "val" / "val_labels.csv").unlink()
    assert len(NTIRE(raw_path=ntire_raw, storage=storage).load("val")) == 8
    assert csv_path.stat().st_mtime_ns == stamp


def test_preparation_errors(ntire_raw: Path, storage: StorageConfig) -> None:
    builder = NTIRE(raw_path=ntire_raw, storage=storage)
    with pytest.raises(LayoutError, match="group"):
        builder.load("train")
    with pytest.raises(LayoutError, match="Unknown split"):
        builder.load("nope")
    with pytest.raises(LayoutError, match="auto_prepare"):
        NTIRE(
            raw_path=ntire_raw,
            storage=storage.model_copy(update={"auto_prepare": False}),
        ).load("val")
    with pytest.raises(LayoutError, match="not published"):
        NTIRE(storage=storage).load("test_private")


def test_custom_split_from_existing_images(
    ntire_raw: Path, storage: StorageConfig
) -> None:
    builder = NTIRE(raw_path=ntire_raw, storage=storage)
    train = builder.load("shard_0")
    write_records(builder.layout.split_path("my_val"), train[:2])
    assert builder.load("my_val") == train[:2]

    write_records(
        builder.layout.split_path("broken"),
        [SampleRecord(image="images/shard_0/ghost.jpg", label=0)],
    )
    with pytest.raises(LayoutError, match="missing"):
        builder.load("broken")


def test_hub_download_extracts_archives(
    ntire_raw: Path,
    storage: StorageConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "hub_val"
    repo.mkdir()
    with zipfile.ZipFile(repo / "val_images.zip", "w") as archive:
        for image in sorted((ntire_raw / "val" / "val_images").iterdir()):
            archive.write(image, f"val_images/{image.name}")
    shutil.copy(ntire_raw / "val" / "val_labels.csv", repo / "val_labels.csv")
    calls: list[str] = []

    def fake_snapshot_download(
        repo_id: str, repo_type: str, allow_patterns: list[str], local_dir: Path
    ) -> str:
        assert repo_type == "dataset"
        calls.append(f"{repo_id}:{allow_patterns[0]}")
        for pattern in allow_patterns:
            for path in repo.glob(pattern):
                shutil.copy(path, Path(local_dir) / path.name)
        return str(local_dir)

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download)
    builder = NTIRE(storage=storage)
    records = builder.load("val")
    assert len(records) == 8
    raw_val = builder.layout.raw_dir / "val"
    assert not (raw_val / "val_images.zip").exists()
    assert (raw_val / "val_images" / "v0000.jpg").is_file()
    assert calls == [
        "deepfakesMSU/NTIRE-RobustAIGenDetection-val:val_images.zip",
        "deepfakesMSU/NTIRE-RobustAIGenDetection-val:val_labels.csv",
    ]
    spec = NTIRE.split_spec("val")
    assert spec is not None
    hub.fetch_artifact(spec.artifacts[0], builder.layout.raw_dir)
    assert len(calls) == 2


def test_mllm_generated(mllm_raw: Path, storage: StorageConfig) -> None:
    builder = MLLMGenerated(raw_path=mllm_raw, storage=storage)
    assert builder.default_val_splits == ()
    records = builder.load("train")
    assert len(records) == 18
    real = [record for record in records if record.label == 0]
    assert len(real) == 6
    assert {record.generator for record in records} == {None, "GPT Image 2", "FLUX.2"}
    assert {record.extra["domain"] for record in records} == {"hybrid", "texture"}
    assert "images/gpt_image_2/hybrid/0000.png" in {record.image for record in records}


def test_default_storage_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VIY_ROOT", str(tmp_path / "custom"))
    assert StorageConfig().root == tmp_path / "custom"
    assert NTIRE().layout.dir == tmp_path / "custom" / "ntire"
    monkeypatch.delenv("VIY_ROOT")
    assert StorageConfig().root == Path.home() / ".cache" / "viy"
