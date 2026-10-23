import csv
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

from viy.augs import trace_to_json
from viy.augs import AppliedOp
from viy.bench import Benchmark
from viy.bench import BenchmarkConfig
from viy.bench import DetectionTask
from viy.bench import DetectorModel
from viy.bench import ModelConfig
from viy.bench import ModelMeta
from viy.bench import SplitDerivation
from viy.bench import TaskMetadata
from viy.bench import register_task
from viy.bench import task as task_module
from viy.registry import Registry
from viy.bench import METRICS
from viy.bench import Metric
from viy.bench import compute_metrics
from viy.bench import register_metric
from viy.bench import build_slices
from viy.bench import get_tasks
from viy.bench import load_results
from viy.bench import results_table
from viy.data import AUGMENTATIONS_COLUMN
from viy.data import Preprocessor
from viy.datasets import NTIRE
from viy.datasets import StorageConfig
from viy.layout import LayoutError
from viy.layout import SampleRecord


class BrightnessDetector(DetectorModel):
    """Scores images by mean brightness, which separates the synthetic fixture classes."""

    def __init__(self, name: str = "brightness", revision: str | None = None) -> None:
        self.preprocessor = Preprocessor(
            image_size=16, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        )
        self.calls = 0
        self.name = name
        self.revision = revision

    @property
    @override
    def meta(self) -> ModelMeta:
        return ModelMeta(name=self.name, revision=self.revision, n_parameters=0)

    @override
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.preprocessor(image)

    @override
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return batch.mean(dim=(1, 2, 3))


def test_metric_pool() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.4, 0.6, 0.35, 0.8, 0.9], dtype=np.float64)
    values = compute_metrics(labels, scores, 0.5, METRICS.names()).metrics
    assert values["auc"] == pytest.approx(roc_auc_score(labels, scores))
    assert values["average_precision"] == pytest.approx(
        average_precision_score(labels, scores)
    )
    assert values["eer"] == pytest.approx(1 / 3)
    assert values["fpr"] == pytest.approx(1 / 3)
    assert values["fnr"] == pytest.approx(1 / 3)
    assert values["recall"] == pytest.approx(2 / 3)
    assert values["precision"] == pytest.approx(2 / 3)
    assert values["accuracy"] == pytest.approx(4 / 6)
    assert values["f1"] == pytest.approx(2 / 3)
    assert values["mcc"] == pytest.approx(matthews_corrcoef(labels, scores >= 0.5))
    assert values["log_loss"] == pytest.approx(log_loss(labels, scores))
    assert values["brier"] == pytest.approx(brier_score_loss(labels, scores))
    assert values["tpr_at_1pct_fpr"] == pytest.approx(2 / 3)
    assert values["fpr_at_95pct_tpr"] == pytest.approx(2 / 3)
    assert values["mean_score_real"] == pytest.approx(1.1 / 3)
    calibrated = compute_metrics(labels, labels.astype(np.float64), 0.5, ["ece"])
    assert calibrated.metrics["ece"] == pytest.approx(0.0)

    single = compute_metrics(
        labels[:3], scores[:3], 0.5, ["auc", "eer", "fnr", "f1", "accuracy"]
    )
    assert single.metrics == {
        "auc": None,
        "eer": None,
        "fnr": None,
        "f1": None,
        "accuracy": pytest.approx(2 / 3),
    }
    logits = compute_metrics(labels, scores * 10 - 5, 0.0, ["log_loss", "auc"])
    assert logits.metrics["log_loss"] is None
    assert logits.metrics["auc"] == pytest.approx(values["auc"])
    with pytest.raises(KeyError, match="Unknown metric 'nope'"):
        compute_metrics(labels, scores, 0.5, ["nope"])


def test_selected_metrics_and_rescoring(
    ntire_raw: Path, storage: StorageConfig, tmp_path: Path
) -> None:
    config = BenchmarkConfig(
        storage=storage,
        output_dir=tmp_path / "results",
        tasks=["ntire/test_private"],
        raw_paths={"ntire": ntire_raw},
        metrics=["eer", "fpr", "mean_score_real"],
        batch_size=4,
        num_workers=0,
    )
    model = BrightnessDetector()
    (result,) = Benchmark.from_pydantic_model(config).run(model)
    assert set(result.slices["all"].metrics) == {"eer", "fpr", "mean_score_real", "auc"}
    assert result.aggregates["generator_worst_eer"] == pytest.approx(0.0)
    assert "generator_worst_eer_slice" in result.aggregates
    assert "generator_macro_fpr" in result.aggregates
    assert not any("mean_score" in key for key in result.aggregates)

    calls = model.calls
    wider = config.model_copy(update={"metrics": ["eer", "brier", "mcc"]})
    (rescored,) = Benchmark.from_pydantic_model(wider).run(model)
    assert model.calls == calls
    assert {"brier", "mcc"} <= set(rescored.slices["all"].metrics)
    assert rescored.slices["all"].metrics["eer"] == result.slices["all"].metrics["eer"]
    with pytest.raises(KeyError, match="Unknown metric"):
        Benchmark.from_pydantic_model(config.model_copy(update={"metrics": ["auroc"]}))


def test_custom_metric() -> None:
    register_metric(
        Metric(
            "score_gap",
            "mean fake score minus mean real score",
            lambda scored: float(
                scored.scores[scored.fake].mean() - scored.scores[scored.real].mean()
            ),
        )
    )
    labels = np.array([0, 1], dtype=np.int64)
    values = compute_metrics(labels, np.array([0.2, 0.7]), 0.5, ["score_gap"]).metrics
    assert values["score_gap"] == pytest.approx(0.5)
    assert (
        TaskMetadata(
            name="a/b", dataset="ntire", split="val", main_score="score_gap"
        ).main_score
        == "score_gap"
    )


def test_slices(ntire_raw: Path, storage: StorageConfig) -> None:
    builder = NTIRE(raw_path=ntire_raw, storage=storage)
    names = [item.name for item in build_slices(builder.load("test_private"))]
    assert names[:3] == ["all", "clean", "distorted"]
    assert "distortion:watermark@pixelseal" in names
    assert "distortion:downscale" in names
    assert "generator:hidream" in names
    assert [item.name for item in build_slices(builder.load("shard_0"))] == ["all"]


def test_task_selection() -> None:
    names = [task.name for task in get_tasks("ntire")]
    assert names == ["ntire/val", "ntire/val_hard", "ntire/test_public"]
    assert [task.name for task in get_tasks(["ntire", "ntire/test_private"])][
        -1
    ] == "ntire/test_private"
    mllm = get_tasks(["mllm_generated/train"])[0]
    assert mllm.metadata.evaluates_train_split
    with pytest.raises(KeyError, match="Unknown task"):
        get_tasks(["nope"])


def test_benchmark_run(
    ntire_raw: Path, mllm_raw: Path, storage: StorageConfig, tmp_path: Path
) -> None:
    config = BenchmarkConfig(
        storage=storage,
        output_dir=tmp_path / "results",
        tasks=["ntire/test_private", "mllm_generated"],
        raw_paths={"ntire": ntire_raw, "mllm_generated": mllm_raw},
        batch_size=4,
        num_workers=0,
    )
    benchmark = Benchmark.from_pydantic_model(config)
    model = BrightnessDetector()
    results = benchmark.run(model)
    assert [result.task for result in results] == [
        "ntire/test_private",
        "mllm_generated/train",
    ]
    private = results[0]
    assert private.main_score == pytest.approx(1.0)
    assert private.slices["all"].num_samples == 6
    assert private.aggregates["generator_worst_auc"] == pytest.approx(1.0)
    assert "distortion_macro_auc" in private.aggregates

    result_path = benchmark.result_path(model.meta, benchmark.tasks[0])
    assert result_path.name == "ntire__test_private.json"
    with result_path.with_suffix(".predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {"image", "label", "score"} == set(rows[0])

    calls = model.calls
    assert [result.task for result in benchmark.run(model)] == [
        result.task for result in results
    ]
    assert model.calls == calls

    other = BrightnessDetector(revision="v2")
    Benchmark.from_pydantic_model(
        config.model_copy(update={"tasks": ["ntire/test_private"]})
    ).run(other)
    loaded = load_results(config.output_dir)
    assert len(loaded) == 3
    table = results_table(loaded)
    assert "| brightness |" in table
    assert "| brightness@v2 |" in table
    assert "ntire/test_private" in table.splitlines()[0]


def _augmented_record(index: int, trace: list[AppliedOp]) -> SampleRecord:
    return SampleRecord(
        image=f"images/{index}.png",
        label=index % 2,
        extra={AUGMENTATIONS_COLUMN: trace_to_json(trace)},
    )


def test_augmentation_slices() -> None:
    records = [
        _augmented_record(0, []),
        _augmented_record(1, [AppliedOp("jpeg", 50), AppliedOp("jpeg", 30)]),
        _augmented_record(2, [AppliedOp("albumentations.HorizontalFlip")]),
        _augmented_record(3, [AppliedOp("jpeg", 50)]),
    ]
    slices = {item.name: item.indices.tolist() for item in build_slices(records)}
    assert slices["not_augmented"] == [0]
    assert slices["augmented"] == [1, 2, 3]
    assert slices["augmentation:jpeg"] == [1, 3]
    assert slices["augmentation:jpeg@50"] == [1, 3]
    assert slices["augmentation:jpeg@30"] == [1]
    assert slices["augmentation:albumentations.HorizontalFlip"] == [2]
    assert not any(
        name.startswith("augmentation:albumentations.HorizontalFlip@")
        for name in slices
    )
    assert "clean" not in slices


def _derived_config(
    mllm_raw: Path, storage: StorageConfig, output_dir: Path, seed: int
) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(
        {
            "storage": storage.model_dump(),
            "output_dir": output_dir,
            "tasks": [],
            "custom_tasks": [
                {
                    "name": "mllm_generated/val_mix",
                    "dataset": "mllm_generated",
                    "split": "train",
                    "derivation": {
                        "seed": seed,
                        "limit": 12,
                        "augmentation": {
                            "steps": [
                                {
                                    "kind": "ntire_preset",
                                    "preset": "val",
                                    "p": 0.8,
                                }
                            ]
                        },
                    },
                }
            ],
            "raw_paths": {"mllm_generated": mllm_raw},
            "batch_size": 4,
            "num_workers": 0,
        }
    )


def test_custom_derived_task(
    mllm_raw: Path, storage: StorageConfig, tmp_path: Path
) -> None:
    config = _derived_config(mllm_raw, storage, tmp_path / "results", seed=5)
    model = BrightnessDetector()
    (result,) = Benchmark.from_pydantic_model(config).run(model)
    derivation = config.custom_tasks[0].derivation
    assert derivation is not None
    assert result.task == "mllm_generated/val_mix"
    assert result.split == "train"
    assert result.derived_split == derivation.split_name("train")
    assert result.derivation == derivation
    assert result.slices["all"].num_samples == 12
    assert "augmented" in result.slices
    assert any(name.startswith("augmentation:") for name in result.slices)
    assert "augmentation_macro_auc" in result.aggregates

    calls = model.calls
    Benchmark.from_pydantic_model(config).run(model)
    assert model.calls == calls
    changed = _derived_config(mllm_raw, storage, tmp_path / "results", seed=6)
    with pytest.raises(ValueError, match="Rename the task or set overwrite=True"):
        Benchmark.from_pydantic_model(changed).run(model)
    rerun = Benchmark.from_pydantic_model(
        changed.model_copy(update={"overwrite": True})
    )
    assert rerun.run(model)[0].derivation == changed.custom_tasks[0].derivation


def test_task_names_are_unique(tmp_path: Path) -> None:
    task = DetectionTask(TaskMetadata(name="x/y", dataset="ntire", split="val"))
    with pytest.raises(ValueError, match="unique"):
        Benchmark([task, task], output_dir=tmp_path)


def test_model_from_config() -> None:
    model = DetectorModel.from_pydantic_model(
        ModelConfig(target="test_bench:BrightnessDetector", kwargs={"revision": "r1"})
    )
    assert model.meta.revision == "r1"
    with pytest.raises(TypeError, match="not a DetectorModel"):
        DetectorModel.from_pydantic_model(
            ModelConfig(target="test_bench:ModelMeta", kwargs={"name": "x"})
        )
    with pytest.raises(ValidationError, match="pattern"):
        ModelConfig(target="no_colon")


def test_df26_task(df26_raw: Path, storage: StorageConfig, tmp_path: Path) -> None:
    config = BenchmarkConfig(
        storage=storage,
        output_dir=tmp_path / "results",
        tasks=["df26"],
        raw_paths={"df26": df26_raw},
        batch_size=4,
        num_workers=0,
    )
    (result,) = Benchmark.from_pydantic_model(config).run(BrightnessDetector())
    assert result.task == "df26/test"
    assert result.main_score == pytest.approx(1.0)
    assert result.slices["all"].num_samples == 20
    assert result.slices["generator:Veo_3.1"].num_samples == 4 + 8
    assert "generator_worst_auc" in result.aggregates


def test_run_checks_data_before_inference(
    mllm_raw: Path, storage: StorageConfig, tmp_path: Path
) -> None:
    config = BenchmarkConfig(
        storage=storage,
        output_dir=tmp_path / "results",
        tasks=["mllm_generated", "ntire/test_private"],
        raw_paths={"mllm_generated": mllm_raw},
        num_workers=0,
    )
    model = BrightnessDetector()
    with pytest.raises(LayoutError, match="ntire/test_private"):
        Benchmark.from_pydantic_model(config).run(model)
    assert model.calls == 0


def test_benchmark_accepts_names_and_descriptions(
    ntire_raw: Path,
    mllm_raw: Path,
    storage: StorageConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(task_module, "CUSTOM_TASKS", Registry("task"))
    derived = TaskMetadata(
        dataset="mllm_generated",
        split="train",
        derivation=SplitDerivation(seed=1, limit=6),
    )
    assert derived.name == f"mllm_generated/{derived.derived_split}"
    assert derived.evaluates_train_split
    benchmark = Benchmark(
        ["ntire/test_private", derived],
        output_dir=tmp_path / "results",
        storage=storage,
        raw_paths={"ntire": ntire_raw, "mllm_generated": mllm_raw},
        metrics="auc",
        num_workers=0,
    )
    results = benchmark.run(BrightnessDetector(revision="r"))
    assert [result.task for result in results] == ["ntire/test_private", derived.name]

    assert "generator:hidream" in results[0].slices
    by_slice = results_table(results, metric="auc", slice_name="generator:hidream")
    assert "1.0000" in by_slice
    worst = results_table(results, aggregate="generator_worst_auc_slice")
    assert "| brightness@r |" in worst
    with pytest.raises(ValueError, match="No result has 'eer'"):
        results_table(results, metric="eer")

    register_task(TaskMetadata(dataset="ntire", split="val", name="custom/val"))
    register_task(
        TaskMetadata(dataset="ntire", split="val_hard", name="custom/val"), replace=True
    )
    assert get_tasks("custom/val")[0].metadata.split == "val_hard"


def test_level_slices_sort_numerically() -> None:
    records = [
        SampleRecord(
            image=f"{index}.png",
            label=index % 2,
            distortions=("jpeg",),
            distortion_scales=(level,),
        )
        for index, level in enumerate([7, 28, 7, 45, 28, 45])
    ]
    names = [item.name for item in build_slices(records) if "@" in item.name]
    assert names == ["distortion:jpeg@7", "distortion:jpeg@28", "distortion:jpeg@45"]
