import csv
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from viy.bench import Benchmark
from viy.bench import BenchmarkConfig
from viy.bench import DetectorModel
from viy.bench import ModelMeta
from viy.bench import binary_metrics
from viy.bench import build_slices
from viy.bench import get_tasks
from viy.bench import load_results
from viy.bench import results_table
from viy.data import Preprocessor
from viy.datasets import DatasetSourceConfig
from viy.datasets import NTIRE
from viy.datasets import StorageConfig


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


def test_binary_metrics() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.4, 0.6, 0.35, 0.8, 0.9], dtype=np.float64)
    metrics = binary_metrics(labels, scores, threshold=0.5)
    assert metrics.auc == pytest.approx(roc_auc_score(labels, scores))
    assert metrics.fpr == pytest.approx(1 / 3)
    assert metrics.fnr == pytest.approx(1 / 3)
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.mean_score_real == pytest.approx(1.1 / 3)
    single = binary_metrics(labels[:3], scores[:3], threshold=0.5)
    assert single.auc is None
    assert single.fnr is None
    assert single.f1 is None


def test_slices(ntire_raw: Path, storage: StorageConfig) -> None:
    builder = NTIRE(storage.root, raw_path=ntire_raw)
    names = [item.name for item in build_slices(builder.load("test_private"))]
    assert names[:3] == ["all", "clean", "distorted"]
    assert "distortion:watermark@pixelseal" in names
    assert "distortion:downscale" in names
    assert "generator:hidream" in names
    assert [item.name for item in build_slices(builder.load("shard_0"))] == ["all"]


def test_task_selection() -> None:
    names = [task.name for task in get_tasks(["ntire"])]
    assert names == [
        "ntire/val",
        "ntire/val_hard",
        "ntire/test_public",
        "ntire/test_private",
    ]
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
        sources=[
            DatasetSourceConfig(name="ntire", raw_path=ntire_raw),
            DatasetSourceConfig(name="mllm_generated", raw_path=mllm_raw),
        ],
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
