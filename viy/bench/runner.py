import csv
import re
import time
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import override

import numpy as np
import torch
from kostyl import StrictPydanticBaseModel
from kostyl.utils import setup_logger
from kostyl.utils import tqdm
from PIL import Image
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from viy.datasets import DatasetBuilder
from viy.datasets import DatasetSourceConfig
from viy.datasets import StorageConfig
from viy.layout import SampleRecord

from .metrics import Scores
from .model import DetectorModel
from .model import ModelMeta
from .results import TaskResult
from .task import DetectionTask
from .task import get_tasks


logger = setup_logger(fmt="detailed")


class BenchmarkConfig(StrictPydanticBaseModel):
    """
    Benchmark run settings.

    Attributes:
        storage (StorageConfig): Canonical layout location and preparation settings.
        output_dir (Path): Directory results are written to, one folder per model.
        tasks (list[str] | None): Task or dataset names; ``None`` runs every task.
        sources (list[DatasetSourceConfig]): Local raw copies of datasets; datasets
            not listed are downloaded from the Hub.
        batch_size (int): Inference batch size.
        num_workers (int): DataLoader workers decoding and preprocessing images.
        overwrite (bool): Recompute tasks that already have results.
    """

    storage: StorageConfig = Field(default_factory=StorageConfig)
    output_dir: Path
    tasks: list[str] | None = None
    sources: list[DatasetSourceConfig] = Field(default_factory=list)
    batch_size: int = Field(default=64, gt=0)
    num_workers: int = Field(default=8, ge=0)
    overwrite: bool = False


class _InferenceDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, paths: Sequence[Path], preprocess: Callable[[Image.Image], torch.Tensor]
    ) -> None:
        self.paths = list(paths)
        self.preprocess = preprocess
        return

    def __len__(self) -> int:
        return len(self.paths)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB")), index


class Benchmark:
    """
    Runs a detector on benchmark tasks and stores one JSON result per task.

    Tasks with an existing result are loaded instead of recomputed unless
    ``overwrite`` is set, so an interrupted run continues where it stopped.
    Per-image scores are stored next to every result for later analysis.

    Example:
        >>> benchmark = Benchmark(
        ...     tasks=get_tasks(["ntire"]),
        ...     output_dir=Path("results"),
        ... )
        >>> results = benchmark.run(MyDetector())
        >>> print(results_table(results))
    """

    def __init__(
        self,
        tasks: Sequence[DetectionTask],
        output_dir: Path,
        storage: StorageConfig | None = None,
        sources: Sequence[DatasetSourceConfig] = (),
        batch_size: int = 64,
        num_workers: int = 8,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize the benchmark.

        Args:
            tasks (Sequence[DetectionTask]): Tasks to run in order.
            output_dir (Path): Directory results are written to.
            storage (StorageConfig | None): Canonical layout location and preparation
                settings; ``None`` uses the defaults.
            sources (Sequence[DatasetSourceConfig]): Local raw copies of datasets.
            batch_size (int): Inference batch size.
            num_workers (int): DataLoader workers decoding and preprocessing images.
            overwrite (bool): Recompute tasks that already have results.
        """
        if len(tasks) == 0:
            raise ValueError("A benchmark needs at least one task.")
        self.tasks = list(tasks)
        self.storage = StorageConfig() if storage is None else storage
        self.output_dir = Path(output_dir)
        self.sources = {source.name: source for source in sources}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.overwrite = overwrite
        return

    @classmethod
    def from_pydantic_model(cls, config: BenchmarkConfig) -> "Benchmark":
        """Build the benchmark from its configuration."""
        return cls(
            tasks=get_tasks(config.tasks),
            storage=config.storage,
            output_dir=config.output_dir,
            sources=config.sources,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            overwrite=config.overwrite,
        )

    def run(self, model: DetectorModel) -> list[TaskResult]:
        """
        Evaluate ``model`` on every task.

        Args:
            model (DetectorModel): Detector to evaluate.

        Returns:
            list[TaskResult]: Results in task order.
        """
        results = []
        for task in self.tasks:
            path = self.result_path(model.meta, task)
            if path.is_file() and not self.overwrite:
                logger.info(
                    f"Task '{task.name}' already evaluated for '{model.meta.name}', loading {path}"
                )
                results.append(TaskResult.load(path))
                continue
            results.append(self._run_task(model, task, path))
        return results

    def result_path(self, model: ModelMeta, task: DetectionTask) -> Path:
        """Location of the result of ``model`` on ``task``."""
        model_dir = (
            model.name if model.revision is None else f"{model.name}__{model.revision}"
        )
        return (
            self.output_dir
            / _safe(model_dir)
            / f"{_safe(task.name.replace('/', '__'))}.json"
        )

    def _run_task(
        self, model: DetectorModel, task: DetectionTask, path: Path
    ) -> TaskResult:
        if task.metadata.evaluates_train_split:
            logger.warning(
                f"Task '{task.name}' scores the training split because '{task.metadata.dataset}' "
                "has no evaluation split; skip it for models trained on this dataset."
            )
        builder = self._builder(task.metadata.dataset)
        records = builder.load(task.metadata.split)
        started = time.perf_counter()
        scores = self._predict(
            model, [builder.layout.image_path(record) for record in records], task.name
        )
        result = task.evaluate(
            records, scores, model.threshold, model.meta, time.perf_counter() - started
        )
        result.save(path)
        _write_predictions(path.with_suffix(".predictions.csv"), records, scores)
        logger.info(
            f"Task '{task.name}': {result.main_score_name} = {result.main_score}"
        )
        return result

    def _builder(self, dataset: str) -> DatasetBuilder:
        source = self.sources.get(dataset, DatasetSourceConfig(name=dataset))
        return DatasetBuilder.from_pydantic_model(source, self.storage)

    def _predict(
        self, model: DetectorModel, paths: Sequence[Path], task_name: str
    ) -> Scores:
        loader = DataLoader(
            _InferenceDataset(paths, model.preprocess),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        scores = np.full(len(paths), np.nan, dtype=np.float64)
        with torch.inference_mode():
            for images, indices in tqdm(loader, desc=task_name):
                values = (
                    model.predict(images).detach().float().reshape(-1).cpu().numpy()
                )
                if values.shape[0] != indices.shape[0]:
                    raise ValueError(
                        f"predict returned {values.shape[0]} scores for a batch of {indices.shape[0]} images."
                    )
                scores[indices.numpy()] = values
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"Model '{model.meta.name}' produced non-finite scores on '{task_name}'."
            )
        if scores.min() < 0.0 or scores.max() > 1.0:
            logger.warning(
                f"Scores of '{model.meta.name}' on '{task_name}' leave [0, 1]; "
                "threshold metrics assume probabilities."
            )
        return scores


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _write_predictions(
    path: Path, records: Sequence[SampleRecord], scores: Scores
) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "score"])
        writer.writerows(
            [record.image, record.label, float(score)]
            for record, score in zip(records, scores, strict=True)
        )
    temporary.replace(path)
    return
