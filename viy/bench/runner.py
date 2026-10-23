import csv
import re
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import override

import numpy as np
import torch
from kostyl import StrictPydanticBaseModel
from kostyl.utils import setup_logger
from PIL import Image
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from viy.data import derive_split
from viy.datasets import DATASETS
from viy.datasets import DatasetBuilder
from viy.datasets import StorageConfig
from viy.layout import SampleRecord
from viy.progress import track

from .metrics import DEFAULT_METRICS
from .metrics import METRICS
from .metrics import Scores
from .model import DetectorModel
from .model import ModelConfig
from .model import ModelMeta
from .results import TaskResult
from .task import DetectionTask
from .task import TaskMetadata
from .task import get_tasks


logger = setup_logger(fmt="detailed")


class BenchmarkConfig(StrictPydanticBaseModel):
    """
    Benchmark run settings.

    Attributes:
        storage (StorageConfig): Canonical layout location and preparation settings.
        output_dir (Path): Directory results are written to, one folder per model.
        tasks (list[str]): Built-in or registered task or dataset names, e.g.
            ``ntire/val`` or ``ntire``; ``[]`` runs only ``custom_tasks``.
        custom_tasks (list[TaskMetadata]): Tasks defined in the config, run after
            the selected ones, e.g. augmented copies of a split.
        raw_paths (dict[str, Path]): Local raw copies by dataset name; datasets
            not listed are downloaded from the Hub.
        metrics (list[str]): Metrics of the pool computed on every slice; see
            ``viy bench metrics``. Each task's main score is added.
        batch_size (int): Inference batch size.
        num_workers (int): DataLoader workers decoding and preprocessing images.
        overwrite (bool): Recompute tasks that already have results.
    """

    storage: StorageConfig = Field(default_factory=StorageConfig)
    output_dir: Path
    tasks: list[str]
    custom_tasks: list[TaskMetadata] = Field(default_factory=list)
    raw_paths: dict[str, Path] = Field(default_factory=dict)
    metrics: list[str] = Field(
        default_factory=lambda: list(DEFAULT_METRICS), min_length=1
    )
    batch_size: int = Field(default=64, gt=0)
    num_workers: int = Field(default=8, ge=0)
    overwrite: bool = False


class BenchmarkRunConfig(BenchmarkConfig):
    """
    A benchmark run of one detector, as read by ``viy bench run``.

    Attributes:
        model (ModelConfig): Detector to evaluate.
        imports (list[str]): Modules imported before the run, e.g. ones that
            register datasets, tasks or metrics.
    """

    model: ModelConfig
    imports: list[str] = Field(default_factory=list)


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
    Per-image scores are stored next to every result. When a stored result
    lacks a selected metric, the metrics are recomputed from these scores
    without running the model.

    Example:
        >>> benchmark = Benchmark(["ntire/val", "ntire/test_public"], output_dir="results")
        >>> results = benchmark.run(MyDetector())
        >>> print(results_table(results))
    """

    def __init__(
        self,
        tasks: str | Sequence[str | TaskMetadata | DetectionTask],
        output_dir: Path | str,
        storage: StorageConfig | None = None,
        raw_paths: Mapping[str, Path | str] | None = None,
        metrics: str | Sequence[str] = DEFAULT_METRICS,
        batch_size: int = 64,
        num_workers: int = 8,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize the benchmark.

        Args:
            tasks (str | Sequence[str | TaskMetadata | DetectionTask]): Tasks to run
                in order: task or dataset names as for :func:`get_tasks`, task
                descriptions or tasks. Task names must be unique.
            output_dir (Path | str): Directory results are written to.
            storage (StorageConfig | None): Canonical layout location and preparation
                settings; ``None`` uses the defaults.
            raw_paths (Mapping[str, Path | str] | None): Local raw copies by
                dataset name; other datasets are downloaded from the Hub.
            metrics (str | Sequence[str]): Metrics of the pool computed on every
                slice, see ``viy bench metrics``.
            batch_size (int): Inference batch size.
            num_workers (int): DataLoader workers decoding and preprocessing images.
            overwrite (bool): Recompute tasks that already have results.
        """
        self.tasks = _resolve_tasks(tasks)
        if len(self.tasks) == 0:
            raise ValueError("A benchmark needs at least one task.")
        names = [task.name for task in self.tasks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Task names must be unique: {duplicates}.")
        for task in self.tasks:
            METRICS.get(task.metadata.main_score)
        self.storage = StorageConfig() if storage is None else storage
        self.output_dir = Path(output_dir)
        self.raw_paths = {name: Path(path) for name, path in (raw_paths or {}).items()}
        self.metrics = [
            METRICS.get(name).name
            for name in ([metrics] if isinstance(metrics, str) else metrics)
        ]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.overwrite = overwrite
        return

    @classmethod
    def from_pydantic_model(cls, config: BenchmarkConfig) -> "Benchmark":
        """Build the benchmark from its configuration."""
        return cls(
            tasks=[*config.tasks, *config.custom_tasks],
            storage=config.storage,
            output_dir=config.output_dir,
            raw_paths=config.raw_paths,
            metrics=config.metrics,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            overwrite=config.overwrite,
        )

    def run(self, model: DetectorModel) -> list[TaskResult]:
        """
        Evaluate ``model`` on every task.

        Before any inference, every task without a stored result is checked
        for available data, so a missing local copy or gated access fails the
        run at once instead of after hours of evaluation.

        Args:
            model (DetectorModel): Detector to evaluate.

        Returns:
            list[TaskResult]: Results in task order.
        """
        if model.meta.revision is None:
            logger.warning(
                f"Model '{model.meta.name}' has no revision, so its results are reused "
                "for any later version of it; set ModelMeta.revision to keep them apart."
            )
        for task in self.tasks:
            if self.overwrite or not self.result_path(model.meta, task).is_file():
                builder = self._builder(task.metadata.dataset)
                builder.check_available(task.metadata.split)
        results = []
        for task in self.tasks:
            path = self.result_path(model.meta, task)
            if path.is_file() and not self.overwrite:
                logger.info(
                    f"Task '{task.name}' already evaluated for '{model.meta.name}', loading {path}"
                )
                stored = _load_matching(path, task)
                missing = set(self.metrics) - set(stored.slices["all"].metrics)
                results.append(
                    self._rescore(model, task, path, stored) if missing else stored
                )
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
        split = task.metadata.split
        if task.metadata.derivation is not None:
            split = derive_split(
                builder, split, task.metadata.derivation, self.num_workers
            )
        records = builder.load(split)
        started = time.perf_counter()
        scores = self._predict(
            model, [builder.layout.image_path(record) for record in records], task.name
        )
        result = task.evaluate(
            records,
            scores,
            model.threshold,
            model.meta,
            time.perf_counter() - started,
            self.metrics,
        )
        result.save(path)
        _write_predictions(path.with_suffix(".predictions.csv"), records, scores)
        logger.info(
            f"Task '{task.name}': {result.main_score_name} = {result.main_score}"
        )
        return result

    def _rescore(
        self, model: DetectorModel, task: DetectionTask, path: Path, stored: TaskResult
    ) -> TaskResult:
        logger.info(
            f"Task '{task.name}': computing {self.metrics} from the stored predictions"
        )
        builder = self._builder(task.metadata.dataset)
        records = builder.load(stored.derived_split or stored.split)
        scores = _read_predictions(path.with_suffix(".predictions.csv"), records)
        result = task.evaluate(
            records,
            scores,
            model.threshold,
            model.meta,
            stored.evaluation_seconds,
            self.metrics,
        )
        result.save(path)
        return result

    def _builder(self, dataset: str) -> DatasetBuilder:
        return DATASETS.get(dataset)(
            raw_path=self.raw_paths.get(dataset), storage=self.storage
        )

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
        batches = iter(loader)
        with torch.inference_mode():
            for images, indices in track(batches, task_name, total=len(loader)):
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


def _resolve_tasks(
    tasks: str | Sequence[str | TaskMetadata | DetectionTask],
) -> list[DetectionTask]:
    resolved: list[DetectionTask] = []
    for task in [tasks] if isinstance(tasks, str) else tasks:
        match task:
            case str():
                resolved.extend(get_tasks(task))
            case TaskMetadata():
                resolved.append(DetectionTask(task))
            case DetectionTask():
                resolved.append(task)
    return resolved


def _load_matching(path: Path, task: DetectionTask) -> TaskResult:
    result = TaskResult.load(path)
    stored = result.derived_split or result.split
    expected = task.metadata.derived_split or task.metadata.split
    if stored != expected:
        raise ValueError(
            f"{path} holds a result on split '{stored}', but task '{task.name}' now "
            f"evaluates '{expected}'. Rename the task or set overwrite=True."
        )
    return result


def _read_predictions(path: Path, records: Sequence[SampleRecord]) -> Scores:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [row["image"] for row in rows] != [record.image for record in records]:
        raise ValueError(
            f"{path} does not list the images of the split in order; set overwrite=True "
            "to run the model again."
        )
    return np.asarray([float(row["score"]) for row in rows], dtype=np.float64)


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
