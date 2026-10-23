from collections.abc import Mapping
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from importlib.metadata import version
from typing import Self

import numpy as np
from kostyl import StrictPydanticBaseModel
from pydantic import model_validator

from viy.data import SplitDerivation
from viy.datasets import DATASETS
from viy.datasets import SplitRole
from viy.layout import SampleRecord
from viy.registry import Registry

from .metrics import DEFAULT_METRICS
from .metrics import METRICS
from .metrics import Scores
from .metrics import SliceMetrics
from .metrics import compute_metrics
from .model import ModelMeta
from .results import AggregateValue
from .results import TaskResult
from .slices import build_slices


AGGREGATED_PREFIXES = ("generator", "distortion", "augmentation")


class TaskMetadata(StrictPydanticBaseModel):
    """
    Description of a benchmark task.

    Attributes:
        name (str): Unique task name; defaults to ``<dataset>/<split>``, or to
            ``<dataset>/<derived split>`` for a task with a derivation.
        dataset (str): Registered dataset name.
        split (str): Split evaluated by the task.
        description (str): Short description.
        main_score (str): Metric of the ``all`` slice used in summary tables; it
            is computed even when a benchmark does not select it.
        derivation (SplitDerivation | None): Subsampling and augmentation applied
            to ``split`` once, with a fixed seed, before evaluation; ``None``
            evaluates ``split`` as it is.
    """

    name: str = ""
    dataset: str
    split: str
    description: str = ""
    main_score: str = "auc"
    derivation: SplitDerivation | None = None

    @property
    def evaluates_train_split(self) -> bool:
        """Whether the task scores a training split; models trained on it must skip the task."""
        spec = DATASETS.get(self.dataset).split_spec(self.split)
        return spec is not None and spec.role == SplitRole.TRAIN

    @property
    def derived_split(self) -> str | None:
        """Name of the derived split the task evaluates, or ``None`` without a derivation."""
        if self.derivation is None:
            return None
        return self.derivation.split_name(self.split)

    @model_validator(mode="after")
    def _default_name(self) -> Self:
        if not self.name:
            self.name = f"{self.dataset}/{self.derived_split or self.split}"
        return self


class DetectionTask:
    """Scores of a model on one split turned into metrics per slice."""

    def __init__(self, metadata: TaskMetadata) -> None:
        """
        Initialize the task.

        Args:
            metadata (TaskMetadata): Task description.
        """
        self.metadata = metadata
        return

    @classmethod
    def from_pydantic_model(cls, config: TaskMetadata) -> "DetectionTask":
        """Build the task from its description."""
        return cls(config)

    @property
    def name(self) -> str:
        """Task name."""
        return self.metadata.name

    def evaluate(
        self,
        records: Sequence[SampleRecord],
        scores: Scores,
        threshold: float,
        model: ModelMeta,
        evaluation_seconds: float,
        metrics: Sequence[str] = DEFAULT_METRICS,
    ) -> TaskResult:
        """
        Compute the selected metrics of every slice of the split.

        Args:
            records (Sequence[SampleRecord]): Records of the evaluated split,
                derived if the task has a derivation.
            scores (Scores): Score of every record.
            threshold (float): Decision threshold of threshold-dependent metrics.
            model (ModelMeta): Evaluated model.
            evaluation_seconds (float): Wall time spent on the task.
            metrics (Sequence[str]): Names of metrics from the pool; the main
                score is added when missing.

        Returns:
            TaskResult: Result of the task.
        """
        names = list(dict.fromkeys([*metrics, self.metadata.main_score]))
        labels = np.asarray([record.label for record in records], dtype=np.int64)
        slices = {
            item.name: compute_metrics(
                labels[item.indices], scores[item.indices], threshold, names
            )
            for item in build_slices(records)
        }
        return TaskResult(
            task=self.name,
            dataset=self.metadata.dataset,
            split=self.metadata.split,
            derived_split=self.metadata.derived_split,
            derivation=self.metadata.derivation,
            model=model,
            viy_version=version("viy"),
            evaluated_at=datetime.now(UTC),
            threshold=threshold,
            main_score_name=self.metadata.main_score,
            main_score=slices["all"].metrics[self.metadata.main_score],
            slices=slices,
            aggregates=aggregate(slices, names),
            evaluation_seconds=evaluation_seconds,
        )


def aggregate(
    slices: Mapping[str, SliceMetrics], metrics: Sequence[str]
) -> dict[str, AggregateValue]:
    """
    Macro and worst values of every metric over generator, distortion and augmentation slices.

    Level slices are left out, and so are descriptive metrics without a direction.

    Args:
        slices (Mapping[str, SliceMetrics]): Metrics per slice.
        metrics (Sequence[str]): Metrics to aggregate.

    Returns:
        dict[str, AggregateValue]: ``<prefix>_macro_<metric>``, ``<prefix>_worst_<metric>``
        and ``<prefix>_worst_<metric>_slice`` for every prefix and metric with at
        least one defined value.
    """
    aggregates: dict[str, AggregateValue] = {}
    for prefix in AGGREGATED_PREFIXES:
        group = {
            name.removeprefix(f"{prefix}:"): values.metrics
            for name, values in slices.items()
            if name.startswith(f"{prefix}:") and "@" not in name
        }
        for metric in metrics:
            higher_is_better = METRICS.get(metric).higher_is_better
            defined = {
                name: value
                for name, values in group.items()
                if (value := values.get(metric)) is not None
            }
            if higher_is_better is None or not defined:
                continue
            pick = min if higher_is_better else max
            worst = pick(defined, key=defined.__getitem__)
            aggregates[f"{prefix}_macro_{metric}"] = float(
                np.mean(list(defined.values()))
            )
            aggregates[f"{prefix}_worst_{metric}"] = defined[worst]
            aggregates[f"{prefix}_worst_{metric}_slice"] = worst
    return aggregates


CUSTOM_TASKS: Registry[DetectionTask] = Registry("task")


def register_task(task: DetectionTask | TaskMetadata, replace: bool = False) -> None:
    """
    Make a custom task available to :func:`get_tasks` and to config files.

    Args:
        task (DetectionTask | TaskMetadata): Task or its description.
        replace (bool): Replace a task of the same name, e.g. when a notebook
            cell runs again.
    """
    task = DetectionTask(task) if isinstance(task, TaskMetadata) else task
    CUSTOM_TASKS.add(task.name, task, replace=replace)
    return


def builtin_tasks() -> list[DetectionTask]:
    """
    One task per validation or test split of every registered dataset.

    A dataset without such splits gets tasks on its training splits, which
    report ``evaluates_train_split``.
    """
    tasks = []
    for dataset in DATASETS.names():
        builder_cls = DATASETS.get(dataset)
        for split in builder_cls.eval_splits() or builder_cls.train_role_splits():
            spec = builder_cls.split_spec(split)
            tasks.append(
                DetectionTask(
                    TaskMetadata(
                        name=f"{dataset}/{split}",
                        dataset=dataset,
                        split=split,
                        description="" if spec is None else spec.description,
                    )
                )
            )
    return tasks


def get_tasks(names: str | Sequence[str] | None = None) -> list[DetectionTask]:
    """
    Select tasks by name.

    A dataset name selects the tasks of the dataset whose data can be
    downloaded. Splits that exist only as local copies, such as
    ``ntire/test_private``, are selected by their full task name.

    Args:
        names (str | Sequence[str] | None): Task names such as ``ntire/val`` or
            dataset names such as ``ntire``; ``None`` selects every task whose
            data can be downloaded.

    Returns:
        list[DetectionTask]: Selected tasks without repeats.
    """
    available = {task.name: task for task in builtin_tasks()}
    available.update({name: CUSTOM_TASKS.get(name) for name in CUSTOM_TASKS.names()})
    if names is None:
        return [task for task in available.values() if _downloadable(task)]
    selected: dict[str, DetectionTask] = {}
    for name in [names] if isinstance(names, str) else names:
        matched = [
            task
            for task in available.values()
            if name == task.name
            or (name == task.metadata.dataset and _downloadable(task))
        ]
        if not matched:
            raise KeyError(
                f"Unknown task or dataset '{name}'. Available tasks: {', '.join(available)}."
            )
        selected.update((task.name, task) for task in matched)
    return list(selected.values())


def _downloadable(task: DetectionTask) -> bool:
    spec = DATASETS.get(task.metadata.dataset).split_spec(task.metadata.split)
    return spec is None or not spec.local_only
