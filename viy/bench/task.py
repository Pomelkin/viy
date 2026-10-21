from collections.abc import Mapping
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from importlib.metadata import version
from typing import Self

import numpy as np
from kostyl import StrictPydanticBaseModel
from pydantic import model_validator

from viy.datasets import DATASETS
from viy.layout import SampleRecord
from viy.registry import Registry

from .metrics import BinaryMetrics
from .metrics import Scores
from .metrics import binary_metrics
from .model import ModelMeta
from .results import AggregateValue
from .results import TaskResult
from .slices import build_slices


AGGREGATED_PREFIXES = ("generator", "distortion")


class TaskMetadata(StrictPydanticBaseModel):
    """
    Description of a benchmark task.

    Attributes:
        name (str): Unique task name, ``<dataset>/<split>`` for built-in tasks.
        dataset (str): Registered dataset name.
        split (str): Split evaluated by the task.
        description (str): Short description.
        main_score (str): Metric of the ``all`` slice used in summary tables.
        evaluates_train_split (bool): The dataset has no evaluation split, so the
            task scores its training split; models trained on it must skip the task.
    """

    name: str
    dataset: str
    split: str
    description: str = ""
    main_score: str = "auc"
    evaluates_train_split: bool = False

    @model_validator(mode="after")
    def _validate_main_score(self) -> Self:
        if self.main_score not in BinaryMetrics.model_fields:
            raise ValueError(
                f"Unknown main score '{self.main_score}'. Available: {', '.join(BinaryMetrics.model_fields)}."
            )
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
    ) -> TaskResult:
        """
        Compute metrics of every slice of the split.

        Args:
            records (Sequence[SampleRecord]): Records of the split.
            scores (Scores): Score of every record.
            threshold (float): Decision threshold of threshold-dependent metrics.
            model (ModelMeta): Evaluated model.
            evaluation_seconds (float): Wall time spent on the task.

        Returns:
            TaskResult: Result of the task.
        """
        labels = np.asarray([record.label for record in records], dtype=np.int64)
        slices = {
            item.name: binary_metrics(
                labels[item.indices], scores[item.indices], threshold
            )
            for item in build_slices(records)
        }
        return TaskResult(
            task=self.name,
            dataset=self.metadata.dataset,
            split=self.metadata.split,
            model=model,
            viy_version=version("viy"),
            evaluated_at=datetime.now(UTC),
            threshold=threshold,
            main_score_name=self.metadata.main_score,
            main_score=getattr(slices["all"], self.metadata.main_score),
            slices=slices,
            aggregates=aggregate(slices),
            evaluation_seconds=evaluation_seconds,
        )


def aggregate(slices: Mapping[str, BinaryMetrics]) -> dict[str, AggregateValue]:
    """
    Macro and worst-case AUC over generator and distortion slices.

    Args:
        slices (Mapping[str, BinaryMetrics]): Metrics per slice.

    Returns:
        dict[str, AggregateValue]: ``<prefix>_macro_auc``, ``<prefix>_worst_auc``
        and ``<prefix>_worst`` for every prefix with at least one scored slice.
    """
    aggregates: dict[str, AggregateValue] = {}
    for prefix in AGGREGATED_PREFIXES:
        aucs = {
            name.removeprefix(f"{prefix}:"): metrics.auc
            for name, metrics in slices.items()
            if name.startswith(f"{prefix}:")
            and "@" not in name
            and metrics.auc is not None
        }
        if not aucs:
            continue
        worst = min(aucs, key=lambda name: aucs[name] or 0.0)
        aggregates[f"{prefix}_macro_auc"] = float(
            np.mean([auc for auc in aucs.values() if auc is not None])
        )
        aggregates[f"{prefix}_worst_auc"] = aucs[worst]
        aggregates[f"{prefix}_worst"] = worst
    return aggregates


CUSTOM_TASKS: Registry[DetectionTask] = Registry("task")


def register_task(task: DetectionTask) -> None:
    """Make a custom task available to :func:`get_tasks`."""
    CUSTOM_TASKS.add(task.name, task)
    return


def builtin_tasks() -> list[DetectionTask]:
    """
    One task per validation or test split of every registered dataset.

    A dataset without such splits gets tasks on its training splits, flagged
    with ``evaluates_train_split``.
    """
    tasks = []
    for dataset in DATASETS.names():
        builder_cls = DATASETS.get(dataset)
        splits = builder_cls.eval_splits()
        on_train = len(splits) == 0
        for split in builder_cls.train_role_splits() if on_train else splits:
            spec = builder_cls.split_spec(split)
            tasks.append(
                DetectionTask(
                    TaskMetadata(
                        name=f"{dataset}/{split}",
                        dataset=dataset,
                        split=split,
                        description="" if spec is None else spec.description,
                        evaluates_train_split=on_train,
                    )
                )
            )
    return tasks


def get_tasks(names: Sequence[str] | None = None) -> list[DetectionTask]:
    """
    Select tasks by name.

    Args:
        names (Sequence[str] | None): Task names such as ``ntire/val`` or dataset
            names such as ``ntire`` selecting all tasks of the dataset; ``None``
            selects every task.

    Returns:
        list[DetectionTask]: Selected tasks without repeats.
    """
    available = {task.name: task for task in builtin_tasks()}
    available.update({name: CUSTOM_TASKS.get(name) for name in CUSTOM_TASKS.names()})
    if names is None:
        return list(available.values())
    selected: dict[str, DetectionTask] = {}
    for name in names:
        matched = [
            task
            for task in available.values()
            if name in {task.name, task.metadata.dataset}
        ]
        if not matched:
            raise KeyError(
                f"Unknown task or dataset '{name}'. Available tasks: {', '.join(available)}."
            )
        selected.update((task.name, task) for task in matched)
    return list(selected.values())
