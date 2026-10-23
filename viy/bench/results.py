from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from kostyl import StrictPydanticBaseModel

from viy.data import SplitDerivation

from .metrics import SliceMetrics
from .model import ModelMeta


type AggregateValue = float | str | None


class TaskResult(StrictPydanticBaseModel):
    """
    Result of one model on one task.

    Attributes:
        task (str): Task name, e.g. ``ntire/val_hard``.
        dataset (str): Dataset name.
        split (str): Split name.
        derived_split (str | None): Derived split actually evaluated, if any.
        derivation (SplitDerivation | None): How ``derived_split`` was built from ``split``.
        model (ModelMeta): Evaluated model.
        viy_version (str): Library version that produced the result.
        evaluated_at (datetime): Evaluation time.
        threshold (float): Decision threshold of threshold-dependent metrics.
        main_score_name (str): Metric reported in summary tables.
        main_score (float | None): Value of the main metric on the whole split.
        slices (dict[str, SliceMetrics]): Selected metrics per slice, starting with ``all``.
        aggregates (dict[str, AggregateValue]): Macro and worst values of every
            selected metric over generator, distortion and augmentation slices.
        evaluation_seconds (float): Wall time of inference and scoring.
    """

    task: str
    dataset: str
    split: str
    derived_split: str | None = None
    derivation: SplitDerivation | None = None
    model: ModelMeta
    viy_version: str
    evaluated_at: datetime
    threshold: float
    main_score_name: str
    main_score: float | None
    slices: dict[str, SliceMetrics]
    aggregates: dict[str, AggregateValue]
    evaluation_seconds: float

    def save(self, path: Path) -> None:
        """Write the result as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return

    @classmethod
    def load(cls, path: Path) -> "TaskResult":
        """Read a result written by :meth:`save`."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def load_results(directory: Path) -> list[TaskResult]:
    """Read every result stored under ``directory``."""
    return [TaskResult.load(path) for path in sorted(Path(directory).rglob("*.json"))]


@dataclass(frozen=True, slots=True)
class ResultsMatrix:
    """
    Results laid out with models as rows and tasks as columns.

    Attributes:
        models (tuple[str, ...]): Model labels, ``name@revision`` when a revision is set.
        tasks (tuple[str, ...]): Task names.
        cells (Mapping[tuple[str, str], str]): Formatted value per ``(model, task)``;
            pairs without a result are absent.
    """

    models: tuple[str, ...]
    tasks: tuple[str, ...]
    cells: Mapping[tuple[str, str], str]

    def row(self, model: str) -> list[str]:
        """Cells of ``model`` in task order, empty where no result exists."""
        return [self.cells.get((model, task), "") for task in self.tasks]


def results_matrix(
    results: Sequence[TaskResult],
    metric: str | None = None,
    slice_name: str = "all",
    aggregate: str | None = None,
) -> ResultsMatrix:
    """
    Arrange results with models as rows and tasks as columns.

    Args:
        results (Sequence[TaskResult]): Results to arrange.
        metric (str | None): Metric to show from ``slice_name``; ``None`` shows each
            task's main score.
        slice_name (str): Slice the metric comes from, e.g. ``augmentation:jpeg``.
        aggregate (str | None): Aggregate to show instead of a metric, e.g.
            ``generator_worst_auc``.

    Returns:
        ResultsMatrix: Sorted models and tasks with formatted values; tasks
        without the requested slice or aggregate have empty cells.

    Raises:
        ValueError: If no result has the requested metric or aggregate.
    """
    if aggregate is not None and (metric is not None or slice_name != "all"):
        raise ValueError("Pass either an aggregate or a metric with its slice.")
    if aggregate is not None:
        _require_value(
            aggregate, {name for result in results for name in result.aggregates}
        )
    elif metric is not None:
        _require_value(
            metric,
            {
                name
                for result in results
                if slice_name in result.slices
                for name in result.slices[slice_name].metrics
            },
            f" on slice '{slice_name}'",
        )
    cells: dict[tuple[str, str], str] = {}
    for result in results:
        value = _value(result, metric, slice_name, aggregate)
        if value is not _MISSING:
            cells[_model_label(result.model), result.task] = _format(value)
    return ResultsMatrix(
        models=tuple(sorted({_model_label(result.model) for result in results})),
        tasks=tuple(sorted({result.task for result in results})),
        cells=cells,
    )


def results_table(
    results: Sequence[TaskResult],
    metric: str | None = None,
    slice_name: str = "all",
    aggregate: str | None = None,
) -> str:
    """
    Markdown table with models as rows and tasks as columns.

    Args:
        results (Sequence[TaskResult]): Results to tabulate.
        metric (str | None): Metric to show from ``slice_name``; ``None`` shows each
            task's main score.
        slice_name (str): Slice the metric comes from, e.g. ``augmentation:jpeg``.
        aggregate (str | None): Aggregate to show instead of a metric, e.g.
            ``generator_worst_auc``.

    Returns:
        str: Markdown table.
    """
    matrix = results_matrix(results, metric, slice_name, aggregate)
    lines = [
        "| model | " + " | ".join(matrix.tasks) + " |",
        "|---|" + "---:|" * len(matrix.tasks),
    ]
    lines.extend(
        f"| {model} | " + " | ".join(matrix.row(model)) + " |"
        for model in matrix.models
    )
    return "\n".join(lines)


_MISSING: Final = object()


def _value(
    result: TaskResult,
    metric: str | None,
    slice_name: str,
    aggregate: str | None,
) -> object:
    if aggregate is not None:
        return result.aggregates.get(aggregate, _MISSING)
    if metric is None:
        return result.main_score
    if slice_name not in result.slices:
        return _MISSING
    return result.slices[slice_name].metrics.get(metric, _MISSING)


def _require_value(name: str, available: set[str], where: str = "") -> None:
    if name not in available:
        raise ValueError(
            f"No result has '{name}'{where}. Available: {', '.join(sorted(available)) or 'none'}."
        )
    return


def _format(value: object) -> str:
    match value:
        case None:
            return "n/a"
        case float() | int():
            return f"{value:.4f}"
        case _:
            return str(value)


def _model_label(model: ModelMeta) -> str:
    return model.name if model.revision is None else f"{model.name}@{model.revision}"
