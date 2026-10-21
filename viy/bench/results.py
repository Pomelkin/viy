from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from kostyl import StrictPydanticBaseModel

from .metrics import BinaryMetrics
from .model import ModelMeta


type AggregateValue = float | str | None


class TaskResult(StrictPydanticBaseModel):
    """
    Result of one model on one task.

    Attributes:
        task (str): Task name, e.g. ``ntire/val_hard``.
        dataset (str): Dataset name.
        split (str): Split name.
        model (ModelMeta): Evaluated model.
        viy_version (str): Library version that produced the result.
        evaluated_at (datetime): Evaluation time.
        threshold (float): Decision threshold of threshold-dependent metrics.
        main_score_name (str): Metric reported in summary tables.
        main_score (float | None): Value of the main metric on the whole split.
        slices (dict[str, BinaryMetrics]): Metrics per slice, starting with ``all``.
        aggregates (dict[str, AggregateValue]): Macro and worst-case values over
            generator and distortion slices.
        evaluation_seconds (float): Wall time of inference and scoring.
    """

    task: str
    dataset: str
    split: str
    model: ModelMeta
    viy_version: str
    evaluated_at: datetime
    threshold: float
    main_score_name: str
    main_score: float | None
    slices: dict[str, BinaryMetrics]
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


def results_table(results: Sequence[TaskResult], metric: str | None = None) -> str:
    """
    Markdown table with models as rows and tasks as columns.

    Args:
        results (Sequence[TaskResult]): Results to tabulate.
        metric (str | None): Metric of the ``all`` slice to show; ``None`` shows
            each task's main score.

    Returns:
        str: Markdown table.
    """
    tasks = sorted({result.task for result in results})
    models = sorted({_model_label(result.model) for result in results})
    cells = {
        (_model_label(result.model), result.task): _cell(result, metric)
        for result in results
    }
    lines = [
        "| model | " + " | ".join(tasks) + " |",
        "|---|" + "---:|" * len(tasks),
    ]
    lines.extend(
        f"| {model} | "
        + " | ".join(cells.get((model, task), "") for task in tasks)
        + " |"
        for model in models
    )
    return "\n".join(lines)


def _model_label(model: ModelMeta) -> str:
    return model.name if model.revision is None else f"{model.name}@{model.revision}"


def _cell(result: TaskResult, metric: str | None) -> str:
    value = (
        result.main_score if metric is None else getattr(result.slices["all"], metric)
    )
    return "n/a" if value is None else f"{value:.4f}"
