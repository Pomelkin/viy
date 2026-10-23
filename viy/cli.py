import importlib
import sys
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import rich_click as click
from rich.console import Console
from rich.table import Column
from rich.table import Table

from viy import load_config
from viy.bench import DEFAULT_METRICS
from viy.bench import METRICS
from viy.bench import Benchmark
from viy.bench import BenchmarkRunConfig
from viy.bench import DetectorModel
from viy.bench import TaskResult
from viy.bench import get_tasks
from viy.bench import load_results
from viy.bench import results_matrix
from viy.data import derived_info_path
from viy.data import read_derived_info
from viy.datasets import DATASETS
from viy.datasets import DatasetBuilder
from viy.datasets import StorageConfig


console = Console()

_ROOT_OPTION = click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory of canonical datasets; defaults to $VIY_ROOT or ~/.cache/viy.",
)


_RAW_PATH_OPTION = click.option(
    "--raw-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Local copy of the raw dataset; files are downloaded from the Hub otherwise.",
)


@click.group()
@click.version_option(package_name="viy")
@click.option(
    "--import",
    "imports",
    multiple=True,
    help="Module to import first, e.g. one that registers datasets, tasks or metrics; repeatable.",
)
def main(imports: Sequence[str]) -> None:
    """
    Datasets, augmentations and a benchmark for AI-generated image detection.

    The current directory is importable, so `--import my_datasets` loads
    `my_datasets.py` from it.
    """
    _import_modules(imports, Path.cwd())
    return


@main.group()
def bench() -> None:
    """Evaluate detectors on benchmark tasks."""
    return


@bench.command("run")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--overwrite", is_flag=True, help="Recompute tasks that already have results."
)
def bench_run(config: Path, overwrite: bool) -> None:
    """
    Evaluate the detector described in CONFIG.

    CONFIG is a YAML or JSON file with the fields of `BenchmarkConfig`, a
    `model` entry, `{target: package.module:Detector, kwargs: {...}}`, and
    optional `imports`, modules that register datasets, tasks or metrics. The
    directory of CONFIG and the current directory are importable, so the
    detector can live next to the config. Relative paths in CONFIG resolve
    against the current directory.
    """
    run_config = BenchmarkRunConfig.model_validate(load_config(config))
    if overwrite:
        run_config = run_config.model_copy(update={"overwrite": True})
    _import_modules(run_config.imports, config.resolve().parent)
    model = DetectorModel.from_pydantic_model(run_config.model)
    results = Benchmark.from_pydantic_model(run_config).run(model)
    console.print(_summary_table(results))
    return


@bench.command("metrics")
def bench_metrics() -> None:
    """List the metric pool; select metrics with the `metrics` list of the config."""
    table = Table(
        "Metric", "Better", "Threshold", "Default", "Meaning", title="Metrics"
    )
    arrows = {True: "higher", False: "lower", None: "—"}
    for name in METRICS.names():
        metric = METRICS.get(name)
        table.add_row(
            name,
            arrows[metric.higher_is_better],
            "yes" if metric.threshold else "",
            "✓" if name in DEFAULT_METRICS else "",
            metric.description,
        )
    console.print(table)
    return


@bench.command("tasks")
@click.argument("names", nargs=-1)
def bench_tasks(names: Sequence[str]) -> None:
    """List built-in and registered tasks, or only those of NAMES."""
    table = Table(_folding("Task"), "Split", "Description", title="Benchmark tasks")
    for task in get_tasks(list(names) or None):
        description = task.metadata.description
        if task.metadata.evaluates_train_split:
            description = f"{description} [yellow](training split)[/yellow]"
        table.add_row(task.name, task.metadata.split, description)
    console.print(table)
    return


@bench.command("results")
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--metric",
    default=None,
    help="Metric to show; the main score of every task by default.",
)
@click.option(
    "--slice",
    "slice_name",
    default="all",
    show_default=True,
    help="Slice the metric comes from, e.g. augmentation:jpeg or generator:Veo_3.1.",
)
@click.option(
    "--aggregate",
    default=None,
    help="Aggregate to show instead of a metric, e.g. generator_worst_auc.",
)
def bench_results(
    directory: Path, metric: str | None, slice_name: str, aggregate: str | None
) -> None:
    """Compare the results stored in DIRECTORY, models as rows and tasks as columns."""
    results = load_results(directory)
    if aggregate is not None:
        _require_choice(
            aggregate,
            {name for result in results for name in result.aggregates},
            "--aggregate",
        )
    elif metric is not None:
        _require_choice(
            slice_name,
            {name for result in results for name in result.slices},
            "--slice",
        )
        _require_choice(
            metric,
            {
                name
                for result in results
                if slice_name in result.slices
                for name in result.slices[slice_name].metrics
            },
            "--metric",
        )
    matrix = results_matrix(results, metric, slice_name, aggregate)
    title = aggregate or (f"{metric} on {slice_name}" if metric else "Main score")
    table = Table(_folding("Model"), title=title)
    for task in matrix.tasks:
        table.add_column(task, justify="right", overflow="fold")
    for model in matrix.models:
        table.add_row(model, *matrix.row(model))
    console.print(table)
    return


@main.group()
def datasets() -> None:
    """Prepare and inspect datasets in the canonical layout."""
    return


@datasets.command("list")
@_ROOT_OPTION
def datasets_list(root: Path | None) -> None:
    """List the supported datasets and every split with whether it is prepared."""
    storage = StorageConfig() if root is None else StorageConfig(root=root)
    datasets_table = Table("Dataset", "Description", title="Datasets")
    for name in DATASETS.names():
        builder_cls = DATASETS.get(name)
        note = (
            " [yellow](benchmark only)[/yellow]" if builder_cls.benchmark_only else ""
        )
        datasets_table.add_row(name, builder_cls.description + note)
    console.print(datasets_table)
    table = Table("Dataset", "Split", "Role", "Prepared", "Description", title="Splits")
    for name in DATASETS.names():
        builder = DATASETS.get(name)(storage=storage)
        known = {spec.name for spec in builder.split_specs}
        for spec in builder.split_specs:
            table.add_row(
                name,
                spec.name,
                str(spec.role),
                _mark(prepared=builder.layout.has_split(spec.name)),
                spec.description,
            )
        for split in builder.layout.splits():
            if split not in known:
                table.add_row(
                    name, split, "", _mark(prepared=True), _custom_split(builder, split)
                )
    console.print(table)
    return


@datasets.command("check")
@click.argument("name")
@click.option(
    "--split",
    "splits",
    multiple=True,
    help="Split or split group to check; repeatable. Defaults to every split.",
)
@_RAW_PATH_OPTION
@_ROOT_OPTION
def datasets_check(
    name: str, splits: Sequence[str], raw_path: Path | None, root: Path | None
) -> None:
    """
    Check that splits of dataset NAME are prepared or can be prepared, without downloading them.

    Covers local copies of splits that are not on the Hub and access to gated
    datasets with your Hugging Face token.
    """
    builder = _builder(name, raw_path, root)
    requested = splits or [spec.name for spec in builder.split_specs]
    for split in builder.expand_splits(requested):
        builder.check_available(split)
        console.print(f"[green]✓[/green] {name}/{split}")
    return


@datasets.command("prepare")
@click.argument("name")
@click.option(
    "--split",
    "splits",
    multiple=True,
    help="Split or split group to prepare; repeatable. Defaults to the training and validation splits, or to the evaluation splits of a benchmark-only dataset.",
)
@_RAW_PATH_OPTION
@_ROOT_OPTION
def datasets_prepare(
    name: str, splits: Sequence[str], raw_path: Path | None, root: Path | None
) -> None:
    """Download and convert splits of dataset NAME ahead of training or benchmarking."""
    builder = _builder(name, raw_path, root)
    defaults = (
        builder.eval_splits()
        if builder.benchmark_only
        else (*builder.default_train_splits, *builder.default_val_splits)
    )
    requested = splits or defaults
    for split in builder.expand_splits(requested):
        builder.prepare(split)
        console.print(f"[green]✓[/green] {name}/{split}")
    return


def _builder(name: str, raw_path: Path | None, root: Path | None) -> DatasetBuilder:
    if name not in DATASETS:
        raise click.BadParameter(
            f"Unknown dataset '{name}'. Available: {', '.join(DATASETS.names())}. "
            "Load a module that defines more with --import.",
            param_hint="NAME",
        )
    storage = StorageConfig() if root is None else StorageConfig(root=root)
    return DATASETS.get(name)(raw_path=raw_path, storage=storage)


def _import_modules(modules: Iterable[str], directory: Path) -> None:
    for path in (Path.cwd(), directory):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for module in modules:
        importlib.import_module(module)
    return


def _summary_table(results: Sequence[TaskResult]) -> Table:
    table = Table(
        _folding("Task"),
        _folding("Evaluated split"),
        "Metric",
        "Score",
        "Images",
        "Seconds",
    )
    for result in results:
        score = "n/a" if result.main_score is None else f"{result.main_score:.4f}"
        table.add_row(
            result.task,
            result.derived_split or result.split,
            result.main_score_name,
            score,
            str(result.slices["all"].num_samples),
            f"{result.evaluation_seconds:.1f}",
        )
    return table


def _folding(header: str) -> Column:
    return Column(header, overflow="fold")


def _require_choice(value: str, available: set[str], option: str) -> None:
    if value not in available:
        raise click.BadParameter(
            f"'{value}' is not in the results. Available: {', '.join(sorted(available))}.",
            param_hint=option,
        )
    return


def _mark(*, prepared: bool) -> str:
    return "[green]yes[/green]" if prepared else "[dim]no[/dim]"


def _custom_split(builder: DatasetBuilder, split: str) -> str:
    if not derived_info_path(builder.layout, split).is_file():
        return "custom split"
    info = read_derived_info(builder.layout, split)
    return f"derived from '{info.split}' with seed {info.derivation.seed}"
