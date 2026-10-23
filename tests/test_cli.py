from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from viy.cli import main
from viy.datasets import DATASETS
from viy.datasets import StorageConfig
from viy.datasets import hub


def _write_config(path: Path, mllm_raw: Path, storage: StorageConfig) -> Path:
    config = {
        "model": {
            "target": "test_bench:BrightnessDetector",
            "kwargs": {"revision": "cli"},
        },
        "storage": {"root": str(storage.root)},
        "output_dir": str(path.parent / "results"),
        "tasks": ["mllm_generated"],
        "custom_tasks": [
            {
                "name": "mllm_generated/jpeg",
                "dataset": "mllm_generated",
                "split": "train",
                "derivation": {
                    "seed": 0,
                    "augmentation": {"steps": [{"kind": "op", "op": "jpeg"}]},
                },
            }
        ],
        "raw_paths": {"mllm_generated": str(mllm_raw)},
        "batch_size": 4,
        "num_workers": 0,
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_bench_run_and_results(
    mllm_raw: Path, storage: StorageConfig, tmp_path: Path
) -> None:
    config = _write_config(tmp_path / "bench.yaml", mllm_raw, storage)
    runner = CliRunner()
    run = runner.invoke(main, ["bench", "run", str(config)])
    assert run.exit_code == 0, run.output
    assert "mllm_generated/jpeg" in run.output
    assert (
        tmp_path / "results" / "brightness__cli" / "mllm_generated__jpeg.json"
    ).is_file()

    results = runner.invoke(
        main, ["bench", "results", str(tmp_path / "results"), "--metric", "fpr"]
    )
    assert results.exit_code == 0, results.output
    assert "brightness@cli" in results.output

    listing = runner.invoke(main, ["datasets", "list", "--root", str(storage.root)])
    assert listing.exit_code == 0, listing.output
    assert "derived from" in listing.output


def test_bench_tasks() -> None:
    result = CliRunner().invoke(main, ["bench", "tasks", "ntire"])
    assert result.exit_code == 0, result.output
    assert "ntire/val_hard" in result.output
    assert "mllm_generated" not in result.output


def test_datasets_prepare(mllm_raw: Path, storage: StorageConfig) -> None:
    result = CliRunner().invoke(
        main,
        [
            "datasets",
            "prepare",
            "mllm_generated",
            "--raw-path",
            str(mllm_raw),
            "--root",
            str(storage.root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mllm_generated/train" in result.output
    assert (storage.root / "mllm_generated" / "splits" / "train.csv").is_file()


def test_datasets_check(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApi:
        def auth_check(self, repo_id: str, repo_type: str) -> None:
            assert (repo_id, repo_type) == ("DF26/DF26", "dataset")

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub, "get_token", lambda: "token")
    result = CliRunner().invoke(main, ["datasets", "check", "df26"])
    assert result.exit_code == 0, result.output
    assert "df26/test" in result.output

    monkeypatch.setattr(hub, "get_token", lambda: None)
    result = CliRunner().invoke(main, ["datasets", "check", "df26"])
    assert isinstance(result.exception, PermissionError)


def test_datasets_prepare_benchmark_only(
    df26_raw: Path, storage: StorageConfig
) -> None:
    result = CliRunner().invoke(
        main,
        [
            "datasets",
            "prepare",
            "df26",
            "--raw-path",
            str(df26_raw),
            "--root",
            str(storage.root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "df26/test" in result.output


def test_bench_metrics() -> None:
    result = CliRunner().invoke(main, ["bench", "metrics"])
    assert result.exit_code == 0, result.output
    assert "eer" in result.output
    assert "fpr_at_95pct_tpr" in result.output


def test_imports_and_config_directory(
    mllm_raw: Path,
    storage: StorageConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "cli_detectors.py").write_text(
        "from test_bench import BrightnessDetector\n"
        "from viy.bench import Metric, register_metric\n"
        "register_metric(Metric('always_one', 'constant', lambda scored: 1.0), replace=True)\n",
        encoding="utf-8",
    )
    config = _write_config(configs / "bench.yaml", mllm_raw, storage)
    data = yaml.safe_load(config.read_text())
    data["model"]["target"] = "cli_detectors:BrightnessDetector"
    data["metrics"] = ["auc", "always_one"]
    data["custom_tasks"][0]["main_score"] = "always_one"
    config.write_text(yaml.safe_dump(data))
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["bench", "run", str(config)])
    assert result.exit_code == 0, result.output
    results = runner.invoke(
        main,
        [
            "bench",
            "results",
            str(configs / "results"),
            "--aggregate",
            "generator_worst_auc",
        ],
    )
    assert results.exit_code == 0, results.output
    assert "brightness@cli" in results.output
    bad = runner.invoke(
        main,
        [
            "bench",
            "results",
            str(configs / "results"),
            "--slice",
            "nope",
            "--metric",
            "auc",
        ],
    )
    assert bad.exit_code != 0
    assert "Available" in bad.output

    monkeypatch.setattr(DATASETS, "_items", dict(DATASETS._items))  # noqa: SLF001
    (tmp_path / "extra_datasets.py").write_text(
        "from viy.datasets import MLLMGenerated\n\n"
        "class Mirror(MLLMGenerated):\n    name = 'mirror'\n",
        encoding="utf-8",
    )
    listing = runner.invoke(
        main,
        [
            "--import",
            "extra_datasets",
            "datasets",
            "check",
            "mirror",
            "--raw-path",
            str(mllm_raw),
        ],
    )
    assert listing.exit_code == 0, listing.output
    assert "mirror/train" in listing.output
    unknown = runner.invoke(main, ["datasets", "check", "nope"])
    assert unknown.exit_code != 0
    assert "--import" in unknown.output
