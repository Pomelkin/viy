# viy

Data and benchmark library for detecting AI-generated images.

- **Datasets.** Every supported dataset is downloaded and converted on first use into one canonical layout of symlinked images and split CSV files, with sanity checks on every load.
- **Training data.** Describe datasets, preprocessing and augmentations in a pydantic config and get ready `torch` datasets for training and validation.
- **Augmentations.** NTIRE 2026 distortions under their metadata names, mixes reconstructed from the challenge splits, and any `albumentations` transform, all reproducible from a seed and traced per sample.
- **Benchmark.** Implement `preprocess` and `predict` on a model and run it on every evaluation split with one call. Metrics are reported per distortion, severity and generator.

## Install

```bash
uv add "viy @ git+<repo-url>"
```

Requires Python 3.13 or newer.

## Quick start

Training data:

```python
from torch.utils.data import DataLoader
from viy.data import TrainDataConfig, TrainValDatasets

config = TrainDataConfig.model_validate({
    "storage": {"root": "/data/viy"},
    "datasets": [{"name": "ntire", "train_splits": ["shard_0"]}],   # downloaded from the Hub on first use
    "preprocessing": {"image_size": 256},
    "train_augmentation": {"steps": [{"kind": "ntire_preset", "preset": "val"}]},
})
data = TrainValDatasets.from_pydantic_model(config)
train_loader = DataLoader(data.train, batch_size=64, shuffle=True, num_workers=8)
```

Benchmark:

```python
from pathlib import Path
from viy.bench import Benchmark, get_tasks, results_table
from viy.datasets import StorageConfig

benchmark = Benchmark(
    tasks=get_tasks(["ntire/val", "ntire/val_hard"]),
    storage=StorageConfig(root=Path("/data/viy")),
    output_dir=Path("results"),
)
print(results_table(benchmark.run(MyDetector())))   # MyDetector implements viy.bench.DetectorModel
```

## Documentation

| Page | Contents |
|---|---|
| [Getting started](docs/getting-started.md) | Install, storage, the first training loader and the first benchmark run |
| [Datasets](docs/datasets.md) | Canonical layout, supported datasets, custom splits, sanity checks |
| [Training data](docs/training.md) | `TrainDataConfig`, samples, validation rules, preprocessing, DDP and Lightning |
| [Augmentations](docs/augmentations.md) | Steps, NTIRE presets, operations reference, proxies |
| [Benchmark](docs/benchmark.md) | `DetectorModel`, tasks, results, metrics and slices |
| [Extending](docs/extending.md) | New datasets, operations and tasks |
| [Agent notes](docs/agents.md) | Architecture, invariants and conventions for coding agents |

## Supported datasets

| Name | Splits |
|---|---|
| `ntire` | NTIRE 2026 Robust AI-Generated Image Detection: training shards, `val`, `val_hard`, `test_public`; unpublished `shard_6` and `test_private` from a local copy |
| `mllm_generated` | MLLM-Generated Image Detection: GPT Image 2, Nano Banana 2, FLUX.2, Qwen Image 2.1 |

## Development

```bash
uv sync
uv run pytest
uv run ruff format && uv run ruff check viy && uv run ty check
```

## Limitations

- Learned codecs, adversarial attacks and watermarks are available only as proxies. Each proxy logs a warning when it is used.
- Downloading the NTIRE training shards from the Hub has not been tested on the real archives yet.
