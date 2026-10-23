# viy

Data and benchmark library for detecting AI-generated images.

- **Datasets.** Every supported dataset is downloaded and converted on first use into one canonical layout of images and split CSV files, with sanity checks on every load. Image folders are linked, Parquet shards and videos are converted.
- **Training data.** Describe datasets, preprocessing and augmentations in Python or in a pydantic config and get ready `torch` datasets and DataLoaders for training and validation, reproducible from one seed.
- **Augmentations.** NTIRE 2026 distortions under their metadata names, mixes reconstructed from the challenge splits, and any `albumentations` transform, all reproducible from a seed and traced per sample.
- **Benchmark.** Implement `preprocess` and `predict` on a model and run it on evaluation splits with one call or one command. Pick metrics from a pool; they are reported per distortion, severity and generator. Seeded, augmented copies of any split serve as robustness tasks.

## Install

```bash
uv add "viy @ git+<repo-url>"
```

Requires Python 3.13 or newer.

## Quick start

Training data:

```python
import albumentations as A
from viy.augs import NTIREPresetStep
from viy.data import DatasetSpec, DetectionDataset, Preprocessor

train = DetectionDataset(
    DatasetSpec("ntire", splits="shard_0"),  # downloaded from the Hub on first use
    stage="train",
    preprocessor=Preprocessor(image_size=256),
    augmentation=[A.HorizontalFlip(p=0.5), NTIREPresetStep("aggressive")],
    seed=42,
)
val = DetectionDataset("ntire", stage="val", preprocessor=Preprocessor(image_size=256))

# shuffled with a generator seeded from seed
train_loader = train.dataloader(batch_size=64, num_workers=8)
# {"ntire/val": ..., "ntire/val_hard": ...}
val_loaders = val.dataloaders(batch_size=64)
```

Datasets, augmentations, preprocessing and benchmarks can also be built from YAML configs with `from_pydantic_model`.

Benchmark:

```python
from viy.bench import Benchmark, results_table

# MyDetector implements viy.bench.DetectorModel
benchmark = Benchmark(["ntire/val", "ntire/val_hard"], output_dir="results")
print(results_table(benchmark.run(MyDetector())))
```

Or from the command line with a YAML config: `viy bench run bench.yaml`. Benchmark tasks can also evaluate seeded, augmented copies of any split, see [Benchmark](docs/benchmark.md#augmented-and-subsampled-tasks).

## Documentation

| Page                                       | Contents                                                                                  |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- |
| [Getting started](docs/getting-started.md) | Install, storage, the first training loader and the first benchmark run                   |
| [Datasets](docs/datasets.md)               | Canonical layout, supported datasets, custom and derived splits, sanity checks, commands   |
| [Training data](docs/training.md)          | `DetectionDataset`, DataLoaders, stages, configs, preprocessing, seeds, DDP and Lightning |
| [Augmentations](docs/augmentations.md)     | Pipelines, steps, configs, NTIRE presets, proxies                                         |
| [Operations reference](docs/operations.md) | Every operation with its description and hyperparameters                                  |
| [Benchmark](docs/benchmark.md)             | `DetectorModel`, tasks, augmented tasks, metric pool, slices, results, command line       |
| [Extending](docs/extending.md)             | New datasets, operations, tasks and metrics                                               |
| [Agent notes](docs/agents.md)              | Architecture, invariants and conventions for coding agents                                |

## Supported datasets

| Name                         | Dataset                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| `ntire`                      | NTIRE 2026 Robust AI-Generated Image Detection             |
| `mllm_generated`             | MLLM-Generated Image Detection                             |
| `openfake_v1`, `openfake_v2` | OpenFake v1 and v2                                         |
| `df26`                       | DF26 talking-head videos as frames; gated, benchmark only  |

`viy datasets list` shows every dataset and split, and whether it is prepared.

## Development

```bash
uv sync
uv run pytest
uv run ruff format && uv run ruff check viy && uv run ty check
```

## Limitations

- Learned codecs, adversarial attacks and watermarks are available only as proxies. Each proxy logs a warning when it is used.
- Downloading the NTIRE training shards from the Hub has not been tested on the real archives yet.
- The OpenFake and DF26 converters were checked on real metadata and sample files, not on complete downloads.
- DF26 is scored per frame; the paper reports video-level AUROC.
