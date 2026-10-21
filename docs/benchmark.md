# Benchmark

The benchmark has three parts:
- a model implements a small interface;
- tasks are selected by name;
- a runner evaluates the model on every task and stores one result per task.

## The model interface

```python
from typing import override

import torch
from PIL import Image
from viy.bench import DetectorModel, ModelMeta
from viy.data import Preprocessor, ResizeMode


class MyDetector(DetectorModel):
    def __init__(self, checkpoint: str) -> None:
        self.net = load_my_network(checkpoint).eval().cuda()
        self.prep = Preprocessor(image_size=256, resize_mode=ResizeMode.RESIZE_CENTER_CROP)

    @property
    @override
    def meta(self) -> ModelMeta:
        return ModelMeta(
            name="dinov3-distill-s",
            revision="epoch_05",
            n_parameters=18_700_000,
            reference="https://github.com/...",
        )

    @property
    @override
    def threshold(self) -> float:          # optional, 0.5 by default
        return 0.5

    @override
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.prep(image)

    @override
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch.cuda(non_blocking=True)).sigmoid().squeeze(1)
```

| Member | Contract |
|---|---|
| `meta` | `ModelMeta` with at least `name`; results are stored under `name` and `revision` |
| `preprocess(image)` | receives a decoded RGB `PIL.Image` and returns a tensor; tensors of one batch must have equal shapes. It runs in DataLoader worker processes, so it must not use the GPU |
| `predict(batch)` | receives the stacked tensors and returns the probability that each image is generated, shape `(B,)`. It runs under `torch.inference_mode()`, and the model moves the batch to its own device |
| `threshold` | decision threshold of the threshold-dependent metrics |

Scores outside `[0, 1]` are accepted for the threshold-free metrics, with a warning. Non-finite scores are an error.

## Tasks

| Task | Split | Notes |
|---|---|---|
| `ntire/val` | NTIRE validation | 10,000 images, half distorted |
| `ntire/val_hard` | NTIRE hard validation | adds neural codecs and adversarial attacks |
| `ntire/test_public` | NTIRE public test | adds watermarks, Cheng2020, WMForger |
| `ntire/test_private` | NTIRE private test | unpublished, needs a local copy; per-image generators |
| `mllm_generated/train` | MLLM-Generated, whole dataset | no evaluation split exists; skip it for models trained on this dataset |

Every validation and test split of a registered dataset becomes a task automatically. A dataset without such splits gets tasks on its training splits, marked `evaluates_train_split`, and they log a warning when run.

```python
from viy.bench import get_tasks

get_tasks()                                  # every task
get_tasks(["ntire"])                         # every task of a dataset
get_tasks(["ntire/val", "ntire/test_public"])
```

The NTIRE evaluation splits are already distorted by the organizers, and the benchmark scores them as they are. It does not augment benchmark images. Robustness is reported through slices over the recorded distortions.

## Running

```python
from pathlib import Path
from viy.bench import Benchmark, get_tasks, results_table
from viy.datasets import DatasetSourceConfig

benchmark = Benchmark(
    tasks=get_tasks(["ntire", "mllm_generated"]),
    output_dir=Path("results"),
    sources=[DatasetSourceConfig(name="ntire", raw_path=Path("/datasets/NTIRE"))],   # optional local copy
    batch_size=64,
    num_workers=8,
    overwrite=False,
)
results = benchmark.run(MyDetector("epoch_05.pt"))
```

Datasets not listed in `sources` are downloaded from the Hub on first use. Prepared data goes to `~/.cache/viy` or `$VIY_ROOT`; pass `storage=StorageConfig(root=...)` to put it elsewhere. The same run from a config:

```yaml
# bench.yaml
output_dir: results
tasks: [ntire, mllm_generated]
sources:
  - {name: ntire, raw_path: /datasets/NTIRE}
batch_size: 64
num_workers: 8
```

```python
from kostyl.utils import load_file
from viy.bench import Benchmark, BenchmarkConfig

benchmark = Benchmark.from_pydantic_model(BenchmarkConfig.model_validate(load_file("bench.yaml")))
```

A task that already has a result for the same model name and revision is loaded instead of recomputed, so an interrupted run resumes. Set `overwrite=True` to recompute.

## Results

```
results/
    dinov3-distill-s__epoch_05/
        ntire__val.json
        ntire__val.predictions.csv
        ...
```

The `.predictions.csv` file holds `image,label,score` for every image, for later analysis. The JSON is a `TaskResult`:

```json
{
  "task": "ntire/test_private",
  "dataset": "ntire",
  "split": "test_private",
  "model": {"name": "brightness", "revision": "v1", "n_parameters": 0, "reference": null},
  "viy_version": "0.1.0",
  "evaluated_at": "2026-10-22T03:41:12.513Z",
  "threshold": 0.5,
  "main_score_name": "auc",
  "main_score": 1.0,
  "slices": {
    "all": {"...": "..."},
    "clean": {"...": "..."},
    "distortion:jpeg_ai@2": {"...": "..."},
    "generator:hidream": {
      "num_samples": 5, "num_real": 3, "num_fake": 2,
      "auc": 1.0, "average_precision": 1.0,
      "accuracy": 1.0, "balanced_accuracy": 1.0, "f1": 1.0,
      "fpr": 0.0, "fnr": 0.0,
      "tpr_at_1pct_fpr": 1.0, "tpr_at_01pct_fpr": 1.0,
      "mean_score_real": 0.2335, "mean_score_fake": 0.7432
    }
  },
  "aggregates": {
    "generator_macro_auc": 1.0, "generator_worst_auc": 1.0, "generator_worst": "gpt-image-1.5",
    "distortion_macro_auc": 1.0, "distortion_worst_auc": 1.0, "distortion_worst": "downscale"
  },
  "evaluation_seconds": 0.41
}
```

The example comes from a toy model on test fixtures.

### Slices

| Slice | Images | Exists when |
|---|---|---|
| `all` | the whole split | always |
| `clean`, `distorted` | images without / with distortions | every image has a known distortion chain |
| `distortion:<op>` | images the operation was applied to, both classes | as above |
| `distortion:<op>@<level>` | the same at one severity | the operation has at most 10 distinct levels |
| `generator:<name>` | fakes of one generator **plus all real images** | generators are known |

### Metrics

Every slice has:

| Field | Meaning |
|---|---|
| `auc` | ROC AUC; `None` for single-class slices |
| `average_precision` | average precision, fakes as positives |
| `tpr_at_1pct_fpr`, `tpr_at_01pct_fpr` | detection rate at 1% and 0.1% false positives |
| `fpr` | share of real images flagged as generated at `threshold` |
| `fnr` | share of generated images missed at `threshold` |
| `accuracy`, `balanced_accuracy`, `f1` | at `threshold` |
| `mean_score_real`, `mean_score_fake` | mean score per class, to see score shift under distortions |
| `num_samples`, `num_real`, `num_fake` | slice size |

FPR and FNR are reported separately because distortions break detectors in two different ways: fakes become invisible, or reals get flagged.

The aggregates are:
- `<prefix>_macro_auc`: the mean AUC over the slices of that kind;
- `<prefix>_worst_auc` and `<prefix>_worst`: the lowest AUC and the slice it comes from.

The prefix is `generator` or `distortion`, and level slices are excluded.

### Comparing models

```python
from pathlib import Path
from viy.bench import load_results, results_table

results = load_results(Path("results"))
print(results_table(results))                      # main score, AUC by default
print(results_table(results, metric="tpr_at_1pct_fpr"))
```

```
| model | ntire/test_private | ntire/val |
|---|---:|---:|
| dinov3-distill-s@epoch_05 | 0.9312 | 0.9861 |
| safe | 0.8120 | 0.9433 |
```

For anything beyond the main table, read the `slices` of `TaskResult` objects or the prediction CSV files directly.

## Custom tasks

```python
from viy.bench import DetectionTask, TaskMetadata, register_task

register_task(DetectionTask(TaskMetadata(
    name="ntire/shard_5_val",
    dataset="ntire",
    split="shard_5_val",            # a custom split, see datasets.md
    description="Held-out part of shard 5",
    main_score="tpr_at_1pct_fpr",
)))
```

The split can be any split present in the canonical layout, including custom ones. To evaluate differently, subclass `DetectionTask` and override `evaluate`.
