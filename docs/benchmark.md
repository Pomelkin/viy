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
        self.prep = Preprocessor(
            image_size=256, resize_mode=ResizeMode.RESIZE_CENTER_CROP
        )

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
    def threshold(self) -> float:  # optional, 0.5 by default
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

`viy.data.Preprocessor` fits `preprocess` well. Configure it with custom settings or with the transformers image processor of your backbone: `Preprocessor(image_processor="facebook/dinov2-small")`. A `DetectionDataset` passes the same RGB `PIL.Image` to its `preprocessor`, so one function serves training and the benchmark.

Scores outside `[0, 1]` are accepted for the threshold-free metrics, with a warning. Non-finite scores are an error.

## Tasks

| Task | Split | Notes |
|---|---|---|
| `ntire/val` | NTIRE validation | 10,000 images, half distorted |
| `ntire/val_hard` | NTIRE hard validation | adds neural codecs and adversarial attacks |
| `ntire/test_public` | NTIRE public test | adds watermarks, Cheng2020, WMForger |
| `ntire/test_private` | NTIRE private test | stored locally only, needs a local copy; per-image generators; selected by its full name only |
| `mllm_generated/train` | MLLM-Generated, whole dataset | no evaluation split exists; skip it for models trained on this dataset |
| `openfake_v1/test` | OpenFake v1 test | 59,658 images of the training generators |
| `openfake_v2/validation` | OpenFake v2 validation | held-out images of the training generators |
| `openfake_v2/test` | OpenFake v2 test | unseen generators and unseen real sources |
| `openfake_v2/reddit` | OpenFake v2 Reddit set | in the wild; generators unknown |
| `df26/test` | DF26 videos, 4 frames each | gated, evaluation only; per-generator slices |

Every validation and test split of a registered dataset becomes a task automatically. A dataset without such splits gets tasks on its training splits. Any task that scores a training split, built-in or custom, reports `evaluates_train_split` and logs a warning when run.

```python
from viy.bench import get_tasks

get_tasks("ntire")  # the tasks of a dataset whose data can be downloaded
get_tasks(["ntire/val", "ntire/test_private"])  # by task name
get_tasks()  # every task whose data can be downloaded
```

Selecting a dataset by name leaves out splits that exist only as local copies, such as `ntire/test_private`; name them in full to run them. `Benchmark` takes the same names directly.

Built-in tasks score their splits as they are. The NTIRE evaluation splits are already distorted by the organizers, so robustness on them is reported through slices over the recorded distortions. To measure robustness on any other dataset, define an [augmented task](#augmented-and-subsampled-tasks).

## Running

```python
from viy.bench import Benchmark

benchmark = Benchmark(
    ["ntire", "ntire/test_private", "mllm_generated"],  # task or dataset names
    output_dir="results",
    raw_paths={"ntire": "/datasets/NTIRE"},  # optional local copies
    batch_size=64,
    num_workers=8,
)
results = benchmark.run(MyDetector("epoch_05.pt"))
```

`tasks` also takes `TaskMetadata` and `DetectionTask` objects. Datasets not listed in `raw_paths` are downloaded from the Hub on first use. Prepared data goes to `~/.cache/viy` or `$VIY_ROOT`; pass `storage=StorageConfig(root=...)` to put it elsewhere. The same run from a config:

```yaml
# bench.yaml
output_dir: results
tasks: [ntire, ntire/test_private, mllm_generated]
raw_paths: {ntire: /datasets/NTIRE}
batch_size: 64
num_workers: 8
```

```python
from viy import load_config
from viy.bench import Benchmark, BenchmarkConfig

benchmark = Benchmark.from_pydantic_model(
    BenchmarkConfig.model_validate(load_config("bench.yaml"))
)
```

`tasks` is required: list the tasks or datasets to run, since every task together means downloads of hundreds of gigabytes, gated datasets and local-only splits.

Before any inference, the run checks that every task can get its data: the split is prepared, or the Hub publishes it and you have access, or `raw_paths` points at a local copy. A missing local copy or gated access therefore fails the run at once, not after hours of evaluation.

A task that already has a result for the same model name and revision is loaded instead of recomputed, so an interrupted run resumes. Set `overwrite=True` to recompute. Give every checkpoint its own `ModelMeta.revision`: without one, the run warns, because a retrained model would reuse the results of the old one.

### Command line

`viy bench run` reads the same config with the detector to build and, optionally, modules to import first:

```yaml
# bench.yaml
model:
  target: my_detectors:MyDetector    # module:attribute of a DetectorModel subclass or factory
  kwargs: {checkpoint: epoch_05.pt}
imports: [my_datasets]                # optional: modules that register datasets, tasks or metrics
output_dir: results
tasks: [ntire, mllm_generated]
metrics: [auc, eer, tpr_at_1pct_fpr, fpr, fnr]   # optional, see Metrics
batch_size: 64
num_workers: 8
```

```bash
viy bench run bench.yaml              # --overwrite recomputes stored results
viy bench tasks ntire                 # list tasks, all of them without arguments
viy bench metrics                     # the metric pool
viy bench results results --metric tpr_at_1pct_fpr
viy bench results results --metric auc --slice augmentation:jpeg
viy bench results results --aggregate generator_worst_auc
viy --import my_datasets datasets check my_dataset   # --import works with every command
```

The directory of the config and the current directory are importable, so `my_detectors.py` can sit next to the config. Relative paths in the config resolve against the current directory. Every command prints a table, and inference shows a progress bar per task.

## Augmented and subsampled tasks

A task can evaluate a split **derived** from a prepared one: a class-stratified subset, an augmented copy, or both. The derivation runs once with the seed you give, and its images are stored in the canonical layout, so every model is scored on the same pixels.

```yaml
# bench.yaml
output_dir: results
tasks: [ntire]                       # built-in tasks; [] runs only custom_tasks
custom_tasks:
  - name: mllm_generated/test_public_mix   # optional
    dataset: mllm_generated
    split: train
    derivation:
      seed: 0                        # required
      limit: 5000                    # optional, class-stratified
      augmentation:                  # optional, an AugmentationConfig
        steps:
          - {kind: ntire_preset, preset: test_public}
```

The same task in Python:

```python
import albumentations as A
from viy.augs import AugmentationPipeline, NTIREPresetStep
from viy.bench import Benchmark, SplitDerivation, TaskMetadata

mix = TaskMetadata(
    name="mllm_generated/test_public_mix",  # optional; defaults to <dataset>/<derived split>
    dataset="mllm_generated",
    split="train",
    derivation=SplitDerivation(
        seed=0,
        limit=5000,
        augmentation=AugmentationPipeline(
            [NTIREPresetStep("test_public"), A.HorizontalFlip(p=0.5)]
        ).to_config(),
    ),
)
benchmark = Benchmark(["ntire", mix], output_dir="results")
```

The derivation stores the augmentation as a config, because it is hashed and saved with the split; `to_config()` turns a pipeline of Python objects into one. In YAML, `serialized` albumentations steps describe composes such as `OneOf`, see [Augmentations](augmentations.md#albumentations-any-albumentations-transform).

On the first run of the task:
1. the base split is prepared and `limit` images are drawn with `seed`;
2. every image is augmented with a generator seeded from `seed` and the image path, and saved as PNG;
3. the split is written to `splits/<split>-<fingerprint>.csv`, next to a JSON file with the derivation, the seed and the library versions.

```
~/.cache/viy/mllm_generated/
    images/derived/train-3f9a1c2b7d10/...png
    splits/train-3f9a1c2b7d10.csv
    splits/train-3f9a1c2b7d10.json
```

The fingerprint covers the base split and every setting of the derivation. Changing any of them builds a new split and keeps the old one. Later runs and other models reuse the stored split.

The seed is all it takes to reproduce the split. Without the cache, on another machine, the split is rebuilt with the same images as long as viy, NumPy, OpenCV, albumentations and Pillow have the same versions. When a stored split was built with other versions, a warning says so.

A derived image keeps the label, generator and source id of its base image. Its `augmentations` column lists the applied operations with their levels, and `base_image` names the original image. `distortions` are unknown: what happened to the image before the augmentation is a separate history. Robustness to the augmentation is reported through the [augmentation slices](#slices).

The result stores `derived_split` and the whole `derivation`, seed included. A stored result is reused only if it was computed on the same derived split. If the derivation of a task changes but its name does not, the run stops with an error; rename the task or set `overwrite=True`.

A validation `DetectionDataset` with the same seed, `limit` and augmentation sees exactly the images of the derived split, so validation during training matches the benchmark.

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
  "derived_split": null,
  "derivation": null,
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
      "metrics": {
        "auc": 1.0, "average_precision": 1.0, "eer": 0.0,
        "tpr_at_1pct_fpr": 1.0, "tpr_at_01pct_fpr": 1.0,
        "accuracy": 1.0, "balanced_accuracy": 1.0, "f1": 1.0, "fpr": 0.0, "fnr": 0.0,
        "mean_score_real": 0.2335, "mean_score_fake": 0.7432
      }
    }
  },
  "aggregates": {
    "generator_macro_auc": 1.0, "generator_worst_auc": 1.0, "generator_worst_auc_slice": "gpt-image-1.5",
    "generator_macro_eer": 0.0, "generator_worst_eer": 0.0, "generator_worst_eer_slice": "gpt-image-1.5",
    "distortion_macro_fpr": 0.0, "distortion_worst_fpr": 0.0, "distortion_worst_fpr_slice": "downscale",
    "...": "..."
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
| `not_augmented`, `augmented` | images a derived split left as they were / augmented | the split is [derived](#augmented-and-subsampled-tasks) with an augmentation |
| `augmentation:<op>` | images the operation was applied to by the derivation, both classes | as above |
| `augmentation:<op>@<level>` | the same at one level | the operation has at most 10 distinct levels |
| `generator:<name>` | fakes of one generator **plus all real images** | generators are known |

### Metrics

Every slice stores its size, `num_samples`, `num_real` and `num_fake`, and the value of every selected metric under `metrics`. Generated images are the positive class. A metric is `null` where it is undefined, for example AUC on a slice with one class. The pool:

| Metric | Better | Uses `threshold` | Default | Meaning |
|---|---|:---:|:---:|---|
| `auc` | higher | | ✓ | ROC AUC |
| `average_precision` | higher | | ✓ | area under the precision-recall curve |
| `eer` | lower | | ✓ | equal error rate, the point where FPR equals FNR |
| `tpr_at_5pct_fpr` | higher | | | share of fakes detected at 5% false positives |
| `tpr_at_1pct_fpr` | higher | | ✓ | the same at 1% |
| `tpr_at_01pct_fpr` | higher | | ✓ | the same at 0.1% |
| `fpr_at_95pct_tpr` | lower | | | share of reals flagged when 95% of fakes are detected |
| `accuracy` | higher | ✓ | ✓ | share of correct decisions |
| `balanced_accuracy` | higher | ✓ | ✓ | mean of the detection rate and the true negative rate |
| `precision` | higher | ✓ | | share of flagged images that are generated |
| `recall` | higher | ✓ | | share of generated images flagged, the TPR |
| `f1` | higher | ✓ | ✓ | F1 score of the fake class |
| `mcc` | higher | ✓ | | Matthews correlation coefficient |
| `fpr` | lower | ✓ | ✓ | share of real images flagged as generated |
| `fnr` | lower | ✓ | ✓ | share of generated images missed |
| `log_loss` | lower | | | binary cross-entropy |
| `brier` | lower | | | mean squared error of the scores |
| `ece` | lower | | | expected calibration error over 15 equal-width score bins |
| `mean_score_real` | — | | ✓ | mean score of real images |
| `mean_score_fake` | — | | ✓ | mean score of generated images |

`log_loss`, `brier` and `ece` treat scores as probabilities and are `null` when a score leaves `[0, 1]`. `viy bench metrics` prints the same table.

Choose metrics with `metrics` in the config, or the argument of `Benchmark`:

```yaml
metrics: [auc, eer, tpr_at_1pct_fpr, fpr, fnr, ece]
```

```python
benchmark = Benchmark("df26", output_dir="results", metrics=["auc", "eer"])
```

The main score of each task is computed even if it is not selected. When a stored result lacks a selected metric, the run recomputes the metrics from the stored `.predictions.csv` and does not call the model again. Adding a metric to a finished benchmark is therefore cheap.

FPR and FNR are reported separately because distortions break detectors in two different ways: fakes become invisible, or reals get flagged.

The aggregates summarize every selected metric with a direction over the slices of one kind:
- `<prefix>_macro_<metric>`: the mean over the slices;
- `<prefix>_worst_<metric>` and `<prefix>_worst_<metric>_slice`: the worst value and the slice it comes from, the lowest for metrics where higher is better and the highest otherwise.

The prefix is `generator`, `distortion` or `augmentation`. Level slices and the mean scores are excluded.

To add a metric of your own, register it before building the benchmark; it then works like any metric of the pool:

```python
from viy.bench import Metric, ScoredSet, register_metric


def score_gap(scored: ScoredSet) -> float | None:
    if not scored.both_classes:
        return None
    return float(scored.scores[scored.fake].mean() - scored.scores[scored.real].mean())


# replace=True redefines a metric, e.g. when a notebook cell runs again
register_metric(Metric("score_gap", "mean fake score minus mean real score", score_gap))
```

`ScoredSet` exposes `labels`, `scores`, `threshold`, the masks `real`, `fake` and `flagged`, and the cached ROC curve `roc`.

### Comparing models

```python
from pathlib import Path
from viy.bench import load_results, results_table

results = load_results(Path("results"))
print(results_table(results))  # main score, AUC by default
print(results_table(results, metric="tpr_at_1pct_fpr"))  # any computed metric
```

```
| model | ntire/test_private | ntire/val |
|---|---:|---:|
| dinov3-distill-s@epoch_05 | 0.9312 | 0.9861 |
| safe | 0.8120 | 0.9433 |
```

`results_table(results, metric="auc", slice_name="augmentation:jpeg")` tabulates one slice, and `results_table(results, aggregate="generator_worst_auc")` one aggregate. A name that no result has raises an error listing the available ones. For anything beyond these tables, read `result.slices[name].metrics` of `TaskResult` objects or the prediction CSV files directly.

## Custom tasks

```python
from viy.bench import TaskMetadata, register_task

register_task(
    TaskMetadata(
        dataset="ntire",
        split="shard_5_val",  # a custom split, see datasets.md; the task is named ntire/shard_5_val
        description="Held-out part of shard 5",
        main_score="tpr_at_1pct_fpr",
    )
)
```

A registered task can be selected by name, in Python and in configs loaded after the registration. To use one task once, pass the `TaskMetadata` to `Benchmark` instead. `replace=True` redefines a task of the same name, e.g. when a notebook cell runs again. The split can be any split present in the canonical layout, including custom ones. Add a `derivation` to evaluate a seeded subset or an augmented copy of it, see [above](#augmented-and-subsampled-tasks). To evaluate differently, subclass `DetectionTask` and override `evaluate`.
