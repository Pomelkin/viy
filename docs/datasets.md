# Datasets

## Canonical layout

Every dataset, whatever its original format, is converted into the same structure under the storage root:

```
<root>/<dataset>/
    dataset.json          provenance of every prepared split
    images/...            symlinks to the raw images, copies on non-POSIX systems
    splits/<split>.csv    one table per split
    raw/...               raw files downloaded from the Hub; absent when raw_path is used
    .dirlock              cross-process lock used during preparation
```

A split CSV has one row per image:

| Column | Type | Meaning |
|---|---|---|
| `image` | str, required | path relative to `<root>/<dataset>`, e.g. `images/val/ce78...jpg` |
| `label` | int, required | `0` real, `1` generated |
| `generator` | str | generator of a fake image |
| `distortions` | JSON list | applied distortions in order, `[]` for a clean image |
| `distortion_scales` | JSON list | severity of each distortion |
| `source_id` | str | id shared by images derived from the same source image |
| *anything else* | str | dataset-specific value, available as `SampleRecord.extra` |

An empty cell means **unknown**, never "none". Two examples:
- The NTIRE training shards leave `distortions` empty because nobody knows what was applied to them.
- A clean validation image has `distortions` equal to `[]`.

In Python a row is a frozen `viy.layout.SampleRecord`:

```python
SampleRecord(
    image="images/test_private/281cf05519a0ab70fe78.jpg",
    label=1,
    generator="gpt-image-1.5",
    distortions=("jpeg_recompression_2", "adv_embed_resnet", "downscale"),
    distortion_scales=(0, 10, 0.7216),
    source_id=None,
    extra={},
)
```

## Storage

Prepared datasets live under one directory, the storage root. By default it is `~/.cache/viy`; set the `VIY_ROOT` environment variable to move it. Everything that prepares or reads datasets also accepts a `StorageConfig`:

```python
from pathlib import Path
from viy.datasets import StorageConfig
from viy.layout import FileCheck, LinkMode

StorageConfig(
    root=Path("/data/viy"),      # default: $VIY_ROOT or ~/.cache/viy
    auto_prepare=True,           # prepare missing splits on first access
    link_mode=None,              # None: symlink on POSIX, copy elsewhere; or LinkMode.HARDLINK / COPY
    file_check=FileCheck.ALL,    # check every image on load; SAMPLE checks 1000, NONE skips
)
```

Each dataset is downloaded from the Hugging Face Hub unless a local copy of its raw files is given as `raw_path`, e.g. `DatasetSpec("ntire", raw_path="/datasets/NTIRE")` or `NTIRE(raw_path="/datasets/NTIRE")`.

## Supported datasets

### `ntire`: NTIRE 2026 Robust AI-Generated Image Detection

| Split | Role | Source | Size | Labels | Distortions | Generators |
|---|---|---|---:|---|---|---|
| `shard_0` … `shard_5` | train | Hub, about 20 GB zip each | 277,643 | yes | unknown | no |
| `shard_6` | train | local only | 10,000 | yes | genuine `jpeg_ai` or `wmforger` | no |
| `val` | val | Hub | 10,000 | yes | yes, 19 types | no |
| `val_hard` | val | Hub | 2,500 | yes | yes, 21 types | no |
| `test_public` | test | Hub | 2,500 | yes | yes, 24 types | no |
| `test_private` | test | local only | 2,500 | yes | yes | yes, 10 generators |

> **Stored locally.** `shard_6` and `test_private` are not on the Hugging Face Hub; they exist only as local copies. Prepare them by pointing `raw_path` at a directory that contains them, as in the layout below. Without it, loading them raises an error.

Further details:
- **Split group:** `train` = `shard_0` … `shard_5`.
- **Defaults:** training uses `train`; validation uses `val` and `val_hard`.
- **`shard_6`** is an optional extension that is not part of the official release. It holds JPEG AI and WMForger versions of shard 0–5 images, made with the official JPEG AI reference software and the WMForger attack, and its `source_id` is the name of the original image. Stored locally only.
- **`test_private`** is the private NTIRE test set with per-image generators. Stored locally only.
- **Labelled half:** half of every validation and test split is distorted, and the chains are recorded per image. How these chains were built is described in [Augmentations](augmentations.md#ntire-presets).

Expected raw layout for `raw_path`. The published part is exactly what the three Hub repositories give when unpacked side by side under `train/`, `val/` and `test/`:

```
train/shard_<i>/{images/, labels.csv}          labels.csv: image_name, label
val/{val_images/, val_labels.csv, val_images_hard/, val_hard_labels.csv}
test/{test_images/, test_labels.csv}
train/shard_6/{images/, labels.csv}            optional; image_name, label, transform, jpeg_ai_level,
                                               wmforger_learning_rate, source_image_name
test/new_test_images/ds/NTIRE_test_private/v1/{test_set/, clear_metadata.csv, distorted_metadata.csv}   optional
```

### `mllm_generated`: MLLM-Generated Image Detection

The dataset comes from arXiv 2608.01258 and the Hub repository `zr-zhang/MLLM-Generated-Image-Detection-Dataset`. It has about 45k images and takes about 50 GB.

| Split | Role | Content |
|---|---|---|
| `train` | train | everything; the Hub publishes no other split |

What each record carries:
- **`generator`:** `GPT Image 2`, `Nano Banana 2`, `FLUX.2` or `Qwen Image 2.1`; empty for real images.
- **`extra["domain"]`:** `hybrid`, `structure` or `texture`.

The dataset has no validation split, so:
- in training it is used for training only, with a warning;
- in the benchmark its training split is evaluated, with a warning.

**Format shortcut:** real images are small JPEGs and fakes are large PNGs. The sanity check reports this; re-encode both classes identically before trusting metrics.

## Working with a dataset directly

```python
from viy.datasets import NTIRE

ntire = NTIRE()                          # or NTIRE(raw_path="/datasets/NTIRE"), NTIRE(storage=...)
ntire.expand_splits(["train"])           # ['shard_0', ..., 'shard_5']
records = ntire.load("val")              # prepares the split if needed and runs sanity checks
path = ntire.layout.image_path(records[0])
ntire.layout.splits()                    # splits present on disk
ntire.layout.read_manifest().splits      # provenance: sizes, source, preparation time
```

`ntire.prepare("val")` only builds the split. Use it to prepare data in advance, for example before a multi-GPU job.

## Custom splits

A split exists if and only if its CSV exists, so a custom split is just a CSV over images that are already linked. For example, to hold out part of `shard_5` for validation:

```python
from viy.data import subsample
from viy.datasets import NTIRE
from viy.layout import write_records

ntire = NTIRE()
records = ntire.load("shard_5")
held_out = subsample(records, 2000, seed=0)            # class-stratified
held_out_images = {record.image for record in held_out}

write_records(ntire.layout.split_path("shard_5_val"), held_out)
write_records(
    ntire.layout.split_path("shard_5_train"),
    [record for record in records if record.image not in held_out_images],
)
```

```python
from viy.data import DatasetSpec, DetectionDataset, Preprocessor, check_disjoint

shards = ["shard_0", "shard_1", "shard_2", "shard_3", "shard_4", "shard_5_train"]
train = DetectionDataset([DatasetSpec("ntire", splits=shards)], stage="train", preprocessor=Preprocessor(256))
val = DetectionDataset([DatasetSpec("ntire", splits=["shard_5_val", "val"])], stage="val", preprocessor=Preprocessor(256))
check_disjoint(train, val)
```

A hand-written CSV only needs the `image` and `label` columns. Do not combine the `train` group with `shard_5_val`: the group contains `shard_5`, and `check_disjoint` will report the overlap.

## Sanity checks

These run every time a split is loaded:

| Check | Result |
|---|---|
| empty split | error |
| an image listed twice | error |
| missing image or broken symlink | error; with `file_check=sample`, only 1000 random images are checked |
| a single class in the split | warning |
| file formats differ between classes | warning about the "JPEG real vs PNG fake" shortcut |

`check_disjoint(train, val)` raises when an image is used for both training and validation. It compares resolved file paths, so it also catches the same raw file reached through two splits or two datasets.

## Preparation and concurrency

Preparation of a split goes through these steps:
1. download the needed Hub files, unless `raw_path` is set;
2. unpack the archives and delete the zip files;
3. link the images;
4. write the CSV;
5. update `dataset.json`.

All of this runs under a lock on `<root>/<dataset>`. When several processes, for example DDP ranks, ask for the same missing split, one prepares it and the others wait. Then everyone reads the result. A split whose CSV exists is never rebuilt. To rebuild it, delete `splits/<split>.csv`.
