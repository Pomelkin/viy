# Datasets

## Canonical layout

Every dataset, whatever its original format, is converted into the same structure under the storage root:

```
<root>/<dataset>/
    dataset.json          provenance of every prepared split
    images/...            symlinks to the raw images, copies on non-POSIX systems,
                          or image files converted from Parquet shards and videos
    images/derived/...    augmented images of derived splits, PNG
    splits/<split>.csv    one table per split
    splits/<split>.json   provenance of a derived split: base split, seed, augmentation
    raw/...               raw files downloaded from the Hub; absent when raw_path is used
    converted/<split>/... per-shard records of datasets converted from Parquet shards
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
| `augmentations` | JSON list | derived splits only: operations the derivation applied, as `{"name", "scale"}` objects |
| `base_image` | str | derived splits only: the image the row was made from |
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
    root=Path("/data/viy"),  # default: $VIY_ROOT or ~/.cache/viy
    auto_prepare=True,  # prepare missing splits on first access
    link_mode=None,  # None: symlink on POSIX, copy elsewhere; or LinkMode.HARDLINK / COPY
    file_check=FileCheck.ALL,  # check every image on load; SAMPLE checks 1000, NONE skips
)
```

Each dataset is downloaded from the Hugging Face Hub unless a local copy of its raw files is given: `DatasetSpec("ntire", raw_path="/datasets/NTIRE")` in training, `Benchmark(..., raw_paths={"ntire": "/datasets/NTIRE"})` in the benchmark, `NTIRE(raw_path="/datasets/NTIRE")` directly, or `--raw-path` on the command line.

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

### `openfake_v1` and `openfake_v2`: OpenFake

The dataset comes from arXiv 2509.09495 and the Hub repository `ComplexDataLab/OpenFake`. It contains politically salient real photos, such as faces of public figures, news events and memes, against images of proprietary, open and community fine-tuned generators. The two versions are separate datasets, each pinned to a fixed revision of the repository: v1 to the `v1.0` tag, and v2 to its repaired release on `main`.

| Dataset | Split | Role | Size | Real images |
|---|---|---|---:|---|
| `openfake_v1` | `train` | train | 1,870,684, about 1 TB | LAION |
| `openfake_v1` | `test` | test | 59,658 | LAION |
| `openfake_v2` | `train` | train | 2,300,034, about 3.3 TB | LAION, Pexels |
| `openfake_v2` | `validation` | val | 58,453 | LAION, Pexels |
| `openfake_v2` | `test` | test | 91,397, unseen generators | DOCCI, ImageNet |
| `openfake_v2` | `reddit` | test | 36,087, in the wild | photography subreddits |

What each record carries:
- **`generator`:** the generator of a fake image, such as `gpt-image-1.5` or `z-image-turbo`. It is empty for real images and for `reddit`, where the generator is unknown.
- **`extra`:** `type` (`base`, `finetune`, `lora`, `video`, ...) and `release_date`. Real images also carry `real_source`, and v2 adds `row_id` and `image_sha256`.
- **`distortions`:** unknown.

Further details:
- **Defaults:** training uses `train`; v2 validates on `validation`.
- **Generator overlap:** several v2 test generators, such as `recraft-v3` and `midjourney-7`, are in the v1 training split. Do not evaluate v1-trained models on the v2 test.
- **Conversion:** the Hub stores images inside Parquet shards. Each shard is converted on its own: the original image bytes are written without re-encoding, under the suffix of their actual format, which often differs from the file name stored in the shard. A downloaded shard is deleted right after conversion, so preparing `openfake_v2/train` needs about 3.3 TB, not twice that. An interrupted preparation resumes from the first shard that has no records under `converted/`.

Expected raw layout for `raw_path`, the same as the repository: `data/{train,test}-*.parquet` for v1, and `core/{train,validation,test}-*.parquet` with `reddit/test-*.parquet` for v2. Local shards are read in place and kept.

### `df26`: DF26 talking-head videos

The dataset comes from arXiv 2609.07369 and the Hub repository `DF26/DF26`. It has 271 real talking-head videos and 2,420 generated ones, 1280×720 H.264 clips of about 5 s. The fakes come from 10 video generators: 4 commercial APIs and 6 open-weight models, the latter in image-to-video and text-to-video modes.

> **Gated and evaluation only.** Request access on the [dataset page](https://huggingface.co/datasets/DF26/DF26), then log in with `hf auth login` or set `HF_TOKEN`. viy checks access before downloading and raises if no token is configured or access was not granted; `viy datasets check df26` runs the same check. The license allows evaluation only, so `DetectionDataset` refuses `df26` and it is used through benchmark tasks alone.

| Split | Role | Videos | Frames |
|---|---|---:|---:|
| `test` | test | 2,691 | 10,764 |

Every video becomes four frames, decoded once and saved as PNG. The frames are evenly spaced inside the video, skipping the first and the last frame: image-to-video generators start from the real frame, so the first frames of a fake are the least representative. Real and fake frames go through the same decoding, so their formats do not differ. Extracting the frames takes about ten minutes after the download, and they take about 6 GB on top of 3.6 GB of videos.

What each record carries:
- **`generator`:** for example `Veo_3.1`, `Kling_3.0` or `Wan_2.2_14b_i2v`; empty for real frames.
- **`source_id`:** the clip id shared by a real video and its fakes, which show the same speaker and scene.
- **`extra`:** `scenario`, `video` and `frame`; fakes also carry `access` (`open` or `closed`) and `conditioning`.

The benchmark reports one `generator:<name>` slice per generator, as the dataset authors recommend. Metrics are computed per frame, while the paper reports video-level AUROC, so the numbers are not directly comparable.

Expected raw layout for `raw_path`, the same as the repository: `metadata/videos.csv`, `real/<scenario>/<clip>.mp4` and `fake/<scenario>/<generator>/<clip>.mp4`.

## Working with a dataset directly

```python
from viy.datasets import NTIRE

ntire = NTIRE()  # or NTIRE(raw_path="/datasets/NTIRE"), NTIRE(storage=...)
ntire.expand_splits(["train"])  # ['shard_0', ..., 'shard_5']
records = ntire.load("val")  # prepares the split if needed and runs sanity checks
path = ntire.layout.image_path(records[0])
ntire.layout.splits()  # splits present on disk
ntire.layout.read_manifest().splits  # provenance: sizes, source, preparation time
```

`ntire.prepare("val")` only builds the split. Use it to prepare data in advance, for example before a multi-GPU job. The same from the command line:

```bash
viy datasets prepare ntire --split val --raw-path /datasets/NTIRE
viy datasets list
viy datasets check df26
```

`prepare` builds the train and val splits when `--split` is left out, or the test splits of a benchmark-only dataset. `list` shows every dataset and split and whether it is prepared. Every command takes `--root` to use another storage root. `viy datasets check <name>` makes sure every split is prepared or can be prepared, without downloading anything: it checks local copies of splits that are not on the Hub and your access to gated datasets. In Python the same check is `builder.check_available(split)`.

## Custom splits

A split exists if and only if its CSV exists, so a custom split is just a CSV over images that are already linked. For example, to hold out part of `shard_5` for validation:

```python
from viy.data import subsample
from viy.datasets import NTIRE
from viy.layout import write_records

ntire = NTIRE()
records = ntire.load("shard_5")
held_out = subsample(records, 2000, seed=0)  # class-stratified
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
preprocessor = Preprocessor(image_size=256)
train = DetectionDataset(
    DatasetSpec("ntire", splits=shards), stage="train", preprocessor=preprocessor
)
val = DetectionDataset(
    DatasetSpec("ntire", splits=["shard_5_val", "val"]),
    stage="val",
    preprocessor=preprocessor,
)
check_disjoint(train, val)
```

A hand-written CSV only needs the `image` and `label` columns. An invalid value raises an error with the file, the line and the accepted values. Do not combine the `train` group with `shard_5_val`: the group contains `shard_5`, and `check_disjoint` will report the overlap.

## Derived splits

`derive_split` builds a split from a prepared one with a seed: a class-stratified subset, an augmented copy, or both. The benchmark uses it for [augmented tasks](benchmark.md#augmented-and-subsampled-tasks), and it works on its own too:

```python
from viy.augs import AugmentationConfig, OpStepConfig
from viy.data import SplitDerivation, derive_split, read_derived_info
from viy.datasets import MLLMGenerated

mllm = MLLMGenerated()
derivation = SplitDerivation(
    seed=0, limit=2000, augmentation=AugmentationConfig(steps=[OpStepConfig(op="jpeg")])
)
# "train-<fingerprint>"
name = derive_split(mllm, "train", derivation, num_workers=8)
records = mllm.load(name)
# base split, derivation, library versions, build time
info = read_derived_info(mllm.layout, name)
```

Every image is augmented with a generator seeded from `seed` and the image path, so the same derivation gives the same images on any machine with the same library versions. The name ends with a fingerprint of the base split and the derivation. A changed setting therefore makes a new split, and an existing one is reused. A derivation with only a `limit` writes just a CSV over the existing images. To rebuild a derived split, delete its CSV.

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
1. check that the split can be prepared: a local copy for splits that are not on the Hub, a token and access for gated datasets;
2. download the needed Hub files, unless `raw_path` is set;
3. unpack the archives and delete the zip files;
4. link the images, or convert Parquet shards and videos into image files;
5. write the CSV;
6. update `dataset.json`.

Downloads, unpacking, conversion and linking show progress bars. All of this runs under a lock on `<root>/<dataset>`. When several processes, for example DDP ranks, ask for the same missing split, one prepares it and the others wait. Then everyone reads the result. A split whose CSV exists is never rebuilt. To rebuild it, delete `splits/<split>.csv`.
