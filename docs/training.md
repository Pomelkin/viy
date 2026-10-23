# Training data

A `DetectionDataset` combines one or more datasets for one **stage**, either training or validation, into a regular `torch` dataset. Build one for training and one for validation. Every split is prepared on first use and checked whenever it is loaded.

## In Python

```python
import albumentations as A

from viy.augs import NTIREPresetStep
from viy.data import DatasetSpec, DetectionDataset, Preprocessor, check_disjoint

train = DetectionDataset(
    [
        DatasetSpec("ntire", splits="train"),
        DatasetSpec("mllm_generated", limit=5000),
    ],
    stage="train",
    preprocessor=Preprocessor(image_size=256),
    augmentation=[
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(320, 320), scale=(0.78, 1.0), ratio=(0.82, 1.22)),
        NTIREPresetStep("aggressive"),
    ],
    seed=42,
)
val = DetectionDataset(
    "ntire",  # default validation splits: val, val_hard
    stage="val",
    preprocessor=Preprocessor(image_size=256, resize_mode="resize_center_crop"),
)
check_disjoint(train, val)

train_loader = train.dataloader(batch_size=128, num_workers=16, persistent_workers=True)
# {"ntire/val": ..., "ntire/val_hard": ...}
val_loaders = val.dataloaders(batch_size=128, num_workers=8)
```

`dataloader` and `dataloaders` set `collate_fn=DetectionCollator()` and pass other arguments to `DataLoader`. Training data is shuffled with a generator seeded from `seed`. To build a `DataLoader` yourself, pass `collate_fn=DetectionCollator()`: the default collate function of PyTorch cannot batch augmentation traces of different lengths.

### `DetectionDataset`

| Argument | Default | Meaning |
|---|---|---|
| `datasets` | required | a `DatasetSpec`, a dataset name, or a list of them; a name uses the default splits |
| `stage` | required | `"train"` or `"val"`, see below |
| `preprocessor` | required | turns an augmented RGB `PIL.Image` into a model input, usually a `Preprocessor` |
| `augmentation` | `None` | a list of steps and albumentations transforms, an `AugmentationPipeline` or an `AugmentationConfig`, see [Augmentations](augmentations.md) |
| `storage` | `None` | `StorageConfig` with the location of prepared datasets, see [Datasets](datasets.md#storage) |
| `seed` | `0` | seed of subsampling and augmentation, see [Randomness](#randomness-and-reproducibility) |

### `DatasetSpec`

| Field | Default | Meaning |
|---|---|---|
| `name` | required | dataset name, e.g. `"ntire"` |
| `splits` | `None` | a split, a split group or a list of them; `None` uses the dataset defaults for the stage |
| `limit` | `None` | at most this many images from **each** split, class-stratified |
| `raw_path` | `None` | local copy of the raw files; `None` downloads them from the Hugging Face Hub |

### Stages

| | `train` | `val` |
|---|---|---|
| default splits | training splits of the dataset; NTIRE: `shard_0` … `shard_5` | validation splits; NTIRE: `val`, `val_hard` |
| augmentation | random, new in every epoch | fixed per image |

A dataset without default splits for the stage, such as `mllm_generated` for validation, is skipped with a warning; list its splits explicitly to use them anyway. Benchmark-only datasets, such as `df26`, are refused at both stages. If no dataset has splits for the stage, an error is raised. `check_disjoint(train, val)` raises when an image appears in both datasets. Files are compared by resolved path, so the same image reached through different splits is caught as well.

## From a config

Each dataset has a pydantic config, `DetectionDatasetConfig`. It rejects unknown keys and nests into your training config:

```yaml
train_data:
  stage: train
  datasets:
    - {name: ntire, splits: [train]}
    - {name: mllm_generated, limit: 5000}
  preprocessor: {image_size: 256}
  augmentation:
    steps:
      - {kind: albumentations, transform: HorizontalFlip, p: 0.5}
      - kind: albumentations
        transform: RandomResizedCrop
        params: {size: [320, 320], scale: [0.78, 1.0], ratio: [0.82, 1.22]}
      - {kind: ntire_preset, preset: aggressive}
  seed: 42

val_data:
  stage: val
  datasets:
    - {name: ntire}
  preprocessor: {image_size: 256, resize_mode: resize_center_crop}
```

```python
from viy import load_config
from viy.data import DetectionDataset, DetectionDatasetConfig, check_disjoint

config = load_config("train.yaml")
train = DetectionDataset.from_pydantic_model(
    DetectionDatasetConfig.model_validate(config["train_data"])
)
val = DetectionDataset.from_pydantic_model(
    DetectionDatasetConfig.model_validate(config["val_data"])
)
check_disjoint(train, val)
```

`DetectionDatasetConfig` has the same fields as the `DetectionDataset` arguments, with configs in place of objects. In a config, an `albumentations` step with `serialized` also describes composes such as `OneOf`, see [Augmentations](augmentations.md#albumentations-any-albumentations-transform).

| Field | Type |
|---|---|
| `datasets` | list of `{name, splits, limit, raw_path}` |
| `stage` | `train` or `val` |
| `preprocessor` | `PreprocessingConfig`, see [below](#preprocessing) |
| `augmentation` | `AugmentationConfig` or `null` |
| `storage` | `StorageConfig`; optional |
| `seed` | int |

## Samples and batches

Indexing a dataset returns a `DetectionSample`, a `TypedDict`:

| Key | Type | Content |
|---|---|---|
| `image` | `torch.Tensor`, `(3, H, W)` float | preprocessed image |
| `label` | `int` | `0` real, `1` generated |
| `source` | `str` | `<dataset>/<split>` the image comes from |
| `index` | `int` | position of the record in `SplitDataset.records`; with a `limit` it differs from the row of the split CSV |
| `path` | `str` | image path relative to the dataset directory, as in the split CSV and in benchmark predictions |
| `augmentations` | `tuple[AppliedOp, ...]` | applied augmentations in order, each with `name` and `scale` |

`DetectionCollator` turns a list of samples into a `DetectionBatch`, also a `TypedDict`:

| Key | Type | Content |
|---|---|---|
| `images` | `torch.Tensor`, `(B, 3, H, W)` float | stacked images |
| `labels` | `torch.Tensor`, `(B,)` int64 | labels |
| `sources` | `list[str]` | source of every image |
| `indices` | `torch.Tensor`, `(B,)` int64 | `index` of every sample |
| `paths` | `list[str]` | `path` of every sample |
| `augmentations` | `list[tuple[AppliedOp, ...]]` | augmentation trace of every image |

```python
from viy.augs import trace_to_json
from viy.data import DetectionBatch

batch: DetectionBatch = next(iter(train_loader))
logits = model(batch["images"].cuda())
loss = criterion(logits.squeeze(1), batch["labels"].cuda().float())
# (AppliedOp(name='albumentations.HorizontalFlip', scale=None), AppliedOp(name='jpeg', scale=28))
batch["augmentations"][0]
# '[{"name": "albumentations.HorizontalFlip", "scale": null}, ...]'
trace_to_json(batch["augmentations"][0])
```

A `DetectionDataset` exposes:
- `.sources` lists its splits as `<dataset>/<split>`;
- `.by_source()` returns one `SplitDataset` per split;
- `.parts` lists the same `SplitDataset`s in order;
- `.labels` gives every label in order, handy for a weighted sampler;
- `.record(source, index)` returns the `SampleRecord` of a sample, with its generator, distortions and other metadata: `train.record(batch["sources"][i], int(batch["indices"][i]))`;
- `.dataloader(...)` and `.dataloaders(...)` build DataLoaders, see above.

A `SplitDataset` has `.records`, a list of `SampleRecord`, and `.image_path(index)`.

## Preprocessing

A `Preprocessor` is configured in one of two ways, never both:

- **Custom settings:** `image_size` plus optional `resize_mode`, `crop_padding`, `interpolation`, `mean` and `std`.
- **A transformers image processor**, which then defines resizing, cropping and normalization. Use it when the model expects the preprocessing of a pretrained backbone.

```python
from viy.data import Preprocessor

custom = Preprocessor(
    image_size=224, resize_mode="shorter_side_center_crop", crop_padding=32
)
# a Hub name, a local path or a transformers image processor object
pretrained = Preprocessor(image_processor="facebook/dinov2-small")
tensor = custom(pil_image_or_uint8_array)  # (3, 224, 224)
```

```yaml
preprocessor: {image_size: 256, resize_mode: resize_center_crop}
# or
preprocessor: {image_processor: facebook/dinov2-small}    # Hub name or local path
```

The `preprocessor` of a dataset can be any function from an RGB `PIL.Image` to a tensor. `DetectorModel.preprocess` of the benchmark has the same signature, so one function serves both, and validation during training sees the inputs the benchmark scores.

Custom settings:

| `resize_mode` | Effect |
|---|---|
| `resize` | squash to `image_size × image_size` |
| `resize_center_crop` | squash to `image_size + crop_padding`, then crop the centre |
| `shorter_side_center_crop` | keep the aspect ratio, scale the shorter side to `image_size + crop_padding`, then crop the centre |

`interpolation` is `bicubic` by default, with `bilinear`, `lanczos`, `box` and `nearest` also available; all except `nearest` are antialiased. `mean` and `std` default to the ImageNet values.

Augmentations run first, on the decoded `uint8` array, and preprocessing runs after them on a `PIL.Image` of the result. Crops and downscales in the augmentations change the resolution; preprocessing brings every image to one size.

## Randomness and reproducibility

All data randomness comes from `seed`, plus the torch seed when you build DataLoaders yourself:

| Seed | Set with | Controls |
|---|---|---|
| `seed` | `DetectionDataset(seed=...)` or the config | subsampling with `limit`, every augmentation, and shuffling in `dataloader()` |
| torch seed | `torch.manual_seed`, `L.seed_everything` or `DataLoader(generator=...)` | shuffling and the base seed of workers of a DataLoader you build yourself |

With `train.dataloader(...)`, `seed` alone reproduces the order and the augmentations of a run.

- **Training.** Each process owns a generator seeded from `seed` and the distributed rank; a DataLoader worker adds its worker seed, which the DataLoader derives from its generator. A run is therefore reproducible, every epoch sees new augmentations, and DDP ranks never repeat each other's augmentations. The rank is read when the dataset is built, so build it after `init_process_group`; Lightning's `setup` already runs after it.
- **Validation.** Every image gets a generator seeded from `seed` and its path. It is augmented the same way in every epoch, whatever the `limit`, the order or the number of workers. A [derived benchmark split](benchmark.md#augmented-and-subsampled-tasks) with the same seed, limit and augmentation contains exactly these images.
- **Trace.** `batch["augmentations"]` records the operations and levels that were applied, so results can be sliced by augmentation after training.

## Multi-GPU

`DetectionDataset` is a regular map-style dataset and works with `DistributedSampler`. Preparation of a split is guarded by a file lock: when every rank builds its datasets, one rank prepares each missing split and the others wait. To avoid waiting inside the job, prepare the splits once beforehand:

```bash
viy datasets prepare ntire --split train --split val --split val_hard
```

or in Python:

```python
from viy.datasets import NTIRE

ntire = NTIRE()
for split in ntire.expand_splits(["train", "val", "val_hard"]):
    ntire.prepare(split)
```

## Lightning

```python
import lightning as L
from torch.utils.data import DataLoader
from viy.data import DetectionDataset, DetectionDatasetConfig, check_disjoint


class DetectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_config: DetectionDatasetConfig,
        val_config: DetectionDatasetConfig,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train: DetectionDataset | None = None
        self.val: DetectionDataset | None = None

    def setup(self, stage: str) -> None:
        if self.train is None:
            self.train = DetectionDataset.from_pydantic_model(self.train_config)
            self.val = DetectionDataset.from_pydantic_model(self.val_config)
            check_disjoint(self.train, self.val)

    def train_dataloader(self) -> DataLoader:
        return self.train.dataloader(
            self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )

    def val_dataloader(self) -> list[DataLoader]:
        return list(
            self.val.dataloaders(self.batch_size, num_workers=self.num_workers).values()
        )
```

With a list of validation loaders, Lightning reports metrics per `dataloader_idx`; the matching split names are in `val.sources`.
