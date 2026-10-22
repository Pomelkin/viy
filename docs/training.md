# Training data

A `DetectionDataset` combines one or more datasets for one **stage**, either training or validation, into a regular `torch` dataset. Build one for training and one for validation. Every split is prepared on first use and checked whenever it is loaded.

## In Python

```python
import albumentations as A
from torch.utils.data import DataLoader

from viy.augs import AugmentationPipeline, NTIREPresetStep
from viy.data import DatasetSpec, DetectionCollator, DetectionDataset, Preprocessor, check_disjoint

train = DetectionDataset(
    [
        DatasetSpec("ntire", splits=["train"]),
        DatasetSpec("mllm_generated", limit=5000),
    ],
    stage="train",
    preprocessor=Preprocessor(image_size=256),
    augmentation=AugmentationPipeline([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(320, 320), scale=(0.78, 1.0), ratio=(0.82, 1.22)),
        NTIREPresetStep("aggressive"),
    ]),
    seed=42,
)
val = DetectionDataset(
    [DatasetSpec("ntire")],                       # default validation splits: val, val_hard
    stage="val",
    preprocessor=Preprocessor(image_size=256, resize_mode="resize_center_crop"),
)
check_disjoint(train, val)

collate = DetectionCollator()
train_loader = DataLoader(
    train, batch_size=128, shuffle=True, num_workers=16, persistent_workers=True, collate_fn=collate
)
val_loaders = {
    name: DataLoader(part, batch_size=128, num_workers=8, collate_fn=collate)
    for name, part in val.by_source().items()   # {"ntire/val": ..., "ntire/val_hard": ...}
}
```

Always pass `collate_fn=DetectionCollator()`: the default collate function of PyTorch cannot batch augmentation traces of different lengths.

### `DetectionDataset`

| Argument | Default | Meaning |
|---|---|---|
| `datasets` | required | list of `DatasetSpec` |
| `stage` | required | `"train"` or `"val"`, see below |
| `preprocessor` | required | turns an augmented `uint8` image into a model input, usually a `Preprocessor` |
| `augmentation` | `None` | `AugmentationPipeline` applied before preprocessing, see [Augmentations](augmentations.md) |
| `storage` | `None` | `StorageConfig` with the location of prepared datasets, see [Datasets](datasets.md#storage) |
| `seed` | `0` | seed of subsampling and augmentation |

### `DatasetSpec`

| Field | Default | Meaning |
|---|---|---|
| `name` | required | dataset name, e.g. `"ntire"` |
| `splits` | `None` | splits or split groups; `None` uses the dataset defaults for the stage |
| `limit` | `None` | at most this many images from **each** split, class-stratified |
| `raw_path` | `None` | local copy of the raw files; `None` downloads them from the Hugging Face Hub |

### Stages

| | `train` | `val` |
|---|---|---|
| default splits | training splits of the dataset; NTIRE: `shard_0` … `shard_5` | validation splits; NTIRE: `val`, `val_hard` |
| augmentation | random, new in every epoch | fixed per image |

A dataset without splits for the stage, such as `mllm_generated` for validation, is skipped with a warning. If no dataset has splits for the stage, an error is raised. `check_disjoint(train, val)` raises when an image appears in both datasets. Files are compared by resolved path, so the same image reached through different splits is caught as well.

## From a config

Each dataset has a pydantic config, `DetectionDatasetConfig`. It rejects unknown keys and nests into your training config:

```yaml
train_data:
  stage: train
  datasets:
    - {name: ntire, splits: [train]}
    - {name: mllm_generated, limit: 5000}
  preprocessing: {image_size: 256}
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
  preprocessing: {image_size: 256, resize_mode: resize_center_crop}
```

```python
from kostyl.utils import load_file
from viy.data import DetectionDataset, DetectionDatasetConfig, check_disjoint

config = load_file("train.yaml")
train = DetectionDataset.from_pydantic_model(DetectionDatasetConfig.model_validate(config["train_data"]))
val = DetectionDataset.from_pydantic_model(DetectionDatasetConfig.model_validate(config["val_data"]))
check_disjoint(train, val)
```

`DetectionDatasetConfig` has the same fields as the `DetectionDataset` arguments, with configs in place of objects:

| Field | Type |
|---|---|
| `datasets` | list of `{name, splits, limit, raw_path}` |
| `stage` | `train` or `val` |
| `preprocessing` | `PreprocessingConfig`, see [below](#preprocessing) |
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
| `index` | `int` | row of the image in its split |
| `augmentations` | `tuple[AppliedOp, ...]` | applied augmentations in order, each with `name` and `scale` |

`DetectionCollator` turns a list of samples into a `DetectionBatch`, also a `TypedDict`:

| Key | Type | Content |
|---|---|---|
| `images` | `torch.Tensor`, `(B, 3, H, W)` float | stacked images |
| `labels` | `torch.Tensor`, `(B,)` int64 | labels |
| `sources` | `list[str]` | source of every image |
| `indices` | `torch.Tensor`, `(B,)` int64 | row of every image in its split |
| `augmentations` | `list[tuple[AppliedOp, ...]]` | augmentation trace of every image |

```python
from viy.augs import trace_to_json
from viy.data import DetectionBatch

batch: DetectionBatch = next(iter(train_loader))
logits = model(batch["images"].cuda())
loss = criterion(logits.squeeze(1), batch["labels"].cuda().float())
batch["augmentations"][0]          # (AppliedOp(name='albumentations.HorizontalFlip', scale=None), AppliedOp(name='jpeg', scale=28))
trace_to_json(batch["augmentations"][0])   # '[{"name": "albumentations.HorizontalFlip", "scale": null}, ...]'
```

A `DetectionDataset` exposes:
- `.sources` lists its splits as `<dataset>/<split>`;
- `.by_source()` returns one `SplitDataset` per split;
- `.parts` lists the same `SplitDataset`s in order;
- `.labels` gives every label in order, handy for a weighted sampler.

A `SplitDataset` has `.records`, a list of `SampleRecord`, and `.image_path(index)`. With `batch["indices"]`, they give the full metadata of any sample, such as its generator or original distortions.

## Preprocessing

A `Preprocessor` is configured in one of two ways, never both:

- **Custom settings:** `image_size` plus optional `resize_mode`, `crop_padding`, `interpolation`, `mean` and `std`.
- **A transformers image processor**, which then defines resizing, cropping and normalization. Use it when the model expects the preprocessing of a pretrained backbone.

```python
from transformers import AutoImageProcessor
from viy.data import Preprocessor

custom = Preprocessor(image_size=224, resize_mode="shorter_side_center_crop", crop_padding=32)
pretrained = Preprocessor(image_processor=AutoImageProcessor.from_pretrained("facebook/dinov2-small"))
tensor = custom(pil_image_or_uint8_array)       # (3, 224, 224)
```

```yaml
preprocessing: {image_size: 256, resize_mode: resize_center_crop}
# or
preprocessing: {image_processor: facebook/dinov2-small}    # Hub name or local path
```

Custom settings:

| `resize_mode` | Effect |
|---|---|
| `resize` | squash to `image_size × image_size` |
| `resize_center_crop` | squash to `image_size + crop_padding`, then crop the centre |
| `shorter_side_center_crop` | keep the aspect ratio, scale the shorter side to `image_size + crop_padding`, then crop the centre |

`interpolation` is `bicubic` by default, with `bilinear`, `lanczos`, `box` and `nearest` also available; all except `nearest` are antialiased. `mean` and `std` default to the ImageNet values.

Augmentations run first, on the decoded `uint8` image, and preprocessing runs after them. Crops and downscales in the augmentations change the resolution; preprocessing brings every image to one size.

## Randomness and reproducibility

- **Training.** Each DataLoader worker owns a generator seeded from `torch.initial_seed()` and `seed`. Seed the DataLoader (`generator=torch.Generator().manual_seed(...)`) to reproduce a run. Augmentations change from epoch to epoch.
- **Validation.** Augmentation is seeded per image from `(seed, index)`, so every epoch sees the same images.
- **Trace.** `batch["augmentations"]` records the operations and levels that were applied, so results can be sliced by augmentation after training.

## Multi-GPU

`DetectionDataset` is a regular map-style dataset and works with `DistributedSampler`. Preparation of a split is guarded by a file lock: when every rank builds its datasets, one rank prepares each missing split and the others wait. To avoid waiting inside the job, prepare the splits once beforehand:

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
from viy.data import DetectionCollator, DetectionDataset, DetectionDatasetConfig, check_disjoint


class DetectionDataModule(L.LightningDataModule):
    def __init__(
        self, train_config: DetectionDatasetConfig, val_config: DetectionDatasetConfig, batch_size: int, num_workers: int
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
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=DetectionCollator(),
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(part, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=DetectionCollator())
            for part in self.val.by_source().values()
        ]
```

With a list of validation loaders, Lightning reports metrics per `dataloader_idx`; the matching split names are in `val.sources`.
