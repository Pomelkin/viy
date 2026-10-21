# Training data

`TrainValDatasets.from_pydantic_model(TrainDataConfig)` does all of the following:
- prepares every configured split;
- runs the sanity checks;
- builds training and validation datasets with augmentation and preprocessing.

## Configuration

`TrainDataConfig` is a `StrictPydanticBaseModel` from `kostyl`: it rejects unknown keys and nests into a larger training config.

```yaml
data:
  storage:
    root: /data/viy
    file_check: all
  datasets:
    - name: ntire
      raw_path: /datasets/NTIRE        # optional local copy; omit to download from the Hub
      train_splits: [train, shard_6]
      val_splits: [val, val_hard]
    - name: mllm_generated
      train_limit: 5000
  preprocessing:
    image_size: 256
  val_preprocessing:
    image_size: 256
    resize_mode: resize_center_crop
    crop_padding: 32
  train_augmentation:
    steps:
      - {kind: albumentations, transform: HorizontalFlip, p: 0.5}
      - kind: albumentations
        transform: RandomResizedCrop
        params: {size: [320, 320], scale: [0.78, 1.0], ratio: [0.82, 1.22]}
      - {kind: ntire_preset, preset: aggressive_stage6}
  seed: 42
```

```python
from kostyl.utils import load_file
from viy.data import TrainDataConfig, TrainValDatasets

config = TrainDataConfig.model_validate(load_file("train.yaml")["data"])
data = TrainValDatasets.from_pydantic_model(config)
```

Inside your own config, declare the field as `data: TrainDataConfig` and pass `config.data`.

### `TrainDataConfig`

| Field | Default | Meaning |
|---|---|---|
| `storage` | required | `StorageConfig`, see [Datasets](datasets.md#storage-settings) |
| `datasets` | required | list of `TrainDatasetConfig`, each dataset at most once |
| `preprocessing` | required | `PreprocessingConfig` for training images |
| `val_preprocessing` | `None` | preprocessing of validation images; `None` reuses `preprocessing` |
| `train_augmentation` | `None` | `AugmentationConfig`, see [Augmentations](augmentations.md) |
| `val_augmentation` | `None` | augmentation of validation images, fixed per image |
| `seed` | `0` | seed of subsampling and augmentation |

### `TrainDatasetConfig`

| Field | Default | Meaning |
|---|---|---|
| `name` | required | dataset name, e.g. `ntire` |
| `raw_path` | `None` | local raw copy; `None` downloads from the Hub |
| `train_splits` | `None` | splits or groups for training; `None` uses the dataset defaults, `[]` excludes the dataset from training |
| `val_splits` | `None` | splits for validation; `None` uses the dataset defaults, `[]` excludes the dataset from validation |
| `train_limit` | `None` | at most this many images from **each** training split, class-stratified |
| `val_limit` | `None` | the same for validation splits |

### Rules

- **Dataset without validation splits.** The dataset is used for training only, and a warning says so.
- **No validation splits at all.** If none of the datasets has a validation split, an error is raised.
- **Overlap.** An image present in both training and validation is an error.
- **Subsampling.** Limits keep the class ratio and the original order. The same `seed` gives the same subset.

## Using the datasets

```python
import json

from torch.utils.data import DataLoader

train_loader = DataLoader(data.train, batch_size=128, shuffle=True, num_workers=16, persistent_workers=True)
val_loaders = {part.source: DataLoader(part, batch_size=128, num_workers=8) for part in data.val.parts}

for batch in train_loader:
    images = batch["image"]           # float tensor (B, 3, H, W), normalized
    labels = batch["label"]           # int tensor (B,), 0 real / 1 fake
    sources = batch["source"]         # list of "<dataset>/<split>"
    indices = batch["index"]          # row of each image in its split
    traces = [json.loads(trace) for trace in batch["augmentations"]]
    # traces[0] == [{"name": "jpeg", "scale": 28}, {"name": "downscale", "scale": 0.61}]
```

`data.train` and `data.val` are `ConcatDetectionDataset` objects:
- `.parts` lists the per-split `DetectionDataset`s;
- `.sources` gives their names;
- `.part("ntire/val")` returns one of them;
- `.labels` gives the labels in order, handy for a weighted sampler.

Each `DetectionDataset` has `.records` (a list of `SampleRecord`), `.labels` and `.image_path(index)`. Together with `batch["index"]` they give access to the full metadata of any sample, for example its generator or its original distortions.

## Preprocessing

| `resize_mode` | Effect |
|---|---|
| `resize` | squash to `image_size × image_size` |
| `resize_center_crop` | squash to `image_size + crop_padding`, then crop the centre |
| `shorter_side_center_crop` | keep the aspect ratio, scale the shorter side to `image_size + crop_padding`, then crop the centre |

Other fields:
- `interpolation`: `bicubic` by default; `bilinear`, `lanczos`, `box` and `nearest` are also available. Every filter except `nearest` is antialiased.
- `mean`, `std`: ImageNet values by default.

`Preprocessor` also works on its own, for example inside a benchmark model:

```python
from viy.data import Preprocessor, ResizeMode

prep = Preprocessor(image_size=224, resize_mode=ResizeMode.SHORTER_SIDE_CENTER_CROP, crop_padding=32)
tensor = prep(pil_image_or_uint8_array)       # (3, 224, 224)
```

Augmentations run first, on the decoded `uint8` image, and preprocessing runs after them. Crops and downscales inside augmentations change the resolution; preprocessing brings every image back to `image_size`.

## Randomness and reproducibility

- **Training.** Each DataLoader worker owns a generator seeded from `torch.initial_seed()` and `seed`. Seed the DataLoader (`generator=torch.Generator().manual_seed(...)`) to reproduce a run. Augmentations change from epoch to epoch.
- **Validation.** Augmentation is seeded per image from `(seed, index)`, so every epoch sees the same validation images.
- **Trace.** `batch["augmentations"]` records the operations and levels that were applied, so results can be sliced by augmentation after training.

## Multi-GPU

`ConcatDetectionDataset` is a regular map-style dataset, so `DistributedSampler` works with it. Split preparation is guarded by a file lock: if every rank builds `TrainValDatasets`, one prepares a missing split and the others wait. To avoid the wait inside the job, prepare once beforehand:

```python
from pathlib import Path

from viy.datasets import NTIRE

ntire = NTIRE(Path("/data/viy"))
for split in ntire.expand_splits(["train", "val", "val_hard"]):
    ntire.prepare(split)
```

## Lightning

```python
import lightning as L
from torch.utils.data import DataLoader
from viy.data import TrainDataConfig, TrainValDatasets


class DetectionDataModule(L.LightningDataModule):
    def __init__(self, config: TrainDataConfig, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data: TrainValDatasets | None = None

    def setup(self, stage: str) -> None:
        if self.data is None:
            self.data = TrainValDatasets.from_pydantic_model(self.config)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(part, batch_size=self.batch_size, num_workers=self.num_workers)
            for part in self.data.val.parts
        ]
```

With a list of validation loaders, Lightning reports metrics per `dataloader_idx`; the matching split names are in `data.val.sources`.
