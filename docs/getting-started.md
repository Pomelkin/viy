# Getting started

## Install

```bash
uv add "viy @ git+<repo-url>"        # as a dependency of your training project
# or, for development
git clone <repo-url> && cd viy && uv sync
```

Requires Python 3.13 or newer.

## Pick a storage root

Every dataset is converted into a canonical layout under one directory, the storage **root**. Nothing is copied when the raw data is already on disk: images are symlinked.

```python
from pathlib import Path
from viy.datasets import StorageConfig

storage = StorageConfig(root=Path("/data/viy"))
```

A dataset can come from one of two places:

- **Hugging Face Hub**, the default. On first access the needed files are downloaded into `<root>/<dataset>/raw`, archives are unpacked and the images are linked.
- **A local raw copy.** Set `raw_path` to a directory that already holds the raw files to skip the download. [Datasets](datasets.md) describes the expected layout of each dataset.

## Training data in five lines

```python
from torch.utils.data import DataLoader
from viy.data import TrainDataConfig, TrainValDatasets

config = TrainDataConfig.model_validate({
    "storage": {"root": "/data/viy"},
    "datasets": [{"name": "ntire", "train_splits": ["shard_0"]}],
    "preprocessing": {"image_size": 256},
    "train_augmentation": {"steps": [{"kind": "ntire_preset", "preset": "val"}]},
})
data = TrainValDatasets.from_pydantic_model(config)

train_loader = DataLoader(data.train, batch_size=64, shuffle=True, num_workers=8)
batch = next(iter(train_loader))
batch["image"].shape   # torch.Size([64, 3, 256, 256])
batch["label"]         # 0 real, 1 fake
batch["source"]        # ["ntire/shard_2", ...]
```

The first call downloads one training shard (about 20 GB) and both validation splits (about 4 GB) from the Hugging Face Hub, unpacks them and links the images. Later calls only read the split CSV files and check that the images exist. Drop `train_splits` to train on all six shards.

## A benchmark run

```python
from pathlib import Path
from typing import override

import torch
from PIL import Image

from viy.bench import Benchmark, DetectorModel, ModelMeta, get_tasks, results_table
from viy.data import Preprocessor
from viy.datasets import StorageConfig


class MyDetector(DetectorModel):
    def __init__(self) -> None:
        self.net = load_my_network().eval().cuda()
        self.prep = Preprocessor(image_size=256)

    @property
    @override
    def meta(self) -> ModelMeta:
        return ModelMeta(name="my-detector", revision="epoch_05")

    @override
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.prep(image)

    @override
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch.cuda()).sigmoid().squeeze(1)


benchmark = Benchmark(
    tasks=get_tasks(["ntire/val", "ntire/val_hard", "ntire/test_public"]),
    storage=StorageConfig(root=Path("/data/viy")),
    output_dir=Path("results"),
)
results = benchmark.run(MyDetector())
print(results_table(results))
```

```
| model | ntire/test_public | ntire/val | ntire/val_hard |
|---|---:|---:|---:|
| my-detector@epoch_05 | 0.9405 | 0.9861 | 0.9120 |
```

The numbers above are illustrative. Per-slice metrics are in `results/my-detector__epoch_05/*.json`; see [Benchmark](benchmark.md).

## Where to go next

- [Datasets](datasets.md): what each split contains and how to make your own.
- [Training data](training.md): the full configuration.
- [Augmentations](augmentations.md): presets and operations.
