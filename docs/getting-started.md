# Getting started

## Install

```bash
uv add "viy @ git+<repo-url>"        # as a dependency of your training project
# or, for development
git clone <repo-url> && cd viy && uv sync
```

Requires Python 3.13 or newer.

## Where the data goes

Every dataset is converted once into a canonical layout and reused afterwards. By default the layout lives in `~/.cache/viy`. To move it, set the `VIY_ROOT` environment variable or pass `storage=StorageConfig(root=...)`. Image files already on disk are symlinked, not copied; datasets published as Parquet shards or videos are converted into image files once.

A dataset can come from one of two places:

- **Hugging Face Hub**, the default. On first access the needed files are downloaded into `<root>/<dataset>/raw`, archives are unpacked and the images are linked.
- **A local raw copy.** Point viy at a directory that already holds the raw files to skip the download: `raw_path` of a `DatasetSpec`, `raw_paths` of a `Benchmark`, or `--raw-path` on the command line. [Datasets](datasets.md) describes the expected layout of each dataset.

Gated datasets, such as DF26, need an accepted access request on their Hub page and a token from `hf auth login` or `HF_TOKEN`. `viy datasets check <name>` tells whether a dataset can be prepared, without downloading it.

## Training data

```python
import albumentations as A

from viy.augs import NTIREPresetStep
from viy.data import DatasetSpec, DetectionDataset, Preprocessor

train = DetectionDataset(
    DatasetSpec("ntire", splits="shard_0"),
    stage="train",
    preprocessor=Preprocessor(image_size=256),
    augmentation=[A.HorizontalFlip(p=0.5), NTIREPresetStep("aggressive")],
    seed=42,
)
val = DetectionDataset("ntire", stage="val", preprocessor=Preprocessor(image_size=256))

train_loader = train.dataloader(batch_size=64, num_workers=8)
val_loaders = val.dataloaders(batch_size=64)
batch = next(iter(train_loader))
batch["images"].shape  # torch.Size([64, 3, 256, 256])
batch["labels"]  # 0 real, 1 generated
batch["sources"]  # ["ntire/shard_0", ...]
```

`train.dataloader` shuffles with a generator seeded from `seed` and batches samples with `DetectionCollator`; `val.dataloaders` gives one loader per validation split. The first call downloads one training shard (about 20 GB) and both validation splits (about 4 GB) from the Hugging Face Hub, unpacks them and links the images. Later calls only read the split CSV files and check that the images exist. Drop `splits` to train on all six shards.

## A benchmark run

```python
from typing import override

import torch
from PIL import Image

from viy.bench import Benchmark, DetectorModel, ModelMeta, results_table
from viy.data import Preprocessor


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
    ["ntire/val", "ntire/val_hard", "ntire/test_public"], output_dir="results"
)
results = benchmark.run(MyDetector())
print(results_table(results))
```

```
| model | ntire/test_public | ntire/val | ntire/val_hard |
|---|---:|---:|---:|
| my-detector@epoch_05 | 0.9405 | 0.9861 | 0.9120 |
```

The numbers above are illustrative. Per-slice metrics are in `results/my-detector__epoch_05/*.json`; see [Benchmark](benchmark.md). `preprocess` receives the same RGB `PIL.Image` as the `preprocessor` of a `DetectionDataset`, so one preprocessing function serves both, and validation during training sees the inputs the benchmark scores.

The same run from the command line, with the detector in `my_detectors.py` next to the config:

```yaml
# bench.yaml
model: {target: my_detectors:MyDetector}
output_dir: results
tasks: [ntire/val, ntire/val_hard, ntire/test_public]
```

```bash
viy bench run bench.yaml
viy bench results results
```

## Where to go next

- [Datasets](datasets.md): what each split contains and how to make your own.
- [Training data](training.md): the full configuration.
- [Augmentations](augmentations.md): presets and operations.
