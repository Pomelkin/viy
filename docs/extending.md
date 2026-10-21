# Extending viy

Datasets, operations and tasks register themselves when their class or object is created. A new one becomes available once the module defining it is imported. To add it to the library itself, put it in the matching package and import it from that package's `__init__.py`.

## A new dataset

Subclass `DatasetBuilder` and implement `collect`, which yields every raw image of a split together with its canonical record. Downloading, unpacking, linking, locking and the sanity checks are shared.

The example below covers a dataset published as `<split>/{real,fake}/*.jpg`:

```python
from collections.abc import Iterator
from pathlib import Path
from typing import override

from viy.datasets import DatasetBuilder, HubArtifact, RawSample, SplitRole, SplitSpec
from viy.layout import SampleRecord


class MyFolders(DatasetBuilder):
    """Real and generated images sorted into folders: ``<split>/{real,fake}/<file>``."""

    name = "my_folders"
    description = "Real and generated images sorted into folders."
    split_specs = (
        SplitSpec(
            name="train",
            role=SplitRole.TRAIN,
            artifacts=(HubArtifact("org/my-folders", "train/**"),),
            description="Training images.",
        ),
        SplitSpec(
            name="test",
            role=SplitRole.TEST,
            artifacts=(HubArtifact("org/my-folders", "test.zip", extract=True),),
            description="Test images, published as a zip with a test/ folder inside.",
        ),
    )
    default_train_splits = ("train",)
    default_val_splits = ()

    @override
    def collect(self, spec: SplitSpec, raw_dir: Path) -> Iterator[RawSample]:
        for label, folder in enumerate(("real", "fake")):
            for path in sorted((raw_dir / spec.name / folder).glob("*.jpg")):
                yield RawSample(
                    source=path,
                    record=SampleRecord(
                        image=f"images/{spec.name}/{folder}/{path.name}",
                        label=label,
                    ),
                )
```

What each part does:

| Part | Purpose |
|---|---|
| `name` | dataset name used in configs and as the directory under the storage root |
| `split_specs` | splits the builder can prepare; role `val` or `test` makes the split a benchmark task |
| `HubArtifact(repo_id, pattern, subdir, extract)` | Hub files the split needs; `pattern` is a glob over the repository, files land in `raw/<subdir>`, and zip archives are unpacked and deleted when `extract=True` |
| a split without artifacts | local only: preparing it requires `raw_path` |
| `split_groups` | named lists of splits, e.g. `{"train": ("shard_0", ..., "shard_5")}` |
| `default_train_splits`, `default_val_splits` | used when a training config does not list splits |
| `collect` | maps raw files to records; set `generator`, `distortions` and `source_id` when the source provides them, and leave them `None` otherwise |

Guidelines:
- Keep record paths under `images/<split>/...` and make them unique within the dataset.
- Never guess unknown metadata: `None` means unknown, and `()` means a known clean image.
- Raise `FileNotFoundError` with the expected path when the raw layout does not match. The sanity checks catch missing images on their own.
- Test the builder on a tiny synthetic raw directory, as `tests/conftest.py` does for NTIRE. Do not download real data in tests.

## A new augmentation operation

Subclass `ImageOp`. The operation must:
- take its randomness only from `rng`;
- leave the input array unmodified;
- return a `uint8` RGB array, which may have a different size.

```python
from typing import override

import numpy as np

from viy.augs import ImageOp
from viy.common import ImageArray, ScaleValue


class Vignette(ImageOp):
    """Darken the corners; ``level`` is the darkening at the corners."""

    name = "vignette"
    default_levels = (0.1, 0.2, 0.4)

    @override
    def validate_level(self, level: ScaleValue) -> None:
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"'{self.name}' level must be in [0, 1], got {level}.")
        return

    @override
    def apply(self, image: ImageArray, level: ScaleValue, rng: np.random.Generator) -> ImageArray:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        radius = np.hypot((xx - width / 2) / (width / 2), (yy - height / 2) / (height / 2)) / np.sqrt(2)
        gain = 1.0 - float(level) * radius**2
        return np.clip(image * gain[..., None], 0, 255).astype(np.uint8)
```

Optional class attributes:

| Attribute | Default | Purpose |
|---|---|---|
| `level_type` | `float` | type the levels are converted to: `int`, `float` or `str` |
| `proxy_of` | `None` | real distortion this operation approximates; a proxy must also end its name with `_proxy` and logs a warning when created |

For a severity drawn from a continuous range instead of discrete levels, subclass `RangeImageOp`; its levels are `(low, high)`.

Once the module is imported, the operation works in every step kind: `{kind: op, op: vignette}`, inside chain groups, and through `build_op("vignette")`. When it reproduces a distortion recorded in a dataset, use the same name as the dataset metadata, so traces and benchmark slices line up.

## A new benchmark task

Built-in tasks come from dataset splits automatically. Register a task explicitly to evaluate a custom split, or to change the main score:

```python
from viy.bench import DetectionTask, TaskMetadata, register_task

register_task(DetectionTask(TaskMetadata(
    name="my_folders/test_tpr",
    dataset="my_folders",
    split="test",
    main_score="tpr_at_1pct_fpr",
)))
```

To compute something new, subclass `DetectionTask` and override `evaluate(records, scores, threshold, model, evaluation_seconds)`. It must return a `TaskResult`.
