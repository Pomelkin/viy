# Extending viy

Datasets and operations register themselves when their class is created, tasks and metrics through `register_task` and `register_metric`. A new one becomes available once the module defining it is imported; on the command line, import it with `viy --import my_module` or `imports` in a benchmark config. To add it to the library itself, put it in the matching package and import it from that package's `__init__.py`.

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
            artifacts=(HubArtifact("org/my-folders", "train/**", revision="<commit>"),),
            description="Training images.",
        ),
        SplitSpec(
            name="test",
            role=SplitRole.TEST,
            artifacts=(
                HubArtifact(
                    "org/my-folders", "test.zip", extract=True, revision="<commit>"
                ),
            ),
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
| `description` | one line shown by `viy datasets list` |
| `split_specs` | splits the builder can prepare; role `val` or `test` makes the split a benchmark task |
| `HubArtifact(repo_id, pattern, subdir, extract)` | Hub files the split needs; `pattern` is a glob over the repository, files land in `raw/<subdir>`, and zip archives are unpacked and deleted when `extract=True` |
| `HubArtifact(..., revision=...)` | pins a commit, branch or tag, so the split does not change when the repository does |
| `HubArtifact(..., gated=True)` | the repository needs an accepted access request; viy checks the token and the access before downloading |
| `benchmark_only = True` | `DetectionDataset` refuses the dataset, for licenses that allow evaluation only |
| a split without artifacts | local only: preparing it requires `raw_path` |
| `split_groups` | named lists of splits, e.g. `{"train": ("shard_0", ..., "shard_5")}` |
| `default_train_splits`, `default_val_splits` | used when a training config does not list splits |
| `collect` | maps raw files to records; set `generator`, `distortions` and `source_id` when the source provides them, and leave them `None` otherwise |

Guidelines:
- Keep record paths under `images/<split>/...` and make them unique within the dataset.
- Never guess unknown metadata: `None` means unknown, and `()` means a known clean image.
- Raise `FileNotFoundError` with the expected path when the raw layout does not match. The sanity checks catch missing images on their own.
- A builder that produces images itself, for example by decoding videos, writes them to `self.layout.dir / record.image` and yields that path as the `source`; linking leaves such images in place.
- Test the builder on a tiny synthetic raw directory, as `tests/conftest.py` does for NTIRE. Do not download real data in tests.

### Datasets published as Parquet shards

For a dataset whose images are embedded in Parquet shards, as `datasets` publishes them, subclass `ParquetShardBuilder` and implement `record` instead of `collect`. Mark the artifacts `streamed=True`: each shard is then downloaded, converted and deleted in turn, and an interrupted preparation resumes from the first unconverted shard.

```python
from collections.abc import Mapping
from typing import Any, override

from viy.datasets import HubArtifact, ParquetShardBuilder, SplitRole, SplitSpec
from viy.layout import SampleRecord


class MyParquet(ParquetShardBuilder):
    name = "my_parquet"
    description = "Images embedded in Parquet shards."
    split_specs = (
        SplitSpec(
            name="test",
            role=SplitRole.TEST,
            artifacts=(
                HubArtifact(
                    "org/my-parquet",
                    "data/test-*.parquet",
                    revision="<commit>",
                    streamed=True,
                ),
            ),
        ),
    )

    @override
    def record(
        self, spec: SplitSpec, row: Mapping[str, Any], image: str
    ) -> SampleRecord:
        return SampleRecord(
            image=image, label=int(row["label"]), generator=row.get("model")
        )
```

`row` holds every column except `image`, the column with the `{bytes, path}` struct; set `image_column` if it is named differently.

## A new augmentation operation

Subclass `ImageOp`. The operation must:
- take its randomness only from `rng`;
- leave the input array unmodified;
- return a `uint8` RGB array, which may have a different size.

```python
from typing import override

import numpy as np

from viy.augs import ImageArray, ImageOp, ScaleValue


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
    def apply(
        self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
    ) -> ImageArray:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        radius = np.hypot(
            (xx - width / 2) / (width / 2), (yy - height / 2) / (height / 2)
        ) / np.sqrt(2)
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
from viy.bench import TaskMetadata, register_task

register_task(
    TaskMetadata(
        name="my_folders/test_tpr",
        dataset="my_folders",
        split="test",
        main_score="tpr_at_1pct_fpr",
    )
)
```

The command line sees datasets, tasks and metrics defined in your modules once you import them: `viy --import my_datasets ...`, or `imports: [my_datasets]` in a benchmark config.

A task with a `derivation` evaluates a seeded subset or an augmented copy of its split instead, see [Augmented and subsampled tasks](benchmark.md#augmented-and-subsampled-tasks).

A new metric needs no task: register it in the metric pool, see [Metrics](benchmark.md#metrics). To evaluate differently in other ways, subclass `DetectionTask` and override `evaluate(records, scores, threshold, model, evaluation_seconds, metrics)`. It must return a `TaskResult`.
