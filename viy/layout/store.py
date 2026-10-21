import os
import re
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Final

from kostyl import StrictPydanticBaseModel
from kostyl.utils import DirLock
from pydantic import Field

from .records import SampleRecord
from .records import read_records
from .records import write_records


LAYOUT_VERSION: Final = 1
ROOT_ENV_VAR: Final = "VIY_ROOT"
_SPLIT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def default_root() -> Path:
    """Directory of canonical datasets: ``$VIY_ROOT`` if set, ``~/.cache/viy`` otherwise."""
    return Path(os.environ.get(ROOT_ENV_VAR, Path.home() / ".cache" / "viy"))


class SplitEntry(StrictPydanticBaseModel):
    """Provenance of a split prepared by a dataset builder."""

    num_samples: int
    num_real: int
    num_fake: int
    source: str
    prepared_at: datetime


class LayoutManifest(StrictPydanticBaseModel):
    """Content of ``dataset.json`` in a canonical dataset directory."""

    layout_version: int = LAYOUT_VERSION
    dataset: str
    splits: dict[str, SplitEntry] = Field(default_factory=dict)


class DatasetLayout:
    """
    Canonical on-disk layout shared by every dataset.

    ```
    <root>/<dataset>/
        dataset.json          provenance of prepared splits
        images/...            symlinks or copies of the raw images
        splits/<split>.csv    one table per split, paths relative to <root>/<dataset>
        raw/...               raw files downloaded from the Hugging Face Hub
    ```

    A split exists if and only if its CSV exists, so a custom split is created
    by writing a CSV that references images already present in ``images/``.
    """

    MANIFEST_NAME: Final = "dataset.json"

    def __init__(self, root: Path, dataset: str) -> None:
        """
        Initialize the layout of one dataset.

        Args:
            root (Path): Directory holding all canonical datasets.
            dataset (str): Dataset name, used as the directory name.
        """
        self.root = Path(root)
        self.dataset = dataset
        return

    @property
    def dir(self) -> Path:
        """Dataset directory."""
        return self.root / self.dataset

    @property
    def images_dir(self) -> Path:
        """Directory with the materialized images."""
        return self.dir / "images"

    @property
    def splits_dir(self) -> Path:
        """Directory with the split CSV files."""
        return self.dir / "splits"

    @property
    def raw_dir(self) -> Path:
        """Directory for raw files downloaded from the Hub."""
        return self.dir / "raw"

    @property
    def manifest_path(self) -> Path:
        """Path of ``dataset.json``."""
        return self.dir / self.MANIFEST_NAME

    def split_path(self, split: str) -> Path:
        """Path of the CSV of ``split``."""
        if not _SPLIT_NAME.fullmatch(split):
            raise ValueError(
                f"Invalid split name '{split}': use letters, digits, '_', '.' and '-'."
            )
        return self.splits_dir / f"{split}.csv"

    def has_split(self, split: str) -> bool:
        """Whether the CSV of ``split`` exists."""
        return self.split_path(split).is_file()

    def splits(self) -> list[str]:
        """Names of all splits present on disk."""
        if not self.splits_dir.is_dir():
            return []
        return sorted(path.stem for path in self.splits_dir.glob("*.csv"))

    def read_split(self, split: str) -> list[SampleRecord]:
        """Read the records of ``split``."""
        path = self.split_path(split)
        if not path.is_file():
            raise FileNotFoundError(
                f"Split '{split}' of dataset '{self.dataset}' not found: {path}. "
                f"Available: {', '.join(self.splits()) or 'none'}."
            )
        return read_records(path)

    def write_split(
        self, split: str, records: Sequence[SampleRecord], source: str
    ) -> None:
        """
        Write the CSV of ``split`` and record its provenance in the manifest.

        Args:
            split (str): Split name.
            records (Sequence[SampleRecord]): Records of the split.
            source (str): Where the split was prepared from.
        """
        write_records(self.split_path(split), records)
        manifest = self.read_manifest()
        num_fake = sum(record.is_fake for record in records)
        manifest.splits[split] = SplitEntry(
            num_samples=len(records),
            num_real=len(records) - num_fake,
            num_fake=num_fake,
            source=source,
            prepared_at=datetime.now(UTC),
        )
        temporary = self.manifest_path.with_name(f"{self.MANIFEST_NAME}.tmp")
        temporary.write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.manifest_path)
        return

    def read_manifest(self) -> LayoutManifest:
        """Read ``dataset.json``, or an empty manifest if it does not exist yet."""
        if not self.manifest_path.is_file():
            return LayoutManifest(dataset=self.dataset)
        manifest = LayoutManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        if manifest.layout_version != LAYOUT_VERSION:
            raise ValueError(
                f"{self.manifest_path}: layout version {manifest.layout_version} is not supported, "
                f"expected {LAYOUT_VERSION}."
            )
        return manifest

    def image_path(self, record: SampleRecord) -> Path:
        """Absolute path of the image of ``record``."""
        return self.dir / record.image

    def lock(self) -> DirLock:
        """Cross-process lock guarding preparation of this dataset."""
        self.dir.mkdir(parents=True, exist_ok=True)
        return DirLock(self.dir)
