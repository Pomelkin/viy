import inspect
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from typing import ClassVar

from kostyl.utils import setup_logger

from viy.layout import DatasetLayout
from viy.layout import LayoutError
from viy.layout import SampleRecord
from viy.layout import check_split
from viy.layout import default_link_mode
from viy.layout import link_files
from viy.progress import track
from viy.registry import Registry

from .configs import DatasetSourceConfig
from .configs import StorageConfig
from .hub import HubArtifact
from .hub import check_access
from .hub import fetch_artifact


logger = setup_logger(fmt="detailed")


class SplitRole(StrEnum):
    """Intended use of a split."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """
    A split a dataset builder knows how to prepare.

    Attributes:
        name (str): Split name in the canonical layout.
        role (SplitRole): Intended use of the split.
        artifacts (tuple[HubArtifact, ...]): Hub files the split is built from.
            An empty tuple marks a split that is only available from a local copy.
        description (str): Short description shown in listings and results.
    """

    name: str
    role: SplitRole
    artifacts: tuple[HubArtifact, ...] = ()
    description: str = ""

    @property
    def local_only(self) -> bool:
        """Whether the split is not published on the Hub."""
        return len(self.artifacts) == 0


@dataclass(frozen=True, slots=True)
class RawSample:
    """
    A raw image together with its canonical record.

    Attributes:
        source (Path): Image file in the raw dataset.
        record (SampleRecord): Record whose ``image`` is the canonical location.
    """

    source: Path
    record: SampleRecord


DATASETS: Registry[type["DatasetBuilder"]] = Registry("dataset")


class DatasetBuilder(ABC):
    """
    Converts one dataset of the supported pool into the canonical layout and reads it.

    Concrete builders that define their own ``name`` are registered under it
    on class creation, so shared intermediate classes stay unregistered. A builder
    only knows how to turn raw files into records in :meth:`collect`;
    downloading, linking, locking and sanity checks are shared.

    A dataset with ``benchmark_only`` set, for example one whose license
    allows evaluation only, is refused by training and validation datasets
    and is used through benchmark tasks alone.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    split_specs: ClassVar[tuple[SplitSpec, ...]]
    split_groups: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType({})
    default_train_splits: ClassVar[tuple[str, ...]] = ()
    default_val_splits: ClassVar[tuple[str, ...]] = ()
    benchmark_only: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and "name" in vars(cls):
            DATASETS.add(cls.name, cls)
        return

    def __init__(
        self, raw_path: Path | str | None = None, storage: StorageConfig | None = None
    ) -> None:
        """
        Initialize the builder.

        Args:
            raw_path (Path | str | None): Local copy of the raw dataset. ``None``
                downloads raw files from the Hub into ``<root>/<name>/raw``.
            storage (StorageConfig | None): Canonical layout location and preparation
                settings; ``None`` uses the defaults.
        """
        self.storage = StorageConfig() if storage is None else storage
        self.layout = DatasetLayout(self.storage.root, self.name)
        self.raw_path = None if raw_path is None else Path(raw_path)
        self.auto_prepare = self.storage.auto_prepare
        self.link_mode = (
            default_link_mode()
            if self.storage.link_mode is None
            else self.storage.link_mode
        )
        self.file_check = self.storage.file_check
        return

    @classmethod
    def from_pydantic_model(
        cls, source: DatasetSourceConfig, storage: StorageConfig
    ) -> "DatasetBuilder":
        """
        Create the builder registered under ``source.name``.

        Args:
            source (DatasetSourceConfig): Dataset name and raw files location.
            storage (StorageConfig): Canonical layout location and preparation settings.

        Returns:
            DatasetBuilder: Builder of the requested dataset.
        """
        builder_cls = DATASETS.get(source.name)
        if not issubclass(builder_cls, cls):
            raise TypeError(f"Dataset '{source.name}' is not a {cls.__name__}.")
        return builder_cls(raw_path=source.raw_path, storage=storage)

    @classmethod
    def split_spec(cls, split: str) -> SplitSpec | None:
        """Specification of ``split``, or ``None`` for splits the builder does not know."""
        return next((spec for spec in cls.split_specs if spec.name == split), None)

    @classmethod
    def eval_splits(cls) -> tuple[str, ...]:
        """Splits meant for validation or testing."""
        return tuple(
            spec.name
            for spec in cls.split_specs
            if spec.role in {SplitRole.VAL, SplitRole.TEST}
        )

    @classmethod
    def train_role_splits(cls) -> tuple[str, ...]:
        """Splits meant for training."""
        return tuple(
            spec.name for spec in cls.split_specs if spec.role == SplitRole.TRAIN
        )

    def expand_splits(self, names: Iterable[str]) -> list[str]:
        """
        Replace split groups by their members, keeping order and dropping repeats.

        Args:
            names (Iterable[str]): Split or group names, e.g. ``["train", "shard_6"]``.

        Returns:
            list[str]: Concrete split names.
        """
        expanded: list[str] = []
        for name in names:
            for split in self.split_groups.get(name, (name,)):
                if split not in expanded:
                    expanded.append(split)
        return expanded

    def load(self, split: str) -> list[SampleRecord]:
        """
        Load a split, preparing it first if needed, and run sanity checks.

        Args:
            split (str): Concrete split name; groups must be expanded first.

        Returns:
            list[SampleRecord]: Records of the split.
        """
        self.prepare(split)
        records = self.layout.read_split(split)
        check_split(records, f"{self.name}/{split}", self.layout.dir, self.file_check)
        return records

    def check_available(self, split: str) -> None:
        """
        Make sure ``split`` is prepared or can be prepared, without downloading it.

        Checks the split name, ``auto_prepare``, the local copy for ``raw_path``
        and for splits that are not on the Hub, and access to gated repositories.

        Args:
            split (str): Concrete split name.

        Raises:
            LayoutError: If the split is unknown or cannot be prepared.
            PermissionError: If a gated repository needs a Hugging Face token.
            huggingface_hub.errors.GatedRepoError: If access to a gated repository was not granted.
        """
        if self.layout.has_split(split):
            return
        spec = self._known_spec(split)
        if not self.auto_prepare:
            raise LayoutError(
                f"Split '{self.name}/{split}' is not prepared in {self.layout.dir} and auto_prepare is off."
            )
        if self.raw_path is not None:
            if not self.raw_path.is_dir():
                raise LayoutError(
                    f"Raw dataset directory does not exist: {self.raw_path}"
                )
            return
        if spec.local_only:
            raise LayoutError(
                f"Split '{self.name}/{split}' is not published on the Hugging Face Hub; "
                "set raw_path to a local copy of the dataset."
            )
        for repo_id in sorted(
            {artifact.repo_id for artifact in spec.artifacts if artifact.gated}
        ):
            check_access(repo_id)
        return

    def prepare(self, split: str) -> None:
        """
        Build ``split`` in the canonical layout unless it already exists.

        Preparation is guarded by a cross-process lock, so concurrent ranks wait
        for one of them to finish instead of downloading in parallel.

        Raises:
            LayoutError: If the split is unknown or cannot be prepared.
        """
        if self.layout.has_split(split):
            return
        self.check_available(split)
        spec = self._known_spec(split)
        with self.layout.lock():
            if not self.layout.has_split(split):
                self._build(spec)
        return

    def _known_spec(self, split: str) -> SplitSpec:
        if split in self.split_groups:
            raise LayoutError(
                f"'{split}' is a group of splits of '{self.name}': {list(self.split_groups[split])}. "
                "Expand it with expand_splits first."
            )
        spec = self.split_spec(split)
        if spec is None:
            known = sorted(
                {*(known.name for known in self.split_specs), *self.layout.splits()}
            )
            raise LayoutError(
                f"Unknown split '{split}' of dataset '{self.name}'. Known splits: {', '.join(known)}."
            )
        return spec

    def _build(self, spec: SplitSpec) -> None:
        raw_dir = self._materialize_raw(spec)
        logger.info(f"Preparing split '{self.name}/{spec.name}' from {raw_dir}")
        samples = list(
            track(self.collect(spec, raw_dir), f"Collecting {self.name}/{spec.name}")
        )
        if len(samples) == 0:
            raise LayoutError(
                f"No images found for split '{self.name}/{spec.name}' in {raw_dir}."
            )
        link_files(
            [
                (sample.source, self.layout.dir / sample.record.image)
                for sample in samples
            ],
            self.link_mode,
        )
        self.layout.write_split(
            spec.name, [sample.record for sample in samples], source=str(raw_dir)
        )
        logger.info(
            f"Prepared split '{self.name}/{spec.name}' with {len(samples)} images"
        )
        return

    def _materialize_raw(self, spec: SplitSpec) -> Path:
        if self.raw_path is not None:
            return self.raw_path
        for artifact in spec.artifacts:
            if not artifact.streamed:
                fetch_artifact(artifact, self.layout.raw_dir)
        return self.layout.raw_dir

    @abstractmethod
    def collect(self, spec: SplitSpec, raw_dir: Path) -> Iterator[RawSample]:
        """
        Yield the raw images of a split with their canonical records.

        A builder that produces images itself, by decoding videos or Parquet
        shards, writes them to their canonical location and yields that
        location as the source.

        Args:
            spec (SplitSpec): Split to collect.
            raw_dir (Path): Root of the raw dataset, local or downloaded.

        Yields:
            RawSample: Raw image and its record.
        """
        raise NotImplementedError
