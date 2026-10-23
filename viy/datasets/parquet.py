import io
from abc import abstractmethod
from collections.abc import Iterator
from collections.abc import Mapping
from fnmatch import fnmatch
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from typing import ClassVar
from typing import Final
from typing import override

import pyarrow.parquet as pq
from kostyl.utils import setup_logger
from PIL import Image

from viy.layout import SampleRecord
from viy.layout import read_records
from viy.layout import write_records

from .base import DatasetBuilder
from .base import RawSample
from .base import SplitSpec
from .hub import HubArtifact
from .hub import download_artifact_file
from .hub import list_artifact_files


logger = setup_logger(fmt="detailed")

CONVERTED_DIR: Final = "converted"
BATCH_ROWS: Final = 64
IMAGE_SUFFIXES: Final = MappingProxyType(
    {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif"}
)


class ParquetShardBuilder(DatasetBuilder):
    """
    Base of datasets published as Parquet shards with embedded images.

    The shards of a split come from its streamed Hub artifacts, or from the
    same repository paths under ``raw_path``. Each shard is converted on its
    own. The original image bytes go to ``images/<split>/<shard>/`` without
    re-encoding, with the suffix of their actual format. The records go to
    ``converted/<split>/<shard>.csv``. A downloaded shard is deleted right
    after conversion. A split therefore never needs room for its shards and
    its images at once, and an interrupted preparation resumes from the first
    shard without records.
    """

    image_column: ClassVar[str] = "image"

    @abstractmethod
    def record(
        self, spec: SplitSpec, row: Mapping[str, Any], image: str
    ) -> SampleRecord:
        """
        Turn one Parquet row into a record.

        Args:
            spec (SplitSpec): Split being prepared.
            row (Mapping[str, Any]): Row without its image column.
            image (str): Canonical path the image of the row was written to.

        Returns:
            SampleRecord: Record of the row.
        """
        raise NotImplementedError

    @override
    def collect(self, spec: SplitSpec, raw_dir: Path) -> Iterator[RawSample]:
        for artifact in spec.artifacts:
            for shard in self._shards(artifact, raw_dir):
                for record in self._shard_records(spec, artifact, shard, raw_dir):
                    yield RawSample(self.layout.dir / record.image, record)
        return

    def _shards(self, artifact: HubArtifact, raw_dir: Path) -> list[str]:
        if self.raw_path is None:
            return list_artifact_files(artifact)
        base = raw_dir / artifact.subdir
        shards = sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*.parquet")
            if fnmatch(path.relative_to(base).as_posix(), artifact.pattern)
        )
        if not shards:
            raise FileNotFoundError(
                f"No Parquet shards matching '{artifact.pattern}' under {base}."
            )
        return shards

    def _shard_records(
        self, spec: SplitSpec, artifact: HubArtifact, shard: str, raw_dir: Path
    ) -> list[SampleRecord]:
        stem = PurePosixPath(shard).stem
        converted = self.layout.dir / CONVERTED_DIR / spec.name / f"{stem}.csv"
        if converted.is_file():
            return read_records(converted)
        downloaded = self.raw_path is None
        path = (
            download_artifact_file(artifact, shard, self.layout.raw_dir)
            if downloaded
            else raw_dir / artifact.subdir / shard
        )
        records = self._convert(spec, path, PurePosixPath("images", spec.name, stem))
        write_records(converted, records)
        if downloaded:
            path.unlink()
        return records

    def _convert(
        self, spec: SplitSpec, path: Path, directory: PurePosixPath
    ) -> list[SampleRecord]:
        (self.layout.dir / directory).mkdir(parents=True, exist_ok=True)
        records: list[SampleRecord] = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
            for row in batch.to_pylist():
                data = row.pop(self.image_column)["bytes"]
                image = str(directory / f"{len(records):06d}{_suffix(data, path)}")
                (self.layout.dir / image).write_bytes(data)
                records.append(self.record(spec, row, image))
        return records


def _suffix(data: bytes, shard: Path) -> str:
    with Image.open(io.BytesIO(data)) as image:
        image_format = image.format
    if image_format is None:
        raise ValueError(f"{shard}: an embedded image has no recognizable format.")
    return IMAGE_SUFFIXES.get(image_format, f".{image_format.lower()}")
