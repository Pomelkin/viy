from pathlib import Path

from kostyl import StrictPydanticBaseModel
from pydantic import Field

from viy.layout import FileCheck
from viy.layout import LinkMode
from viy.layout import default_root


class StorageConfig(StrictPydanticBaseModel):
    """
    Where canonical datasets live and how they are prepared.

    Attributes:
        root (Path): Directory holding all canonical datasets; defaults to
            ``$VIY_ROOT`` if set and ``~/.cache/viy`` otherwise.
        auto_prepare (bool): Download and convert missing splits on first access.
        link_mode (LinkMode | None): How raw images are materialized; ``None``
            picks symlinks on POSIX systems and copies elsewhere.
        file_check (FileCheck): How many image files are checked for existence
            whenever a split is loaded.
    """

    root: Path = Field(default_factory=default_root)
    auto_prepare: bool = True
    link_mode: LinkMode | None = None
    file_check: FileCheck = FileCheck.ALL


class DatasetSourceConfig(StrictPydanticBaseModel):
    """
    A dataset from the supported pool and where its raw files come from.

    Attributes:
        name (str): Registered dataset name, e.g. ``ntire``.
        raw_path (Path | None): Local copy of the raw dataset. ``None`` downloads
            the raw files from the Hugging Face Hub.
    """

    name: str
    raw_path: Path | None = None
