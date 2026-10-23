import os
import shutil
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from viy.progress import track


class LinkMode(StrEnum):
    """How images of a raw dataset are materialized in the canonical layout."""

    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    COPY = "copy"


def default_link_mode() -> LinkMode:
    """Symlinks on POSIX systems, copies elsewhere."""
    return LinkMode.SYMLINK if os.name == "posix" else LinkMode.COPY


def link_files(pairs: Sequence[tuple[Path, Path]], mode: LinkMode) -> None:
    """
    Materialize every ``(source, target)`` pair.

    Already materialized targets are kept, so an interrupted run can be repeated.
    A pair whose source is its target, an image a builder wrote in place, is
    left as it is.

    Args:
        pairs (Sequence[tuple[Path, Path]]): Raw files and their canonical locations.
        mode (LinkMode): Materialization mode.

    Raises:
        FileNotFoundError: If a source file does not exist.
    """
    for parent in sorted({target.parent for _, target in pairs}):
        parent.mkdir(parents=True, exist_ok=True)
    for source, target in track(pairs, f"Materializing images ({mode})"):
        _link(source, target, mode)
    return


def _link(source: Path, target: Path, mode: LinkMode) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Raw file does not exist: {source}")
    if source == target:
        return
    if target.is_symlink() or target.exists():
        if _is_materialized(source, target, mode):
            return
        target.unlink()
    match mode:
        case LinkMode.SYMLINK:
            target.symlink_to(source.resolve())
        case LinkMode.HARDLINK:
            target.hardlink_to(source)
        case LinkMode.COPY:
            shutil.copy2(source, target)
    return


def _is_materialized(source: Path, target: Path, mode: LinkMode) -> bool:
    match mode:
        case LinkMode.SYMLINK:
            return target.is_symlink() and target.readlink() == source.resolve()
        case LinkMode.HARDLINK:
            return target.samefile(source)
        case LinkMode.COPY:
            return (
                not target.is_symlink()
                and target.stat().st_size == source.stat().st_size
            )
