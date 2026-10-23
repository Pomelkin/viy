import random
from collections import Counter
from collections.abc import Iterable
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from pathlib import PurePosixPath
from typing import Final

from kostyl.utils import setup_logger

from viy.common import FAKE_LABEL
from viy.common import REAL_LABEL
from viy.progress import track

from .records import SampleRecord


logger = setup_logger(fmt="detailed")

FILE_CHECK_SAMPLE_SIZE: Final = 1000
FORMAT_SHORTCUT_DISTANCE: Final = 0.5


class LayoutError(RuntimeError):
    """A canonical dataset is missing, incomplete or inconsistent."""


class FileCheck(StrEnum):
    """How many image files are checked for existence when a split is loaded."""

    ALL = "all"
    SAMPLE = "sample"
    NONE = "none"


def check_split(
    records: Sequence[SampleRecord],
    name: str,
    dataset_dir: Path,
    file_check: FileCheck = FileCheck.ALL,
) -> None:
    """
    Run sanity checks on a loaded split.

    Errors are raised for empty splits, duplicated images and missing files.
    Single-class splits and class-dependent file formats are reported as warnings.

    Args:
        records (Sequence[SampleRecord]): Records of the split.
        name (str): Split name used in messages, e.g. ``ntire/val``.
        dataset_dir (Path): Directory the image paths are relative to.
        file_check (FileCheck): How many image files to check for existence.

    Raises:
        LayoutError: If the split is empty, has duplicated images or missing files.
    """
    if len(records) == 0:
        raise LayoutError(f"Split '{name}' is empty.")
    _check_unique_images(records, name)
    _check_files_exist(records, name, dataset_dir, file_check)
    _warn_single_class(records, name)
    _warn_format_shortcut(records, name)
    return


def _check_unique_images(records: Sequence[SampleRecord], name: str) -> None:
    counts = Counter(record.image for record in records)
    duplicates = [image for image, count in counts.items() if count > 1]
    if duplicates:
        raise LayoutError(
            f"Split '{name}' lists {len(duplicates)} images more than once, e.g. {duplicates[:3]}."
        )
    return


def _check_files_exist(
    records: Sequence[SampleRecord], name: str, dataset_dir: Path, file_check: FileCheck
) -> None:
    match file_check:
        case FileCheck.NONE:
            return
        case FileCheck.SAMPLE if len(records) > FILE_CHECK_SAMPLE_SIZE:
            checked = random.Random(0).sample(list(records), FILE_CHECK_SAMPLE_SIZE)
        case _:
            checked = records
    missing = [
        record.image
        for record in track(checked, f"Checking files of {name}", transient=True)
        if not (dataset_dir / record.image).is_file()
    ]
    if missing:
        raise LayoutError(
            f"Split '{name}': {len(missing)} of {len(checked)} checked images are missing "
            f"or are broken links, e.g. {missing[:3]}."
        )
    return


def _warn_single_class(records: Sequence[SampleRecord], name: str) -> None:
    labels = {record.label for record in records}
    if labels != {REAL_LABEL, FAKE_LABEL}:
        logger.warning(f"Split '{name}' contains a single class: {sorted(labels)}.")
    return


def _warn_format_shortcut(records: Sequence[SampleRecord], name: str) -> None:
    real = _extension_shares(record for record in records if record.label == REAL_LABEL)
    fake = _extension_shares(record for record in records if record.label == FAKE_LABEL)
    if not real or not fake:
        return
    distance = 0.5 * sum(
        abs(real.get(ext, 0.0) - fake.get(ext, 0.0)) for ext in real | fake
    )
    if distance > FORMAT_SHORTCUT_DISTANCE:
        logger.warning(
            f"Split '{name}': file formats differ between classes "
            f"(real: {_format_shares(real)}; fake: {_format_shares(fake)}). "
            "A detector can separate the classes by format alone; "
            "re-encode both classes identically before trusting the metrics."
        )
    return


def _extension_shares(records: Iterable[SampleRecord]) -> dict[str, float]:
    counts = Counter(PurePosixPath(record.image).suffix.lower() for record in records)
    total = sum(counts.values())
    return {ext: count / total for ext, count in counts.items()}


def _format_shares(shares: dict[str, float]) -> str:
    return ", ".join(
        f"{ext or 'none'} {share:.0%}"
        for ext, share in sorted(shares.items(), key=lambda item: -item[1])
    )
