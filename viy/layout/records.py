import csv
import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Final

from viy.common import FAKE_LABEL
from viy.common import LABELS
from viy.common import ScaleValue


CORE_COLUMNS: Final = (
    "image",
    "label",
    "generator",
    "distortions",
    "distortion_scales",
    "source_id",
)
REQUIRED_COLUMNS: Final = frozenset({"image", "label"})


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """
    One image of a dataset split in the canonical layout.

    ``None`` always means that a value is unknown. A clean image has
    ``distortions == ()``, while an image with an unknown processing history has
    ``distortions is None``.

    Attributes:
        image (str): POSIX path of the image relative to the dataset directory.
        label (int): ``0`` for real images, ``1`` for generated ones.
        generator (str | None): Name of the generator of a fake image.
        distortions (tuple[str, ...] | None): Applied distortions in application order.
        distortion_scales (tuple[ScaleValue, ...] | None): Severity of every distortion.
        source_id (str | None): Identifier shared by images derived from the same
            source, used to keep related images on one side of a split.
        extra (dict[str, str]): Dataset specific columns.
    """

    image: str
    label: int
    generator: str | None = None
    distortions: tuple[str, ...] | None = None
    distortion_scales: tuple[ScaleValue, ...] | None = None
    source_id: str | None = None
    extra: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Invalid label {self.label!r} for '{self.image}'.")
        path = PurePosixPath(self.image)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(
                f"Image path must be relative to the dataset directory: '{self.image}'."
            )
        if (self.distortions is None) != (self.distortion_scales is None):
            raise ValueError(
                f"'{self.image}': distortions and distortion_scales must be both known or both unknown."
            )
        if (
            self.distortions is not None
            and self.distortion_scales is not None
            and len(self.distortions) != len(self.distortion_scales)
        ):
            raise ValueError(
                f"'{self.image}': {len(self.distortions)} distortions but {len(self.distortion_scales)} scales."
            )
        return

    @property
    def is_fake(self) -> bool:
        """Whether the image is generated."""
        return self.label == FAKE_LABEL

    @property
    def is_distorted(self) -> bool | None:
        """Whether any distortion was applied, or ``None`` if unknown."""
        if self.distortions is None:
            return None
        return len(self.distortions) > 0


def write_records(path: Path, records: Sequence[SampleRecord]) -> None:
    """
    Atomically write records to a split CSV.

    Lists are stored as JSON strings and unknown values as empty cells.

    Args:
        path (Path): Destination CSV file.
        records (Sequence[SampleRecord]): Records to write.
    """
    extra_columns = sorted({key for record in records for key in record.extra})
    collisions = set(extra_columns) & set(CORE_COLUMNS)
    if collisions:
        raise ValueError(f"Extra columns shadow core columns: {sorted(collisions)}.")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*CORE_COLUMNS, *extra_columns])
        for record in records:
            writer.writerow(
                [
                    record.image,
                    str(record.label),
                    _encode_optional(record.generator),
                    _encode_list(record.distortions),
                    _encode_list(record.distortion_scales),
                    _encode_optional(record.source_id),
                    *(record.extra.get(column, "") for column in extra_columns),
                ]
            )
    temporary.replace(path)
    return


def read_records(path: Path) -> list[SampleRecord]:
    """
    Read a split CSV.

    Only ``image`` and ``label`` columns are required, so a hand-written CSV
    with these two columns is a valid split.

    Args:
        path (Path): Split CSV file.

    Returns:
        list[SampleRecord]: Records in file order.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}.")
        extra_columns = [column for column in columns if column not in CORE_COLUMNS]
        return [
            _decode_row(row, extra_columns, f"{path}, line {line}")
            for line, row in enumerate(reader, start=2)
        ]


def _decode_row(
    row: Mapping[str, str | None], extra_columns: Sequence[str], where: str
) -> SampleRecord:
    distortions = _decode_list(row.get("distortions"), "distortions", where)
    scales = _decode_list(row.get("distortion_scales"), "distortion_scales", where)
    extra = {column: value for column in extra_columns if (value := row.get(column))}
    image = _require(row, "image", where)
    label = _require(row, "label", where)
    if label not in {"0", "1"}:
        raise ValueError(
            f"{where}: the label of '{image}' must be 0 for a real image or 1 for a "
            f"generated one, got {label!r}."
        )
    return SampleRecord(
        image=image,
        label=int(label),
        generator=row.get("generator") or None,
        distortions=None
        if distortions is None
        else tuple(str(item) for item in distortions),
        distortion_scales=None
        if scales is None
        else tuple(_as_scale(item, where) for item in scales),
        source_id=row.get("source_id") or None,
        extra=extra,
    )


def _require(row: Mapping[str, str | None], column: str, where: str) -> str:
    value = row.get(column)
    if not value:
        raise ValueError(f"{where}: the required column '{column}' is empty.")
    return value


def _encode_optional(value: str | None) -> str:
    return "" if value is None else value


def _encode_list(values: Sequence[ScaleValue] | None) -> str:
    return "" if values is None else json.dumps(list(values))


def _decode_list(cell: str | None, column: str, where: str) -> list[object] | None:
    if not cell:
        return None
    if not cell.lstrip().startswith("["):
        raise ValueError(
            f'{where}: {column} must be a JSON list such as ["jpeg"] or [], got {cell!r}.'
        )
    decoded = json.loads(cell)
    if not isinstance(decoded, list):
        raise TypeError(f"{where}: {column} must be a JSON list, got {cell!r}.")
    return decoded


def _as_scale(value: object, where: str) -> ScaleValue:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"{where}: invalid distortion scale {value!r}.")
    return value
