from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from viy.common import REAL_LABEL
from viy.layout import SampleRecord


MAX_LEVELS_PER_DISTORTION: Final = 10


@dataclass(frozen=True, slots=True)
class Slice:
    """
    A named subset of a split.

    Attributes:
        name (str): ``all``, ``clean``, ``distorted``, ``distortion:<op>``,
            ``distortion:<op>@<level>`` or ``generator:<name>``.
        indices (npt.NDArray[np.int64]): Rows of the split in the subset.
    """

    name: str
    indices: npt.NDArray[np.int64]


def build_slices(records: Sequence[SampleRecord]) -> list[Slice]:
    """
    Subsets metrics are reported on, depending on the metadata the split carries.

    Distortion slices exist when every image has a known distortion chain and
    contain every image the distortion was applied to; levels are reported for
    distortions with few discrete levels. A generator slice holds the fakes of
    one generator together with all real images.

    Args:
        records (Sequence[SampleRecord]): Records of the split.

    Returns:
        list[Slice]: Slices starting with ``all``.
    """
    slices = [Slice("all", np.arange(len(records), dtype=np.int64))]
    if records and all(record.distortions is not None for record in records):
        slices.extend(_distortion_slices(records))
    slices.extend(_generator_slices(records))
    return slices


def _distortion_slices(records: Sequence[SampleRecord]) -> list[Slice]:
    by_op: dict[str, list[int]] = defaultdict(list)
    by_level: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    clean: list[int] = []
    for index, record in enumerate(records):
        names = record.distortions or ()
        scales = record.distortion_scales or ()
        if not names:
            clean.append(index)
        for name, scale in zip(names, scales, strict=True):
            by_op[name].append(index)
            by_level[name][str(scale)].append(index)
    distorted = [index for index in range(len(records)) if records[index].distortions]
    slices = [_slice("clean", clean), _slice("distorted", distorted)]
    for name in sorted(by_op):
        slices.append(_slice(f"distortion:{name}", by_op[name]))
        levels = by_level[name]
        if len(levels) <= MAX_LEVELS_PER_DISTORTION:
            slices.extend(
                _slice(f"distortion:{name}@{level}", levels[level])
                for level in sorted(levels)
            )
    return [item for item in slices if item.indices.size]


def _generator_slices(records: Sequence[SampleRecord]) -> list[Slice]:
    reals = [
        index for index, record in enumerate(records) if record.label == REAL_LABEL
    ]
    by_generator: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.generator is not None and record.label != REAL_LABEL:
            by_generator[record.generator].append(index)
    return [
        _slice(f"generator:{name}", sorted(reals + by_generator[name]))
        for name in sorted(by_generator)
    ]


def _slice(name: str, indices: Sequence[int]) -> Slice:
    return Slice(name, np.asarray(indices, dtype=np.int64))
