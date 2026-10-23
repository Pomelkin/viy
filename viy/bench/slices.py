import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from viy.augs import trace_from_json
from viy.common import REAL_LABEL
from viy.common import ScaleValue
from viy.data import AUGMENTATIONS_COLUMN
from viy.layout import SampleRecord


MAX_LEVELS_PER_DISTORTION: Final = 10
_NUMBER: Final = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")

type OpChain = tuple[tuple[str, ScaleValue | None], ...]


@dataclass(frozen=True, slots=True)
class Slice:
    """
    A named subset of a split.

    Attributes:
        name (str): ``all``, ``clean``, ``distorted``, ``distortion:<op>``,
            ``distortion:<op>@<level>``, ``not_augmented``, ``augmented``,
            ``augmentation:<op>``, ``augmentation:<op>@<level>`` or ``generator:<name>``.
        indices (npt.NDArray[np.int64]): Rows of the split in the subset.
    """

    name: str
    indices: npt.NDArray[np.int64]


def build_slices(records: Sequence[SampleRecord]) -> list[Slice]:
    """
    Subsets metrics are reported on, depending on the metadata the split carries.

    Distortion slices exist when every image has a known distortion chain and
    contain every image the distortion was applied to; levels are reported for
    distortions with few discrete levels. Augmentation slices are built the
    same way from the ``augmentations`` column of a derived split. A generator
    slice holds the fakes of one generator together with all real images.

    Args:
        records (Sequence[SampleRecord]): Records of the split.

    Returns:
        list[Slice]: Slices starting with ``all``.
    """
    slices = [Slice("all", np.arange(len(records), dtype=np.int64))]
    if records and all(record.distortions is not None for record in records):
        chains = [
            tuple(
                zip(
                    record.distortions or (),
                    record.distortion_scales or (),
                    strict=True,
                )
            )
            for record in records
        ]
        slices.extend(_chain_slices(chains, "distortion", "clean", "distorted"))
    if records and all(AUGMENTATIONS_COLUMN in record.extra for record in records):
        chains = [
            tuple(
                (op.name, op.scale)
                for op in trace_from_json(record.extra[AUGMENTATIONS_COLUMN])
            )
            for record in records
        ]
        slices.extend(
            _chain_slices(chains, "augmentation", "not_augmented", "augmented")
        )
    slices.extend(_generator_slices(records))
    return slices


def _chain_slices(
    chains: Sequence[OpChain], prefix: str, untouched: str, touched: str
) -> list[Slice]:
    by_op: dict[str, set[int]] = defaultdict(set)
    by_level: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for index, chain in enumerate(chains):
        for name, scale in chain:
            by_op[name].add(index)
            if scale is not None:
                by_level[name][str(scale)].add(index)
    slices = [
        _slice(untouched, [index for index, chain in enumerate(chains) if not chain]),
        _slice(touched, [index for index, chain in enumerate(chains) if chain]),
    ]
    for name in sorted(by_op):
        slices.append(_slice(f"{prefix}:{name}", sorted(by_op[name])))
        levels = by_level[name]
        if len(levels) <= MAX_LEVELS_PER_DISTORTION:
            slices.extend(
                _slice(f"{prefix}:{name}@{level}", sorted(levels[level]))
                for level in sorted(levels, key=_level_order)
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


def _level_order(level: str) -> tuple[int, float, str]:
    if _NUMBER.fullmatch(level):
        return 0, float(level), level
    return 1, 0.0, level


def _slice(name: str, indices: Sequence[int]) -> Slice:
    return Slice(name, np.asarray(indices, dtype=np.int64))
