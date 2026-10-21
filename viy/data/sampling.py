from collections.abc import Sequence

import numpy as np

from viy.layout import SampleRecord


def subsample(
    records: Sequence[SampleRecord], limit: int | None, seed: int
) -> list[SampleRecord]:
    """
    Draw at most ``limit`` records keeping the class ratio and the original order.

    Args:
        records (Sequence[SampleRecord]): Records of a split.
        limit (int | None): Maximum number of records; ``None`` keeps all.
        seed (int): Seed of the draw.

    Returns:
        list[SampleRecord]: Selected records.
    """
    if limit is None or limit >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in sorted({record.label for record in records}):
        indices = [
            index for index, record in enumerate(records) if record.label == label
        ]
        quota = round(limit * len(indices) / len(records))
        selected.extend(int(index) for index in rng.permutation(indices)[:quota])
    return [records[index] for index in sorted(selected)]
