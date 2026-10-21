from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import override

import albumentations as A
import numpy as np

from viy.common import ImageArray

from .base import AppliedOp
from .base import ImageOp


type StepOutput = tuple[ImageArray, list[AppliedOp]]


def _check_probability(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {p}.")
    return


class Step(ABC):
    """A unit of an augmentation pipeline that reports the operations it applied."""

    @abstractmethod
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        """
        Augment ``image``.

        Args:
            image (ImageArray): RGB image.
            rng (np.random.Generator): Source of randomness.

        Returns:
            StepOutput: Augmented image and the applied operations in order.
        """
        raise NotImplementedError


class OpStep(Step):
    """Apply one image operation with probability ``p``."""

    def __init__(self, op: ImageOp, p: float = 1.0) -> None:
        """
        Initialize the step.

        Args:
            op (ImageOp): Operation to apply.
            p (float): Probability of applying it.
        """
        _check_probability(p)
        self.op = op
        self.p = p
        return

    @override
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        if rng.random() >= self.p:
            return image, []
        image, applied = self.op(image, rng)
        return image, [applied]


class SequenceStep(Step):
    """Apply nested steps in order, all of them with probability ``p`` or none."""

    def __init__(self, steps: Sequence[Step], p: float = 1.0) -> None:
        """
        Initialize the step.

        Args:
            steps (Sequence[Step]): Steps to apply in order.
            p (float): Probability of applying the whole sequence.
        """
        _check_probability(p)
        self.steps = list(steps)
        self.p = p
        return

    @override
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        if rng.random() >= self.p:
            return image, []
        trace: list[AppliedOp] = []
        for step in self.steps:
            image, applied = step(image, rng)
            trace.extend(applied)
        return image, trace


@dataclass(frozen=True, slots=True)
class ChainGroup:
    """
    Mutually exclusive operations of a chain, such as all compression codecs.

    Attributes:
        ops (tuple[ImageOp, ...]): Operations of the group, drawn uniformly.
        weight (float): Relative probability of drawing the group.
    """

    ops: tuple[ImageOp, ...]
    weight: float = 1.0


class ChainStep(Step):
    """
    Random chain of operations from distinct groups in random order.

    The chain length ``k`` is drawn from ``lengths``, then ``k`` distinct groups
    are drawn by weight and one operation is drawn uniformly from each group.
    This reproduces the distortion pipeline behind the NTIRE 2026 splits.
    """

    def __init__(
        self, groups: Sequence[ChainGroup], lengths: Mapping[int, float]
    ) -> None:
        """
        Initialize the step.

        Args:
            groups (Sequence[ChainGroup]): Groups of mutually exclusive operations.
            lengths (Mapping[int, float]): Relative probability of every chain length.
        """
        if len(groups) == 0 or any(len(group.ops) == 0 for group in groups):
            raise ValueError("A chain needs at least one non-empty group.")
        if not lengths or not all(1 <= length <= len(groups) for length in lengths):
            raise ValueError(
                f"Chain lengths must be in [1, {len(groups)}], got {sorted(lengths)}."
            )
        self.groups = list(groups)
        self._lengths = np.array(sorted(lengths), dtype=np.int64)
        self._length_p = _normalize([lengths[int(length)] for length in self._lengths])
        self._group_p = _normalize([group.weight for group in self.groups])
        return

    @override
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        length = int(rng.choice(self._lengths, p=self._length_p))
        chosen = rng.choice(
            len(self.groups), size=length, replace=False, p=self._group_p
        )
        trace: list[AppliedOp] = []
        for group_index in chosen:
            group = self.groups[int(group_index)]
            op = group.ops[int(rng.integers(len(group.ops)))]
            image, applied = op(image, rng)
            trace.append(applied)
        return image, trace


class AlbumentationsStep(Step):
    """
    Apply an albumentations transform with probability ``p``.

    The transform is reseeded from the pipeline generator on every call, so it
    is as reproducible as the native operations. It must return ``uint8``
    images; normalization belongs to preprocessing.
    """

    def __init__(
        self, transform: str, params: Mapping[str, Any] | None = None, p: float = 1.0
    ) -> None:
        """
        Initialize the step.

        Args:
            transform (str): Name of an ``albumentations`` transform class, e.g. ``HorizontalFlip``.
            params (Mapping[str, Any] | None): Keyword arguments of the transform except ``p``.
            p (float): Probability of applying the transform.
        """
        _check_probability(p)
        params = dict(params or {})
        if "p" in params:
            raise ValueError(
                "Set the probability of an albumentations step with the step 'p'."
            )
        transform_cls = getattr(A, transform, None)
        if not isinstance(transform_cls, type) or not issubclass(
            transform_cls, A.BasicTransform
        ):
            raise TypeError(f"'{transform}' is not an albumentations transform.")
        self.name = f"albumentations.{transform}"
        self.p = p
        self._compose = A.Compose([transform_cls(**params, p=1.0)])
        return

    @override
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        if rng.random() >= self.p:
            return image, []
        self._compose.set_random_seed(int(rng.integers(2**32)))
        output = self._compose(image=image)["image"]
        if not isinstance(output, np.ndarray) or output.dtype != np.uint8:
            raise TypeError(f"{self.name} must return a uint8 image.")
        return np.ascontiguousarray(output), [AppliedOp(self.name)]


def _normalize(
    weights: Sequence[float],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    array = np.asarray(weights, dtype=np.float64)
    if np.any(array <= 0):
        raise ValueError(f"Weights must be positive, got {list(weights)}.")
    return array / array.sum()
