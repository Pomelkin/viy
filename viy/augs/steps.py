from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import override

import albumentations as A
import numpy as np

from viy.common import ImageArray
from viy.common import ScaleValue

from .base import OPS
from .base import AppliedOp
from .base import ImageOp
from .configs import AlbumentationsStepConfig
from .configs import ChainGroupConfig
from .configs import ChainStepConfig
from .configs import OpStepConfig
from .configs import SequenceStepConfig
from .configs import StepConfig


type StepOutput = tuple[ImageArray, list[AppliedOp]]
type AlbumentationsTransform = A.BasicTransform | A.BaseCompose


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

    def to_config(self) -> StepConfig:
        """
        Configuration that builds an equivalent step.

        Raises:
            TypeError: If the step has no configuration form, e.g. a custom subclass.
        """
        raise TypeError(
            f"{type(self).__name__} has no config form; compose the pipeline from the steps of viy.augs."
        )


class OpStep(Step):
    """Apply one image operation with probability ``p``."""

    def __init__(
        self,
        op: ImageOp | str,
        p: float = 1.0,
        levels: Sequence[ScaleValue] | None = None,
    ) -> None:
        """
        Initialize the step.

        Args:
            op (ImageOp | str): Operation, or the registered name of one, e.g. ``jpeg``.
            p (float): Probability of applying it.
            levels (Sequence[ScaleValue] | None): Severity levels when ``op`` is a
                name; ``None`` uses the operation defaults.
        """
        _check_probability(p)
        if isinstance(op, str):
            op = OPS.get(op)(levels)
        elif levels is not None:
            raise ValueError(
                "Pass levels to the operation itself when it is not given by name."
            )
        self.op = op
        self.p = p
        return

    @override
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        if rng.random() >= self.p:
            return image, []
        image, applied = self.op(image, rng)
        return image, [applied]

    @override
    def to_config(self) -> StepConfig:
        return OpStepConfig(op=self.op.name, p=self.p, levels=list(self.op.levels))


class SequenceStep(Step):
    """Apply nested steps in order, all of them with probability ``p`` or none."""

    def __init__(
        self, steps: Sequence[Step | AlbumentationsTransform], p: float = 1.0
    ) -> None:
        """
        Initialize the step.

        Args:
            steps (Sequence[Step | AlbumentationsTransform]): Steps to apply in
                order; albumentations transforms are wrapped automatically.
            p (float): Probability of applying the whole sequence.
        """
        _check_probability(p)
        self.steps = [
            step if isinstance(step, Step) else AlbumentationsStep(step)
            for step in steps
        ]
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

    @override
    def to_config(self) -> StepConfig:
        return SequenceStepConfig(
            steps=[step.to_config() for step in self.steps], p=self.p
        )


class ChainGroup:
    """Mutually exclusive operations of a chain, such as all compression codecs."""

    def __init__(self, ops: Sequence[ImageOp | str], weight: float = 1.0) -> None:
        """
        Initialize the group.

        Args:
            ops (Sequence[ImageOp | str]): Operations of the group, drawn uniformly,
                or registered names of operations with their default levels.
            weight (float): Relative probability of drawing the group.
        """
        self.ops: tuple[ImageOp, ...] = tuple(
            OPS.get(op)() if isinstance(op, str) else op for op in ops
        )
        self.weight = weight
        return


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
        self.lengths = dict(lengths)
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

    @override
    def to_config(self) -> StepConfig:
        return ChainStepConfig(
            groups=[
                ChainGroupConfig(ops=[op.name for op in group.ops], weight=group.weight)
                for group in self.groups
            ],
            lengths=self.lengths,
            levels={
                op.name: list(op.levels) for group in self.groups for op in group.ops
            },
        )


class AlbumentationsStep(Step):
    """
    Apply an albumentations transform with probability ``p``.

    The transform is given either as a ready object, such as
    ``A.HorizontalFlip(p=0.5)`` or an ``A.Compose``, or by class name with
    keyword ``params``. It is reseeded from the pipeline generator on every
    call, so it is as reproducible as the native operations, and every
    transform it actually applies is recorded in the trace. It must return
    ``uint8`` images; normalization belongs to preprocessing.
    """

    def __init__(
        self,
        transform: AlbumentationsTransform | str,
        params: Mapping[str, Any] | None = None,
        p: float = 1.0,
    ) -> None:
        """
        Initialize the step.

        Args:
            transform (AlbumentationsTransform | str): Albumentations transform or
                compose, or the name of a transform class, e.g. ``HorizontalFlip``.
            params (Mapping[str, Any] | None): Keyword arguments of a transform
                given by name, except ``p``.
            p (float): Probability of applying the step.
        """
        _check_probability(p)
        self._by_name: tuple[str, dict[str, Any]] | None = None
        if isinstance(transform, str):
            self._by_name = (transform, dict(params or {}))
            transform = _transform_by_name(transform, dict(params or {}))
        elif params:
            raise ValueError("Pass params only together with a transform name.")
        self.transform = transform
        self.name = f"albumentations.{type(transform).__name__}"
        self.p = p
        self._compose = A.Compose([transform], save_applied_params=True)
        return

    @override
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        if rng.random() >= self.p:
            return image, []
        self._compose.set_random_seed(int(rng.integers(2**32)))
        output = self._compose(image=image)
        augmented = output["image"]
        if not isinstance(augmented, np.ndarray) or augmented.dtype != np.uint8:
            raise TypeError(f"{self.name} must return a uint8 image.")
        trace = [
            AppliedOp(f"albumentations.{name}")
            for name, _ in output["applied_transforms"]
        ]
        return np.ascontiguousarray(augmented), trace

    @override
    def to_config(self) -> StepConfig:
        if self._by_name is not None:
            name, params = self._by_name
            return AlbumentationsStepConfig(transform=name, params=params, p=self.p)
        serialized = A.to_dict(self.transform)["transform"]
        return AlbumentationsStepConfig(serialized=serialized, p=self.p)


def _transform_by_name(name: str, params: dict[str, Any]) -> A.BasicTransform:
    if "p" in params:
        raise ValueError(
            "Set the probability of an albumentations step with the step 'p'."
        )
    transform_cls = getattr(A, name, None)
    if not isinstance(transform_cls, type) or not issubclass(
        transform_cls, A.BasicTransform
    ):
        raise TypeError(f"'{name}' is not an albumentations transform.")
    return transform_cls(**params, p=1.0)


def _normalize(
    weights: Sequence[float],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    array = np.asarray(weights, dtype=np.float64)
    if np.any(array <= 0):
        raise ValueError(f"Weights must be positive, got {list(weights)}.")
    return array / array.sum()
