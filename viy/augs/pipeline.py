import hashlib
from collections.abc import Sequence
from typing import Any
from typing import assert_never
from typing import override

import albumentations as A
import numpy as np

from viy.common import ImageArray

from .base import OPS
from .base import ImageOp
from .configs import AlbumentationsStepConfig
from .configs import AugmentationConfig
from .configs import ChainStepConfig
from .configs import NTIREPreset
from .configs import NTIREPresetStepConfig
from .configs import OpStepConfig
from .configs import SequenceStepConfig
from .configs import StepConfig
from .presets import ntire_preset
from .steps import AlbumentationsStep
from .steps import AlbumentationsTransform
from .steps import ChainGroup
from .steps import ChainStep
from .steps import OpStep
from .steps import SequenceStep
from .steps import Step
from .steps import StepOutput


def image_rng(seed: int, image: str) -> np.random.Generator:
    """
    Generator for augmenting one image, independent of its position in a split.

    The same seed and image path always give the same augmentation, whatever
    subset of the split is drawn and in whatever order.

    Args:
        seed (int): User seed.
        image (str): Image path relative to the dataset directory.

    Returns:
        np.random.Generator: Generator seeded from ``seed`` and a stable hash of ``image``.
    """
    digest = hashlib.blake2b(image.encode(), digest_size=8).digest()
    return np.random.default_rng((seed, int.from_bytes(digest)))


def build_op(name: str, levels: Sequence[int | float | str] | None = None) -> ImageOp:
    """Instantiate the operation registered under ``name``."""
    return OPS.get(name)(levels)


def build_step(config: StepConfig) -> Step:
    """
    Instantiate a pipeline step from its configuration.

    Args:
        config (StepConfig): Step configuration.

    Returns:
        Step: Ready to use step.
    """
    match config:
        case OpStepConfig():
            return OpStep(config.op, config.p, config.levels)
        case ChainStepConfig():
            groups = [
                ChainGroup(
                    ops=[build_op(name, config.levels.get(name)) for name in group.ops],
                    weight=group.weight,
                )
                for group in config.groups
            ]
            return ChainStep(groups, config.lengths)
        case AlbumentationsStepConfig(serialized=dict() as serialized):
            return AlbumentationsStep(_deserialize(serialized), p=config.p)
        case AlbumentationsStepConfig(transform=str() as transform):
            return AlbumentationsStep(transform, config.params, config.p)
        case AlbumentationsStepConfig():
            raise ValueError("An albumentations step needs transform or serialized.")
        case NTIREPresetStepConfig():
            return NTIREPresetStep(config.preset, config.p, config.exclude_ops)
        case SequenceStepConfig():
            return SequenceStep([build_step(step) for step in config.steps], config.p)
        case _:
            assert_never(config)


def _deserialize(serialized: dict[str, Any]) -> AlbumentationsTransform:
    transform = A.from_dict({"transform": serialized})
    if not isinstance(transform, A.BasicTransform | A.BaseCompose):
        raise TypeError(f"Not an albumentations transform: {serialized}.")
    return transform


class NTIREPresetStep(Step):
    """A distortion mix reconstructed from NTIRE 2026, see :class:`NTIREPreset`."""

    def __init__(
        self,
        preset: NTIREPreset | str,
        p: float | None = None,
        exclude_ops: Sequence[str] = (),
    ) -> None:
        """
        Initialize the step.

        Args:
            preset (NTIREPreset | str): Preset name, e.g. ``aggressive``.
            p (float | None): Share of distorted images; ``None``
                keeps the preset value.
            exclude_ops (Sequence[str]): Operations removed from the preset.
        """
        names = [item.value for item in NTIREPreset]
        if preset not in names:
            raise ValueError(
                f"Unknown NTIRE preset '{preset}'. Available: {', '.join(names)}."
            )
        self.preset = NTIREPreset(preset)
        self.p = p
        self.exclude_ops = list(exclude_ops)
        self.sequence = build_step(ntire_preset(self.preset, p, exclude_ops))
        return

    @override
    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        return self.sequence(image, rng)

    @override
    def to_config(self) -> StepConfig:
        return NTIREPresetStepConfig(
            preset=self.preset, p=self.p, exclude_ops=self.exclude_ops
        )


class AugmentationPipeline:
    """
    Sequence of augmentation steps over ``uint8`` RGB images.

    The pipeline draws all randomness from the generator passed on every call,
    so the same seed reproduces both the image and the trace of applied
    operations.

    Example:
        >>> pipeline = AugmentationPipeline([
        ...     A.HorizontalFlip(p=0.5),
        ...     OpStep("jpeg", p=0.3),
        ...     NTIREPresetStep("aggressive"),
        ... ])
        >>> augmented, trace = pipeline(image, np.random.default_rng(0))
    """

    def __init__(self, steps: Sequence[Step | AlbumentationsTransform]) -> None:
        """
        Initialize the pipeline.

        Args:
            steps (Sequence[Step | AlbumentationsTransform]): Steps applied in order;
                albumentations transforms and composes are wrapped automatically.
        """
        self.steps = [
            step if isinstance(step, Step) else AlbumentationsStep(step)
            for step in steps
        ]
        return

    @classmethod
    def from_pydantic_model(cls, config: AugmentationConfig) -> "AugmentationPipeline":
        """Build the pipeline from its configuration."""
        return cls([build_step(step) for step in config.steps])

    def to_config(self) -> AugmentationConfig:
        """
        Configuration that builds an equivalent pipeline, e.g. for a benchmark derivation.

        Albumentations transforms are stored in their serialized form, so the
        rebuilt pipeline augments every image exactly as this one does.
        """
        return AugmentationConfig(steps=[step.to_config() for step in self.steps])

    def __call__(self, image: ImageArray, rng: np.random.Generator) -> StepOutput:
        """
        Augment ``image``.

        Args:
            image (ImageArray): RGB image.
            rng (np.random.Generator): Source of randomness.

        Returns:
            StepOutput: Augmented image and the applied operations in order.
        """
        trace = []
        for step in self.steps:
            image, applied = step(image, rng)
            trace.extend(applied)
        return image, trace
