from collections.abc import Sequence
from typing import assert_never

import numpy as np

from viy.common import ImageArray

from .base import OPS
from .base import ImageOp
from .configs import AlbumentationsStepConfig
from .configs import AugmentationConfig
from .configs import ChainStepConfig
from .configs import NTIREPresetStepConfig
from .configs import OpStepConfig
from .configs import SequenceStepConfig
from .configs import StepConfig
from .presets import ntire_preset
from .steps import AlbumentationsStep
from .steps import ChainGroup
from .steps import ChainStep
from .steps import OpStep
from .steps import SequenceStep
from .steps import Step
from .steps import StepOutput


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
            return OpStep(build_op(config.op, config.levels), config.p)
        case ChainStepConfig():
            groups = [
                ChainGroup(
                    ops=tuple(
                        build_op(name, config.levels.get(name)) for name in group.ops
                    ),
                    weight=group.weight,
                )
                for group in config.groups
            ]
            return ChainStep(groups, config.lengths)
        case AlbumentationsStepConfig():
            return AlbumentationsStep(config.transform, config.params, config.p)
        case NTIREPresetStepConfig():
            return build_step(
                ntire_preset(
                    config.preset, config.distort_probability, config.exclude_ops
                )
            )
        case SequenceStepConfig():
            return SequenceStep([build_step(step) for step in config.steps], config.p)
        case _:
            assert_never(config)


class AugmentationPipeline:
    """
    Sequence of augmentation steps over ``uint8`` RGB images.

    The pipeline draws all randomness from the generator passed on every call,
    so the same seed reproduces both the image and the trace of applied
    operations.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        """
        Initialize the pipeline.

        Args:
            steps (Sequence[Step]): Steps applied in order.
        """
        self.steps = list(steps)
        return

    @classmethod
    def from_pydantic_model(cls, config: AugmentationConfig) -> "AugmentationPipeline":
        """Build the pipeline from its configuration."""
        return cls([build_step(step) for step in config.steps])

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
