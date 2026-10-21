from enum import StrEnum
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self

from kostyl import StrictPydanticBaseModel
from pydantic import Field
from pydantic import model_validator

from viy.common import ScaleValue


type Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class NTIREPreset(StrEnum):
    """
    Distortion mixes reconstructed from the NTIRE 2026 split metadata.

    ``val``, ``val_hard``, ``test_public`` and ``test_private`` reproduce the
    groups, chain lengths and tail probabilities of the respective splits.
    ``aggressive`` is the public test mix applied to 90% of the images, and
    ``aggressive_stage6`` drops its JPEG AI and WMForger proxies for training
    with genuine examples from the local ``shard_6``.

    Learned codecs, adversarial attacks and watermarks are represented by
    proxies, which log a warning when created; drop them with ``exclude_ops``.
    """

    VAL = "val"
    VAL_HARD = "val_hard"
    TEST_PUBLIC = "test_public"
    TEST_PRIVATE = "test_private"
    AGGRESSIVE = "aggressive"
    AGGRESSIVE_STAGE6 = "aggressive_stage6"


class OpStepConfig(StrictPydanticBaseModel):
    """
    One registered image operation.

    Attributes:
        op (str): Registered operation name, e.g. ``jpeg``.
        p (float): Probability of applying the operation.
        levels (list[ScaleValue] | None): Severity levels; ``None`` uses the operation defaults.
    """

    kind: Literal["op"] = "op"
    op: str
    p: Probability = 1.0
    levels: list[ScaleValue] | None = None


class ChainGroupConfig(StrictPydanticBaseModel):
    """
    Mutually exclusive operations of a chain.

    Attributes:
        ops (list[str]): Registered operation names, drawn uniformly.
        weight (float): Relative probability of drawing the group.
    """

    ops: list[str] = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0.0)


class ChainStepConfig(StrictPydanticBaseModel):
    """
    Random chain of operations from distinct groups in random order.

    Attributes:
        groups (list[ChainGroupConfig]): Groups of mutually exclusive operations.
        lengths (dict[int, float]): Relative probability of every chain length.
        levels (dict[str, list[ScaleValue]]): Severity overrides per operation name.
    """

    kind: Literal["chain"] = "chain"
    groups: list[ChainGroupConfig] = Field(min_length=1)
    lengths: dict[int, float] = Field(min_length=1)
    levels: dict[str, list[ScaleValue]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_chain(self) -> Self:
        names = [name for group in self.groups for name in group.ops]
        if len(names) != len(set(names)):
            raise ValueError(f"An operation appears in several groups: {names}.")
        invalid_lengths = [
            length for length in self.lengths if not 1 <= length <= len(self.groups)
        ]
        if invalid_lengths:
            raise ValueError(
                f"Chain lengths {invalid_lengths} are outside [1, {len(self.groups)}]."
            )
        if any(weight <= 0 for weight in self.lengths.values()):
            raise ValueError("Chain length weights must be positive.")
        unknown = set(self.levels) - set(names)
        if unknown:
            raise ValueError(
                f"Level overrides for operations outside the chain: {sorted(unknown)}."
            )
        return self


class AlbumentationsStepConfig(StrictPydanticBaseModel):
    """
    An ``albumentations`` transform.

    Attributes:
        transform (str): Transform class name, e.g. ``HorizontalFlip``.
        params (dict[str, Any]): Keyword arguments of the transform except ``p``.
        p (float): Probability of applying the transform.
    """

    kind: Literal["albumentations"] = "albumentations"
    transform: str
    params: dict[str, Any] = Field(default_factory=dict)
    p: Probability = 1.0


class NTIREPresetStepConfig(StrictPydanticBaseModel):
    """
    A distortion mix reconstructed from NTIRE 2026.

    Attributes:
        preset (NTIREPreset): Preset name.
        distort_probability (float | None): Share of distorted images; ``None`` keeps the preset value.
        exclude_ops (list[str]): Operations removed from the preset, e.g. held-out transforms.
    """

    kind: Literal["ntire_preset"] = "ntire_preset"
    preset: NTIREPreset
    distort_probability: Probability | None = None
    exclude_ops: list[str] = Field(default_factory=list)


class SequenceStepConfig(StrictPydanticBaseModel):
    """
    Nested steps applied in order, all of them with probability ``p`` or none.

    Attributes:
        steps (list[StepConfig]): Steps to apply in order.
        p (float): Probability of applying the whole sequence.
    """

    kind: Literal["sequence"] = "sequence"
    steps: list["StepConfig"] = Field(min_length=1)
    p: Probability = 1.0


type StepConfig = Annotated[
    OpStepConfig
    | ChainStepConfig
    | AlbumentationsStepConfig
    | NTIREPresetStepConfig
    | SequenceStepConfig,
    Field(discriminator="kind"),
]


class AugmentationConfig(StrictPydanticBaseModel):
    """
    Augmentation pipeline applied to ``uint8`` RGB images before preprocessing.

    Attributes:
        steps (list[StepConfig]): Steps applied in order.
    """

    steps: list[StepConfig] = Field(min_length=1)


SequenceStepConfig.model_rebuild()
AugmentationConfig.model_rebuild()
