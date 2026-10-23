from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from .configs import ChainGroupConfig
from .configs import ChainStepConfig
from .configs import NTIREPreset
from .configs import OpStepConfig
from .configs import SequenceStepConfig


CHAIN_LENGTHS: Final = MappingProxyType({1: 1.0, 2: 1.0, 3: 1.0})
DOWNSCALE_PROBABILITY: Final = 0.75
SPLIT_DISTORT_PROBABILITY: Final = 0.5
TRAIN_DISTORT_PROBABILITY: Final = 0.9


@dataclass(frozen=True, slots=True)
class PresetSpec:
    """
    A distortion mix: a chain of grouped operations followed by watermarking and downscaling.

    Attributes:
        groups (tuple[tuple[str, ...], ...]): Groups of mutually exclusive operations.
        distort_probability (float): Share of distorted images.
        watermark_probability (float): Probability of a watermark after the chain.
        watermarks (tuple[str, ...]): Watermark methods drawn uniformly.
        excluded (frozenset[str]): Operations removed from the mix.
    """

    groups: tuple[tuple[str, ...], ...]
    distort_probability: float
    watermark_probability: float = 0.0
    watermarks: tuple[str, ...] = ()
    excluded: frozenset[str] = frozenset()


VAL_GROUPS: Final = (
    ("jpeg",),
    ("lincontrchange",),
    ("brighten", "darken"),
    ("randomaspectcrop", "randomcrop"),
    ("colorsat", "colorshift", "rgbshift"),
    ("gausblur", "lensblur", "motionblur"),
    ("impulsenoise", "multnoise", "whitenoise"),
    ("jitter", "pixelate", "quantization"),
)
VAL_HARD_GROUPS: Final = (
    ("quantization",),
    ("jpeg",),
    ("jpeg_ai_proxy",),
    ("adv_embed_clip_proxy", "adv_embed_resnet_proxy"),
    ("gausblur", "lensblur"),
    ("clahe", "randomtonecurve", "rgbshift"),
    (
        "jpeg_recompression_1",
        "jpeg_recompression_2",
        "jpeg_recompression_comb_jpegai_proxy",
    ),
    ("perspective", "randomaspectcrop", "randomcrop"),
    ("impulsenoise", "isonoise", "multnoise", "whitenoise"),
)
TEST_PUBLIC_GROUPS: Final = (
    ("brighten",),
    ("lensblur",),
    ("jpeg", "jpeg2000"),
    ("cheng2020_proxy", "jpeg_ai_proxy"),
    (
        "jpeg_recompression_1",
        "jpeg_recompression_2",
        "jpeg_recompression_comb_jpegai_proxy",
    ),
    ("adv_embed_clip_proxy", "adv_embed_resnet_proxy", "wmforger_proxy"),
    ("perspective", "randomaspectcrop", "randomcrop"),
    ("impulsenoise", "isonoise", "shotnoise"),
    ("clahe", "colorsat", "randomtonecurve", "rgbshift"),
)
TEST_PRIVATE_GROUPS: Final = (
    ("brighten",),
    ("cheng2020_proxy", "jpeg_ai_proxy"),
    ("glassblur", "lensblur"),
    ("jpeg", "jpeg2000"),
    ("adv_embed_clip_proxy", "adv_embed_resnet_proxy", "wmforger_proxy"),
    ("impulsenoise", "isonoise", "shotnoise"),
    ("perspective", "randomaspectcrop", "randomcrop"),
    (
        "jpeg_recompression_1",
        "jpeg_recompression_2",
        "jpeg_recompression_comb",
        "jpeg_recompression_comb_jpegai_proxy",
    ),
    ("clahe", "colorsat", "randomtonecurve", "rgbshift"),
)
TEST_PUBLIC_WATERMARKS: Final = (
    "arwgan",
    "dct_marker",
    "mbrs",
    "pimog",
    "pixelseal",
    "stega_stamp",
)
TEST_PRIVATE_WATERMARKS: Final = (
    "arwgan",
    "fin",
    "maskwm",
    "mbrs",
    "sshidden",
    "stega_stamp",
    "syncseal",
)
STAGE6_EXCLUDED: Final = frozenset(
    {"jpeg_ai_proxy", "jpeg_recompression_comb_jpegai_proxy", "wmforger_proxy"}
)

PRESETS: Final = MappingProxyType(
    {
        NTIREPreset.VAL: PresetSpec(VAL_GROUPS, SPLIT_DISTORT_PROBABILITY),
        NTIREPreset.VAL_HARD: PresetSpec(VAL_HARD_GROUPS, SPLIT_DISTORT_PROBABILITY),
        NTIREPreset.TEST_PUBLIC: PresetSpec(
            TEST_PUBLIC_GROUPS, SPLIT_DISTORT_PROBABILITY, 0.36, TEST_PUBLIC_WATERMARKS
        ),
        NTIREPreset.TEST_PRIVATE: PresetSpec(
            TEST_PRIVATE_GROUPS,
            SPLIT_DISTORT_PROBABILITY,
            0.37,
            TEST_PRIVATE_WATERMARKS,
        ),
        NTIREPreset.AGGRESSIVE: PresetSpec(
            TEST_PUBLIC_GROUPS, TRAIN_DISTORT_PROBABILITY, 0.36, TEST_PUBLIC_WATERMARKS
        ),
        NTIREPreset.AGGRESSIVE_STAGE6: PresetSpec(
            TEST_PUBLIC_GROUPS,
            TRAIN_DISTORT_PROBABILITY,
            0.36,
            TEST_PUBLIC_WATERMARKS,
            excluded=STAGE6_EXCLUDED,
        ),
    }
)


def preset_operations(preset: NTIREPreset) -> list[str]:
    """Names of all operations a preset can apply."""
    spec = PRESETS[preset]
    names = [name for group in spec.groups for name in group]
    if spec.watermark_probability > 0:
        names.append("watermark_proxy")
    names.append("downscale")
    return [name for name in names if name not in spec.excluded]


def ntire_preset(
    preset: NTIREPreset,
    p: float | None = None,
    exclude_ops: Sequence[str] = (),
) -> SequenceStepConfig:
    """
    Expand a preset into an explicit step configuration.

    With probability ``p`` an image goes through a chain of
    one to three operations from distinct groups in random order, then an
    optional watermark, then an optional downscale. Groups left empty by the
    exclusions are dropped.

    Args:
        preset (NTIREPreset): Preset name.
        p (float | None): Share of distorted images; ``None`` keeps the preset value.
        exclude_ops (Sequence[str]): Operations to remove from the preset.

    Returns:
        SequenceStepConfig: Equivalent explicit configuration.
    """
    spec = PRESETS[preset]
    unknown = set(exclude_ops) - set(preset_operations(preset))
    if unknown:
        raise ValueError(f"Preset '{preset}' has no operations {sorted(unknown)}.")
    excluded = spec.excluded | set(exclude_ops)
    groups = [
        ChainGroupConfig(ops=kept)
        for group in spec.groups
        if (kept := [name for name in group if name not in excluded])
    ]
    steps: list[OpStepConfig | ChainStepConfig] = [
        ChainStepConfig(
            groups=groups,
            lengths={
                length: weight
                for length, weight in CHAIN_LENGTHS.items()
                if length <= len(groups)
            },
        )
    ]
    if spec.watermark_probability > 0 and "watermark_proxy" not in excluded:
        steps.append(
            OpStepConfig(
                op="watermark_proxy",
                p=spec.watermark_probability,
                levels=list(spec.watermarks),
            )
        )
    if "downscale" not in excluded:
        steps.append(OpStepConfig(op="downscale", p=DOWNSCALE_PROBABILITY))
    probability = spec.distort_probability if p is None else p
    return SequenceStepConfig(steps=list(steps), p=probability)
