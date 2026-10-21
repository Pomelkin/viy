from collections import Counter
from typing import override

import numpy as np
import pytest
from pydantic import ValidationError

from viy.augs import OPS
from viy.augs import AugmentationConfig
from viy.augs import AugmentationPipeline
from viy.augs import NTIREPreset
from viy.augs import ChainStepConfig
from viy.augs import ImageOp
from viy.augs import OpStepConfig
from viy.augs import build_op
from viy.augs import build_step
from viy.augs import ntire_preset
from viy.augs import preset_operations
from viy.augs import base
from viy.augs.presets import PRESETS
from viy.common import ImageArray
from viy.common import ScaleValue


def _image(height: int = 72, width: int = 96, seed: int = 0) -> ImageArray:
    return np.random.default_rng(seed).integers(
        0, 256, size=(height, width, 3), dtype=np.uint8
    )


@pytest.mark.parametrize("name", OPS.names())
def test_every_op_level(name: str) -> None:
    op = build_op(name)
    image = _image()
    original = image.copy()
    for level in op.levels:
        output = op.apply(image, level, np.random.default_rng(1))
        assert output.dtype == np.uint8
        assert output.ndim == 3
        assert output.shape[2] == 3
        assert min(output.shape[:2]) >= 1
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("name", OPS.names())
def test_ops_are_reproducible(name: str) -> None:
    op = build_op(name)
    first, first_applied = op(_image(), np.random.default_rng(7))
    second, second_applied = op(_image(), np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert first_applied == second_applied
    assert first_applied.name == name


def test_op_level_validation() -> None:
    with pytest.raises(ValueError, match="level"):
        build_op("clahe", [9])
    with pytest.raises(ValueError, match="Unknown watermark"):
        build_op("watermark_proxy", ["nope"])
    with pytest.raises(ValueError, match=r"\[low, high\]"):
        build_op("downscale", [0.9, 0.2])
    with pytest.raises(KeyError, match="Unknown image operation"):
        build_op("nope")
    downscale = build_op("downscale", [0.3, 0.8])
    rng = np.random.default_rng(0)
    assert all(0.3 <= float(downscale.sample_level(rng)) <= 0.8 for _ in range(100))


@pytest.mark.parametrize("preset", list(NTIREPreset))
def test_presets_follow_their_groups(preset: NTIREPreset) -> None:
    spec = PRESETS[preset]
    step = build_step(ntire_preset(preset, distort_probability=1.0))
    group_of = {
        name: index for index, group in enumerate(spec.groups) for name in group
    }
    allowed = set(preset_operations(preset))
    lengths: Counter[int] = Counter()
    for seed in range(150):
        _, trace = step(_image(48, 64, seed), np.random.default_rng(seed))
        names = [applied.name for applied in trace]
        assert set(names) <= allowed
        core = [name for name in names if name in group_of]
        assert len({group_of[name] for name in core}) == len(core)
        assert names[: len(core)] == core
        lengths[len(core)] += 1
    assert set(lengths) == {1, 2, 3}


def test_exclude_ops() -> None:
    config = ntire_preset(
        NTIREPreset.TEST_PUBLIC, exclude_ops=["jpeg", "jpeg2000", "downscale"]
    )
    chain = config.steps[0]
    assert isinstance(chain, ChainStepConfig)
    names = [name for group in chain.groups for name in group.ops]
    assert "jpeg" not in names
    assert len(chain.groups) == len(PRESETS[NTIREPreset.TEST_PUBLIC].groups) - 1
    tail = [step.op for step in config.steps[1:] if isinstance(step, OpStepConfig)]
    assert tail == ["watermark_proxy"]
    with pytest.raises(ValueError, match="has no operations"):
        ntire_preset(NTIREPreset.VAL, exclude_ops=["jpeg2000"])


def test_stage6_excludes_proxies_with_genuine_examples() -> None:
    names = preset_operations(NTIREPreset.AGGRESSIVE_STAGE6)
    assert "jpeg_ai_proxy" not in names
    assert "wmforger_proxy" not in names
    assert "cheng2020_proxy" in names


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning_once(self, message: str) -> None:
        self.messages.append(message)


def test_proxies_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(base, "logger", recorder)
    proxies = [name for name in OPS.names() if name.endswith("_proxy")]
    assert len(proxies) == 7
    for name in proxies:
        assert build_op(name).is_proxy
    assert not build_op("jpeg").is_proxy
    assert len(recorder.messages) == len(proxies)
    assert all("not a replacement" in message for message in recorder.messages)


def test_proxies_must_be_marked() -> None:
    with pytest.raises(TypeError, match="proxy_of"):

        class UnmarkedProxy(ImageOp):
            name = "unmarked_proxy"
            default_levels = (1,)

            @override
            def apply(
                self, image: ImageArray, level: ScaleValue, rng: np.random.Generator
            ) -> ImageArray:
                return image


def test_distort_probability() -> None:
    step = build_step(ntire_preset(NTIREPreset.VAL))
    distorted = sum(
        bool(step(_image(32, 32, seed), np.random.default_rng(seed))[1])
        for seed in range(400)
    )
    assert 150 < distorted < 250


def test_pipeline_from_config_is_reproducible() -> None:
    config = AugmentationConfig.model_validate(
        {
            "steps": [
                {"kind": "albumentations", "transform": "HorizontalFlip", "p": 0.5},
                {
                    "kind": "albumentations",
                    "transform": "RandomResizedCrop",
                    "params": {
                        "size": [64, 64],
                        "scale": [0.78, 1.0],
                        "ratio": [0.82, 1.22],
                    },
                },
                {"kind": "ntire_preset", "preset": "aggressive"},
                {
                    "kind": "chain",
                    "groups": [
                        {"ops": ["jpeg", "jpeg2000"]},
                        {"ops": ["brighten"], "weight": 2.0},
                    ],
                    "lengths": {"1": 1.0},
                    "levels": {"jpeg": [90]},
                },
            ]
        }
    )
    assert AugmentationConfig.model_validate(config.model_dump()) == config
    pipeline = AugmentationPipeline.from_pydantic_model(config)
    for seed in range(10):
        first, first_trace = pipeline(_image(seed=seed), np.random.default_rng(seed))
        second, second_trace = pipeline(_image(seed=seed), np.random.default_rng(seed))
        np.testing.assert_array_equal(first, second)
        assert first_trace == second_trace
        assert first_trace[-1].name in {"jpeg", "jpeg2000", "brighten"}


def test_invalid_configs() -> None:
    with pytest.raises(ValidationError):
        AugmentationConfig.model_validate(
            {"steps": [{"kind": "op", "op": "jpeg", "probability": 0.5}]}
        )
    with pytest.raises(ValidationError, match="outside"):
        AugmentationConfig.model_validate(
            {
                "steps": [
                    {
                        "kind": "chain",
                        "groups": [{"ops": ["jpeg"]}],
                        "lengths": {"2": 1.0},
                    }
                ]
            }
        )
    with pytest.raises(ValidationError, match="several groups"):
        AugmentationConfig.model_validate(
            {
                "steps": [
                    {
                        "kind": "chain",
                        "groups": [{"ops": ["jpeg"]}, {"ops": ["jpeg"]}],
                        "lengths": {"1": 1.0},
                    }
                ]
            }
        )
    with pytest.raises(TypeError, match="not an albumentations transform"):
        build_step(
            AugmentationConfig.model_validate(
                {"steps": [{"kind": "albumentations", "transform": "Nope"}]}
            ).steps[0]
        )
    with pytest.raises(ValueError, match="step 'p'"):
        build_step(
            AugmentationConfig.model_validate(
                {
                    "steps": [
                        {
                            "kind": "albumentations",
                            "transform": "HorizontalFlip",
                            "params": {"p": 1.0},
                        }
                    ]
                }
            ).steps[0]
        )
