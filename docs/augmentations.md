# Augmentations

Augmentations operate on decoded `uint8` RGB arrays of shape `(H, W, 3)`, before preprocessing. A pipeline is a list of **steps**. Every step:
- draws all its randomness from the `numpy.random.Generator` it is given;
- returns the augmented image and the list of operations it applied, with their severity levels.

The same seed therefore reproduces both the image and its trace.

## Building a pipeline

In Python, pass steps to `AugmentationPipeline`. Albumentations transforms and composes are accepted as they are:

```python
import albumentations as A
from viy.augs import AugmentationPipeline, NTIREPresetStep, OpStep

pipeline = AugmentationPipeline([
    A.HorizontalFlip(p=0.5),
    A.OneOf([A.Blur(), A.GaussNoise()], p=0.3),
    OpStep("jpeg", p=0.5, levels=[30, 50, 70]),
    NTIREPresetStep("aggressive", exclude_ops=["watermark_proxy"]),
])
```

The same pipeline from a config is `AugmentationPipeline.from_pydantic_model(AugmentationConfig)`. In a config every step is selected by `kind`:

| `kind` | Python object | Does |
|---|---|---|
| `op` | `OpStep("jpeg", p=..., levels=...)` | one operation |
| `chain` | `ChainStep(groups, lengths)` | grouped random chain |
| `sequence` | `SequenceStep(steps, p=...)` | nested steps, all or nothing |
| `albumentations` | any albumentations transform or compose | an albumentations transform |
| `ntire_preset` | `NTIREPresetStep("aggressive", ...)` | a reconstructed NTIRE mix |

## Steps

### `op`: one operation

```yaml
- {kind: op, op: jpeg, p: 0.5}                 # default levels of the operation
- {kind: op, op: jpeg, levels: [30, 50, 70]}   # custom levels, drawn uniformly
- {kind: op, op: downscale, levels: [0.5, 0.9]}   # range operation: factor drawn from [low, high]
```

### `chain`: grouped random chain

The step draws the chain length `k` from `lengths`, then `k` distinct groups by `weight`, then one operation uniformly from each group. The chosen operations are applied in random order. Operations of one group never appear together.

```yaml
- kind: chain
  groups:
    - {ops: [jpeg, jpeg2000]}              # compression: at most one per image
    - {ops: [gausblur, lensblur, motionblur]}
    - {ops: [whitenoise], weight: 0.5}     # drawn half as often as other groups
  lengths: {1: 1.0, 2: 1.0}                # one or two operations, equally likely
  levels: {jpeg: [20, 40]}                 # per-operation level overrides
```

### `sequence`: all or nothing

```yaml
- kind: sequence
  p: 0.3                                    # apply the whole block to 30% of images
  steps:
    - {kind: op, op: brighten}
    - {kind: op, op: jpeg}
```

### `albumentations`: any albumentations transform

```yaml
- {kind: albumentations, transform: HorizontalFlip, p: 0.5}
- kind: albumentations
  transform: RandomResizedCrop
  params: {size: [320, 320], scale: [0.78, 1.0], ratio: [0.82, 1.22]}
```

In a config, `transform` is a class name from `albumentations` and `params` are its keyword arguments; set the probability with the step `p`, not inside `params`. Any transform class is picked up by name, with no registration. In Python, pass the transform object itself, with its own `p`.

Either way the transform is reseeded from the pipeline generator on every call, so it is as reproducible as the native operations. Every transform that actually fires, including those inside `OneOf` or `Compose`, appears in the trace as `albumentations.<Name>` without a level. Transforms must return `uint8` images; normalization belongs to [preprocessing](training.md#preprocessing).

### `ntire_preset`: a reconstructed NTIRE mix

```yaml
- {kind: ntire_preset, preset: test_public}
- {kind: ntire_preset, preset: aggressive, distort_probability: 0.7, exclude_ops: [watermark_proxy, jpeg2000]}
```

A preset expands into the steps above. `exclude_ops` removes operations; groups left empty disappear.

## NTIRE presets

The NTIRE 2026 validation and test splits record the distortion chain of every image. Their statistics show one construction for all splits:

1. exactly half of the images are distorted, equally for both classes;
2. the chain length `k` is drawn uniformly from 1, 2 and 3; `k` distinct **groups** are drawn and one operation from each, applied in random order;
3. on the test splits, a watermark follows with p ≈ 0.36;
4. a downscale by a factor drawn from 0.3–0.8 follows with p = 0.75, always last.

The groups were identified as sets of operations that never co-occur in a chain.

| Preset | Groups | Distorted share | Watermark |
|---|---|---:|---|
| `val` | jpeg · lincontrchange · brighten, darken · randomcrop, randomaspectcrop · colorsat, colorshift, rgbshift · gausblur, lensblur, motionblur · impulsenoise, multnoise, whitenoise · jitter, pixelate, quantization | 0.5 | no |
| `val_hard` | quantization · jpeg · jpeg_ai\* · adv_embed\* · gausblur, lensblur · clahe, randomtonecurve, rgbshift · jpeg_recompression_1, \_2, comb_jpegai\* · perspective, randomcrop, randomaspectcrop · impulsenoise, isonoise, multnoise, whitenoise | 0.5 | no |
| `test_public` | brighten · lensblur · jpeg, jpeg2000 · cheng2020\*, jpeg_ai\* · jpeg_recompression_1, \_2, comb_jpegai\* · adv_embed_clip\*, adv_embed_resnet\*, wmforger\* · perspective, randomcrop, randomaspectcrop · impulsenoise, isonoise, shotnoise · clahe, colorsat, randomtonecurve, rgbshift | 0.5 | 6 methods, p 0.36 |
| `test_private` | like `test_public`, plus `glassblur` in the blur group and `jpeg_recompression_comb` in the recompression group | 0.5 | 7 methods, p 0.37 |
| `aggressive` | `test_public` groups | 0.9 | as `test_public` |
| `aggressive_stage6` | `aggressive` without `jpeg_ai_proxy`, `jpeg_recompression_comb_jpegai_proxy`, `wmforger_proxy` | 0.9 | as `test_public` |

\* proxy, see [below](#proxies).

The validation and test presets are meant for analysis and for building distorted copies of clean data. For training:
- `aggressive` is a strong augmentation: the public test mix applied to 90% of images;
- `aggressive_stage6` is the same without the JPEG AI and WMForger proxies, for use together with the genuine examples of `ntire/shard_6`, a split stored locally only.

Robustness to a transform is only measured honestly when the model never saw it during training. To keep such transforms out of training, exclude them:

```yaml
- kind: ntire_preset
  preset: aggressive
  exclude_ops: [jpeg_ai_proxy, cheng2020_proxy, jpeg_recompression_comb_jpegai_proxy,
                adv_embed_clip_proxy, adv_embed_resnet_proxy, wmforger_proxy, watermark_proxy]
```

## Operations

Names and severity levels follow the `distortions` and `distortion_scales` columns of the NTIRE metadata, so augmentation traces and benchmark slices share one vocabulary. NTIRE does not document the meaning of every level. The semantics of `colorshift`, `clahe`, `isonoise`, `shotnoise`, `perspective` and `jpeg_recompression_2` are interpreted by viy; treat those levels as ordinal.

| Family | Operations |
|---|---|
| color and tone | `brighten`, `darken`, `lincontrchange`, `colorsat`, `colorshift`, `rgbshift`, `jitter`, `quantization`, `randomtonecurve`, `clahe` |
| blur | `gausblur`, `lensblur`, `motionblur`, `glassblur` |
| noise | `whitenoise`, `impulsenoise`, `multnoise`, `isonoise`, `shotnoise` |
| compression and resampling | `jpeg`, `jpeg2000`, `jpeg_recompression_1`, `jpeg_recompression_2`, `jpeg_recompression_comb`, `pixelate`, `downscale` |
| geometry | `randomcrop`, `randomaspectcrop`, `perspective` |
| proxies | `jpeg_ai_proxy`, `cheng2020_proxy`, `jpeg_recompression_comb_jpegai_proxy`, `adv_embed_clip_proxy`, `adv_embed_resnet_proxy`, `wmforger_proxy`, `watermark_proxy` |

The [operations reference](operations.md) describes every operation with its level semantics, default levels and fixed and random parameters. The authoritative list is `viy.augs.OPS.names()`.

## Proxies

Learned codecs, adversarial embedding attacks, WMForger and invisible watermarks need neural networks or per-image optimization, which does not fit in DataLoader workers. Their `_proxy` operations imitate only the magnitude and rough look of the real distortions:

| Proxy | What it really is | What it stands in for |
|---|---|---|
| codec proxies | WebP and resampling | learned codecs, which smooth textures in their own way |
| attack proxies | random noise within the same L∞ budget | gradient-based attacks, which hit models far harder |
| `watermark_proxy` | fixed periodic or noise patterns; only the method names match | neural watermarking |

Every proxy logs one warning per process when it is created:

```
WARNING [base.py] Augmentation 'cheng2020_proxy' is a proxy of the Cheng2020 learned codec, not a replacement: ...
```

`op.is_proxy` and the `_proxy` name suffix identify proxies programmatically. Genuine JPEG AI and WMForger examples exist as the `ntire/shard_6` split, which is stored locally only. Genuine versions of the others are planned as offline-generated splits.

## Running a pipeline

```python
import numpy as np
from PIL import Image
from viy.augs import build_op, trace_to_json

image = np.asarray(Image.open("photo.jpg").convert("RGB"))
augmented, trace = pipeline(image, np.random.default_rng(0))
print(trace_to_json(trace))      # e.g. [{"name": "albumentations.HorizontalFlip", "scale": null}, {"name": "jpeg", "scale": 50}]

jpeg = build_op("jpeg", levels=[50])
compressed, applied = jpeg(image, np.random.default_rng(0))   # applied == AppliedOp("jpeg", 50)
blurred = build_op("gausblur").apply(image, 2.0, np.random.default_rng(0))   # fixed level
```

To write your own operation, see [Extending](extending.md#a-new-augmentation-operation).
