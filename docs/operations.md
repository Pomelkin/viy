# Operations reference

Every augmentation operation of `viy.augs`, with what it does and all of its hyperparameters. For how to combine operations into pipelines, see [Augmentations](augmentations.md).

## How to read this page

Each operation has a single tunable hyperparameter, its **level**, the severity of the distortion. On every call one level is drawn uniformly from the configured list; `downscale` instead draws a value from a range. The operation records the drawn level in the augmentation trace. Everything else an operation needs is either a fixed constant, listed under *fixed*, or drawn at random on every call, listed under *random*.

Override the levels wherever an operation is used:

```python
OpStep("jpeg", p=0.5, levels=[30, 50, 70])
build_op("lensblur", levels=[2, 4])
```

```yaml
- {kind: op, op: jpeg, p: 0.5, levels: [30, 50, 70]}
- kind: chain
  groups: [{ops: [jpeg, jpeg2000]}]
  lengths: {1: 1.0}
  levels: {jpeg: [30, 50, 70]}
```

Operation names and default levels follow the `distortions` and `distortion_scales` columns of the NTIRE 2026 metadata. A name ending in `_proxy` marks a fast stand-in for a distortion that needs a neural network; see [Proxies](#proxies). Unless stated otherwise, an operation keeps the image size.

## Color and tone

### `brighten`
Brighten the image by scaling all intensities by `1 + level`.
- **Level:** relative increase, ≥ 0.
- **Defaults:** 0.1, 0.2, 0.4, 0.7, 1.1.

### `darken`
Darken the image by scaling all intensities by `1 - level`.
- **Level:** relative decrease, from 0 to 1.
- **Defaults:** 0.05, 0.1, 0.2, 0.4, 0.8.

### `lincontrchange`
Linear contrast change: deviations from the mean luminance are scaled by `1 + level`. Negative levels lower the contrast.
- **Level:** relative contrast change, > -1.
- **Defaults:** -0.6, -0.4, 0, 0.15, 0.3.

### `colorsat`
Saturation change: deviations of each pixel from its own luminance are scaled by `1 + level`.
- **Level:** relative saturation change, > -1.
- **Defaults:** -0.4, 0, 0.1, 0.2, 0.4.

### `colorshift`
Chromatic misalignment: the red channel moves `level` pixels in a random direction and the blue channel moves the same distance the opposite way. Edges are replicated.
- **Level:** shift in pixels.
- **Defaults:** 1, 3, 6, 8, 12.
- **Random:** direction, uniform in [0, 2π).

### `rgbshift`
Add a constant offset to every channel.
- **Level:** maximal offset in intensity units on the 0–255 scale.
- **Defaults:** 10, 20, 30, 40, 50.
- **Random:** one integer offset per channel, uniform in [-level, level].

### `jitter`
Color jitter: brightness, contrast and saturation are changed, each by a random factor.
- **Level:** spread of the factors.
- **Defaults:** 0.05, 0.1, 0.2, 0.5, 1.0.
- **Random:** each factor uniform in [max(0, 1 - level), 1 + level]; the order of the three adjustments.

### `quantization`
Posterize every channel to `level` evenly spaced values.
- **Level:** number of values per channel, ≥ 2.
- **Defaults:** 7, 10, 13, 16, 20.

### `randomtonecurve`
Apply a random cubic Bezier tone curve through (0, 0) and (1, 1) to all channels.
- **Level:** standard deviation of the control points.
- **Defaults:** 0.05, 0.15, 0.2, 0.3, 0.4.
- **Random:** the two control points, normal around 0.25 and 0.75 and clipped to [0, 1].

### `clahe`
Contrast-limited adaptive histogram equalization of the lightness channel in LAB space.
- **Level:** severity index, 0 to 4.
- **Defaults:** 0, 1, 2, 3, 4.
- **Fixed:** clip limit 1, 2, 3, 4, 6 for levels 0 to 4; 8 × 8 tiles.

## Blur

### `gausblur`
Gaussian blur with reflected borders.
- **Level:** standard deviation in pixels.
- **Defaults:** 0.1, 0.5, 1, 2, 5.

### `lensblur`
Defocus blur with a uniform disk kernel.
- **Level:** disk radius in pixels.
- **Defaults:** 1, 2, 4, 6, 8.

### `motionblur`
Linear motion blur with a line kernel.
- **Level:** half-length of the line in pixels; the kernel is `2·level + 1` pixels wide.
- **Defaults:** 1, 2, 4, 6, 10.
- **Random:** line angle, uniform in [0, π).

### `glassblur`
Frosted glass, after ImageNet-C:
1. Gaussian blur;
2. move every pixel by a random offset of up to `level` pixels;
3. Gaussian blur again.

- **Level:** maximal pixel displacement.
- **Defaults:** 1, 2, 3, 4, 6.
- **Fixed:** blur sigma `0.5 + 0.1·level`.
- **Random:** per-pixel integer offsets, uniform in [-level, level] on each axis.

## Noise

### `whitenoise`
Additive Gaussian noise.
- **Level:** variance on the [0, 1] intensity scale; the standard deviation is `√level · 255`.
- **Defaults:** 0.001, 0.002, 0.0025, 0.003, 0.005.

### `impulsenoise`
Salt and pepper noise: a `level` share of the pixels becomes black or white, half each, in all channels.
- **Level:** share of affected pixels.
- **Defaults:** 0.001, 0.005, 0.01, 0.015, 0.02.

### `multnoise`
Speckle noise `x · (1 + n)`.
- **Level:** variance of `n`.
- **Defaults:** 0.001, 0.005, 0.01, 0.015, 0.035.

### `isonoise`
Camera sensor noise in HLS space: Poisson noise on lightness and Gaussian noise on hue.
- **Level:** severity index, 0 to 4.
- **Defaults:** 0, 1, 2, 3, 4.
- **Fixed:** (color shift, intensity) = (0.01, 0.1), (0.02, 0.2), (0.03, 0.3), (0.04, 0.4), (0.05, 0.5) for levels 0 to 4.

### `shotnoise`
Poisson photon noise, after ImageNet-C.
- **Level:** severity index, 0 to 4.
- **Defaults:** 0, 1, 2, 3, 4.
- **Fixed:** photon count 60, 25, 12, 5, 3 for levels 0 to 4; fewer photons give more noise.

## Compression and resampling

### `jpeg`
JPEG compression with Pillow.
- **Level:** JPEG quality, 1 to 100.
- **Defaults:** 7, 12, 28, 38, 45.

### `jpeg2000`
JPEG 2000 compression with Pillow, irreversible wavelet.
- **Level:** compression ratio.
- **Defaults:** 16, 32, 45, 120, 170.

### `jpeg_recompression_1`
Repeated JPEG compression.
- **Level:** number of compressions.
- **Defaults:** 2, 3, 4, 5.
- **Random:** quality of each compression, uniform in 30 to 90.

### `jpeg_recompression_2`
Double JPEG compression with the 8 × 8 block grid shifted in between, which leaves misaligned blocking artifacts. The image loses up to 7 pixels at the top and left.
- **Level:** severity index, 0 to 4.
- **Defaults:** 0, 1, 2, 3, 4.
- **Fixed:** qualities (90, 70), (85, 60), (80, 50), (75, 40), (70, 30) for levels 0 to 4.
- **Random:** grid shift of 1 to 7 pixels on each axis.

### `jpeg_recompression_comb`
Repeated compression alternating JPEG and JPEG 2000, starting with JPEG.
- **Level:** number of compressions.
- **Defaults:** 2, 3, 4, 5.
- **Random:** JPEG quality, uniform in 30 to 90; JPEG 2000 ratio from 16, 32, 45, 120, 170.

### `pixelate`
Downsample with area averaging by a factor of `1 - level`, then upsample back with nearest neighbours.
- **Level:** relative size reduction, in [0, 1).
- **Defaults:** 0.01, 0.05, 0.1, 0.2, 0.5.

### `downscale`
Reduce the resolution with antialiased bicubic resampling. **The output is smaller**, never below 8 pixels on a side.
- **Level:** scale factor, drawn uniformly from the range `[low, high]`.
- **Defaults:** range 0.3 to 0.8.

## Geometry

### `randomcrop`
Crop a random window with the image's aspect ratio. **The output is smaller.**
- **Level:** share of the area kept, in (0, 1]; each side shrinks by `√level`.
- **Defaults:** 0.4, 0.5, 0.6, 0.7, 0.8.
- **Random:** window position.

### `randomaspectcrop`
Crop a random window with a changed aspect ratio. **The output is smaller.**
- **Level:** share of the area kept, in (0, 1].
- **Defaults:** 0.4, 0.5, 0.6, 0.7, 0.8.
- **Random:** aspect change `a`, log-uniform in [1/2, 2], giving sides `w·√(level·a)` and `h·√(level/a)` clipped to the image; window position.

### `perspective`
Map a random quadrilateral inside the image onto the full frame, with cubic interpolation and reflected borders.
- **Level:** severity index, 0 to 4.
- **Defaults:** 0, 1, 2, 3, 4.
- **Fixed:** maximal inward corner shift 2%, 4%, 6.5%, 9%, 12% of the image size for levels 0 to 4.
- **Random:** shift of each corner, uniform up to the maximum.

## Proxies

Proxies imitate the magnitude and rough appearance of distortions that need neural networks or per-image optimization; they do not reproduce their mechanism. Each logs a warning when created. Exclude them when these distortions must stay unseen in training.

### `jpeg_ai_proxy`
WebP compression in place of the JPEG AI learned codec.
- **Level:** NTIRE JPEG AI level, 0 to 3. Level 0 is the lowest bitrate, 0.12 bpp; level 3 the highest, 0.75 bpp.
- **Defaults:** 0, 1, 2, 3.
- **Fixed:** WebP quality 18, 28, 42, 58 for levels 0 to 3.

### `cheng2020_proxy`
Resample down and back up, then compress with WebP, in place of the Cheng2020 learned codec.
- **Level:** CompressAI quality index, 1 to 5; 1 is the lowest bitrate.
- **Defaults:** 1, 2, 3, 4, 5.
- **Fixed:** (scale, WebP quality) = (0.5, 20), (0.61, 30), (0.72, 40), (0.82, 50), (0.92, 62) for levels 1 to 5.

### `jpeg_recompression_comb_jpegai_proxy`
Repeated compression alternating JPEG and `jpeg_ai_proxy`, starting with JPEG.
- **Level:** number of compressions.
- **Defaults:** 2, 3, 4, 5.
- **Random:** JPEG quality, uniform in 30 to 90; `jpeg_ai_proxy` level, uniform in 0 to 3.

### `adv_embed_clip_proxy`
Random low-frequency sign pattern in place of an adversarial attack on CLIP image embeddings.
- **Level:** L∞ amplitude in intensity units on the 0–255 scale.
- **Defaults:** 4, 6, 8, 10, 12.
- **Fixed:** the pattern is the sign of a 12 × 12 Gaussian grid upsampled to the image.

### `adv_embed_resnet_proxy`
Random edge-aligned sign pattern in place of an adversarial attack on ResNet embeddings.
- **Level:** L∞ amplitude on the 0–255 scale.
- **Defaults:** 4, 6, 8, 10, 12.
- **Fixed:** the pattern is the sign of the centred Laplacian magnitude of the luminance.
- **Random:** sign of each channel.

### `wmforger_proxy`
Smooth plus edge-aligned residual in place of the WMForger watermark removal attack.
- **Level:** WMForger learning rate as recorded by NTIRE.
- **Defaults:** 0.05, 0.1, 0.2, 0.5, 1.0.
- **Fixed:** amplitude `2 + 7·level`; residual = 0.7 · smooth random field from an 8 × 8 grid + 0.3 · edge sign.

### `watermark_proxy`
A periodic or noise carrier pattern in place of an invisible neural watermark. It is added to one channel and subtracted at 35% from the next channel.
- **Level:** method name: `arwgan`, `dct_marker`, `fin`, `maskwm`, `mbrs`, `pimog`, `pixelseal`, `sshidden`, `stega_stamp` or `syncseal`. Only the names match the real methods.
- **Defaults:** `arwgan`, `dct_marker`, `mbrs`, `pimog`, `pixelseal`, `stega_stamp`.
- **Random:** amplitude, uniform in 2 to 7 on the 0–255 scale; target channel. `pixelseal` and `maskwm` also draw their noise and block mask.
