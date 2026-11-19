# Restore Details (GPU) — ComfyUI custom node

GPU port of **Restore Details** from
[ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters).
Same algorithm, same widget values, same output — but it runs on the GPU
instead of OpenCV on the CPU.

## Why

The original node calls `cv2.GaussianBlur` / `cv2.ximgproc.guidedFilter`, which
are CPU-only, and round-trips every frame through NumPy. On a 25-frame 4K
sequence that step dominated the whole workflow (minutes instead of seconds)
while the GPU sat idle. This port keeps everything in torch on the ComfyUI
compute device.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
No extra dependencies — only `torch` and ComfyUI itself.

Add the node via **Add Node → Image-Filters/image → Restore Details (GPU)**,
or search "restore detail".

## Usage

```
Load EXR Frames ──RGB──► Upscale Image (using Model) ──IMAGE──► images ┐
       └──────────RGB──────────────────────────────────────► detail   ├─► IMAGE
                                                                      ┘
```

`images` is the base (its low frequencies are kept), `detail` is the reference
(its fine detail is transferred). Both must share the same resolution.

**Run it after the downscale to your target resolution, not before.** Cost is
linear in pixel count: filtering an 8K intermediate is 4x the work of filtering
the 4K result, for no visible benefit.

## Inputs

| Input | Type | Default | Description |
|---|---|---|---|
| `images` | IMAGE | — | Base image; supplies low frequencies. RGB, 3 channels. |
| `detail` | IMAGE | — | Reference image; supplies high frequencies. Same size as base. A single image is reused across the batch. |
| `mode` | `add` / `multiply` | `add` | `add`: `(ref - blur(ref)) + blur(base)`. `multiply`: `(ref / blur(ref)) * blur(base)`. |
| `blur_type` | `blur` / `guidedFilter` | `blur` | Gaussian vs. edge-aware color guided filter. |
| `blur_size` | INT 1–1023 | `1` | Detail scale. `d = 2*blur_size + 1`. Gaussian kernel = `d`; guided filter radius = `d` (window `4*blur_size + 3`). |
| `factor` | FLOAT −10–10 | `1.0` | `lerp(base, result, factor)`. 0 = bypass, 1 = full, negative = soften. |
| `match_original_approx` | BOOLEAN | `True` | Reproduce the original's 10%-resolution shortcut when `blur_size >= 50`. |

### Choosing `blur_type`

- **`blur`** — Gaussian, OpenCV-compatible. Cheapest. Can halo across hard edges.
- **`guidedFilter`** — full color guided filter: 3x3 cross-channel covariance with
  a per-pixel inverse, `eps = 0.01`. Preserves edges and color boundaries, so
  noticeably fewer halos. Roughly 3–4x the compute of `blur`, still milliseconds
  per 4K frame on a current GPU.

### `blur_size` is not symmetric between modes

This is inherited from the original node and preserved deliberately:

| `blur_size` | `blur` kernel | `guidedFilter` window |
|---|---|---|
| 1 | 3 px | 7 px |
| 8 | 17 px | 35 px |
| 32 | 65 px | 131 px |

Keep your existing values when swapping nodes; do not halve them.

## Fidelity to the original

- Gaussian kernels replicate `cv2.getGaussianKernel(n, 0)`, including OpenCV's
  fixed binomial tables for n = 1/3/5/7 and `sigma = 0.15*n + 0.35` otherwise.
- Gaussian borders use `BORDER_REFLECT_101`; guided-filter box filters use
  `BORDER_REFLECT` (edge-duplicating), matching `ximgproc` internals.
- The `+0.1` offset before filtering and the final `clamp(0, 1)` are preserved.
- `match_original_approx=True` reproduces the legacy path that, for `d > 100`,
  downsamples to 10%, filters with `d//20*2+1`, and upsamples back. Set it to
  `False` for a true full-resolution filter (better quality, different look).
- Computation is **fp32 by design**. The guided filter's covariance subtracts
  large nearby numbers; fp16 produces visible noise in flat areas.

Expected difference vs. the CPU node: **~1e-5 max** (OpenCV's `boxFilter` uses
running sums and is not bit-exact). With `match_original_approx=False` and
`blur_size >= 50`, expect large differences — that is intended.

### Verify before batch-rendering

```python
a = orig.batch_normalize(img, det, "add", "guidedFilter", 8, 1.0)[0]
b = RestoreDetailGPU().restore(img, det, "add", "guidedFilter", 8, 1.0)[0]
print((a - b).abs().max().item(), (a - b).abs().mean().item())
```

## Performance

- Frames are processed one at a time; peak VRAM is ~1–2 GB per 4K frame
  regardless of batch length.
- If `detail` is a single image, its blurred version is computed once and reused.
- Box filters use separable pooling for small radii and a float64 summed-area
  table for large ones, so cost does not explode with `blur_size`.
- Reflect-padding index vectors are cached per (size, radius, device).

## Limitations

- **RGB only (3 channels).** Feed the `RGB` output of your loader; carry alpha
  separately and re-merge downstream. The guided filter accepts 1 or 3 channels.
- **Output is clamped to 0..1.** Values above 1.0 are clipped, so this node is
  not HDR-safe as-is. If you work with linear HDR, place it after tone mapping.
- `detail` must already match `images` in resolution; there is no internal resize.
- Padding radius must not exceed the image dimensions.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| GPU still idle | Another node in the graph is the CPU bottleneck. Time this node in isolation. |
| Difference concentrated at frame borders | Border-mode mismatch; report your `blur_type` and `blur_size`. |
| Noise in flat areas | Something is forcing fp16 upstream; this node casts to fp32 internally. |
| Output looks nothing like the CPU node | Check `match_original_approx` and confirm `blur_size` was not adjusted. |
| Tooltips missing after edit | Restart ComfyUI **and** hard-reload the browser tab. |

## Credits

Algorithm and original node by the
[ComfyUI-Image-Filters](https://github.com/spacepxl/ComfyUI-Image-Filters) author.
This is an independent GPU reimplementation; check the upstream license before
redistributing. Gaussian and guided-filter semantics follow OpenCV
(`imgproc`, `ximgproc`).