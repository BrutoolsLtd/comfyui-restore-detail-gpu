import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils

# ---------------- padding exacto de OpenCV ----------------
_IDX = {}

def _reflect_idx(n, r, mode, device):
    key = (n, r, mode, str(device))
    idx = _IDX.get(key)
    if idx is None:
        i = torch.arange(-r, n + r, device=device, dtype=torch.long)
        if mode == "reflect101":          # BORDER_REFLECT_101 (GaussianBlur)
            p = max(2 * n - 2, 1)
            i = i % p
            i = torch.where(i >= n, p - i, i)
        else:                             # BORDER_REFLECT (boxFilter del guided)
            p = 2 * n
            i = i % p
            i = torch.where(i >= n, p - 1 - i, i)
        idx = i
        _IDX[key] = idx
    return idx

def _pad(x, r, mode):
    if r == 0:
        return x
    x = x.index_select(-1, _reflect_idx(x.shape[-1], r, mode, x.device))
    x = x.index_select(-2, _reflect_idx(x.shape[-2], r, mode, x.device))
    return x

# ---------------- box filter (ventana 2r+1, BORDER_REFLECT) ----------------
def _box(x, r):
    if r == 0:
        return x
    k = 2 * r + 1
    xp = _pad(x, r, "reflect")
    if k <= 32:
        y = F.avg_pool2d(xp, (1, k), stride=1)
        return F.avg_pool2d(y, (k, 1), stride=1)
    # radios grandes: imagen integral en float64 para no perder precisión
    xd = xp.double()
    P = torch.cumsum(xd, -1)
    P = torch.cat([torch.zeros_like(P[..., :1]), P], -1)
    S = P[..., k:] - P[..., :-k]
    P = torch.cumsum(S, -2)
    P = torch.cat([torch.zeros_like(P[..., :1, :]), P], -2)
    S = P[..., k:, :] - P[..., :-k, :]
    return (S / (k * k)).to(x.dtype)

# ---------------- GaussianBlur de OpenCV (sigma=0) ----------------
_FIXED = {
    1: [1.0],
    3: [0.25, 0.5, 0.25],
    5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
    7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
}

def _gauss_kernel(n, device):
    if n in _FIXED:
        return torch.tensor(_FIXED[n], dtype=torch.float32, device=device)
    sigma = 0.15 * n + 0.35
    t = torch.arange(n, dtype=torch.float64, device=device) - (n - 1) * 0.5
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return (k / k.sum()).float()

def _gauss(x, ksize):
    r = ksize // 2
    if r == 0:
        return x
    C, k = x.shape[1], _gauss_kernel(ksize, x.device)
    wx = k.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    wy = k.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(_pad(x, r, "reflect101"), wx, groups=C)
    return F.conv2d(_pad(x, r, "reflect101"), wy, groups=C)

# ---------------- guided filter a color (guide == src) ----------------
def _guided(I, r, eps=0.01):
    C = I.shape[1]
    m = _box(I, r)
    if C == 1:
        v = _box(I * I, r) - m * m
        a = v / (v + eps)
        b = m - a * m
        return _box(a, r) * I + _box(b, r)
    if C != 3:
        raise ValueError(f"guidedFilter soporta 1 o 3 canales, recibidos {C}")

    m0, m1, m2 = m[:, 0:1], m[:, 1:2], m[:, 2:3]
    i0, i1, i2 = I[:, 0:1], I[:, 1:2], I[:, 2:3]

    s00 = _box(i0 * i0, r) - m0 * m0 + eps
    s01 = _box(i0 * i1, r) - m0 * m1
    s02 = _box(i0 * i2, r) - m0 * m2
    s11 = _box(i1 * i1, r) - m1 * m1 + eps
    s12 = _box(i1 * i2, r) - m1 * m2
    s22 = _box(i2 * i2, r) - m2 * m2 + eps

    c00 = s11 * s22 - s12 * s12
    c01 = s02 * s12 - s01 * s22
    c02 = s01 * s12 - s02 * s11
    c11 = s00 * s22 - s02 * s02
    c12 = s01 * s02 - s00 * s12
    c22 = s00 * s11 - s01 * s01
    det = (s00 * c00 + s01 * c01 + s02 * c02).clamp_min(1e-12)

    # guide == src  =>  A = I - eps*inv(S),  beta = eps*inv(S)*media
    a00 = 1 - eps * c00 / det; a01 = -eps * c01 / det; a02 = -eps * c02 / det
    a11 = 1 - eps * c11 / det; a12 = -eps * c12 / det; a22 = 1 - eps * c22 / det
    b0 = eps * (c00 * m0 + c01 * m1 + c02 * m2) / det
    b1 = eps * (c01 * m0 + c11 * m1 + c12 * m2) / det
    b2 = eps * (c02 * m0 + c12 * m1 + c22 * m2) / det

    a00, a01, a02 = _box(a00, r), _box(a01, r), _box(a02, r)
    a11, a12, a22 = _box(a11, r), _box(a12, r), _box(a22, r)
    b0, b1, b2 = _box(b0, r), _box(b1, r), _box(b2, r)

    return torch.cat([a00 * i0 + a01 * i1 + a02 * i2 + b0,
                      a01 * i0 + a11 * i1 + a12 * i2 + b1,
                      a02 * i0 + a12 * i1 + a22 * i2 + b2], dim=1)

# ---------------- dispatcher con el atajo del 10% del pack original ----------------
def _blur_like_pack(x, d, blur_type, approx):
    if approx and d > 100:
        H, W = x.shape[-2], x.shape[-1]
        small = F.interpolate(x, scale_factor=0.1, mode='bilinear', align_corners=False)
        kd = d // 20 * 2 + 1
        small = _gauss(small, kd) if blur_type == "blur" else _guided(small, kd, 0.01)
        return F.interpolate(small, size=(H, W), mode='bilinear', align_corners=False)
    return _gauss(x, d) if blur_type == "blur" else _guided(x, d, 0.01)

# ---------------- nodo ----------------
class RestoreDetailGPU:
    """GPU port of Image-Filters' Restore Detail. Same math, same widget values."""

    DESCRIPTION = (
        "Transfer high-frequency detail from a reference image onto a base image, "
        "entirely on the GPU.\n\n"
        "The base image is blurred to isolate its low frequencies, the reference is "
        "blurred the same way, and the reference's detail is re-applied over the base. "
        "Typical use: restore the crispness lost during an AI upscale, using the "
        "original frame as the detail reference.\n\n"
        "Drop-in replacement for 'Restore Details' (ComfyUI-Image-Filters): identical "
        "inputs, identical blur_size semantics, no OpenCV or CPU round-trips."
    )
    SEARCH_ALIASES = ["restore detail", "detail transfer", "frequency separation",
                      "sharpen", "guided filter", "upscale detail"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Base image. Its low frequencies (color, exposure, overall "
                               "structure) are preserved while the reference supplies the fine "
                               "detail. In an upscale workflow, connect the upscaled IMAGE here. "
                               "RGB, 3 channels."}),
                "detail": ("IMAGE", {
                    "tooltip": "Reference image that donates the fine detail. Usually the "
                               "original, sharper frame. Must match the base resolution. If a "
                               "single image is given for a batch, it is reused for every frame "
                               "and its blurred version is computed only once."}),
                "mode": (["add", "multiply"], {
                    "tooltip": "How detail is re-applied.\n"
                               "add: (ref - blur(ref)) + blur(base). Linear high-pass transfer; "
                               "keeps base exposure, can push values out of range.\n"
                               "multiply: (ref / blur(ref)) * blur(base). Ratio-based, scales "
                               "with local brightness; gentler in shadows, stronger local "
                               "contrast. Default: add."}),
                "blur_type": (["blur", "guidedFilter"], {
                    "tooltip": "Filter used to split low/high frequencies.\n"
                                 "blur: Gaussian (OpenCV-compatible, sigma derived from kernel "
                                 "size). Fast, but can halo across strong edges.\n"
                                 "guidedFilter: edge-aware color guided filter (3x3 covariance, "
                                 "eps 0.01). Fewer halos and better color edges, roughly 3-4x "
                                 "more compute. Still real-time on a modern GPU."}),
                "blur_size": ("INT", {"default": 1, "min": 1, "max": 1023,
                    "tooltip": "Detail scale, in pixels. Internally d = 2*blur_size + 1.\n"
                               "blur: d is the Gaussian kernel size (d x d).\n"
                               "guidedFilter: d is used as the RADIUS, so the window is "
                               "(2d+1) = 4*blur_size + 3, about twice as wide.\n"
                               "Small values (1-8) restore fine texture; large values affect "
                               "coarser structure and local contrast. Keep the same value when "
                               "migrating from the CPU node."}),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                     "step": 0.01, "round": 0.01,
                    "tooltip": "Blend strength: lerp(base, result, factor).\n"
                               "0 = bypass (base unchanged), 1 = full effect, >1 = exaggerated "
                               "detail, negative = softens instead of sharpening. Output is "
                               "clamped to 0..1 afterwards."}),
            },
            "optional": {
                "match_original_approx": ("BOOLEAN", {"default": True,
                    "label_on": "legacy (10% proxy)", "label_off": "exact full-res",
                    "tooltip": "Only affects blur_size >= 50 (d > 100).\n"
                               "ON: reproduces the original node's shortcut, which downsamples "
                               "to 10%, filters there and upsamples back. Use this to match "
                               "existing looks frame-for-frame.\n"
                               "OFF: filters at full resolution. More accurate and free of "
                               "resampling artifacts, but the result will differ from the CPU "
                               "node. No practical speed penalty on GPU."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Base image with the reference's detail applied, clamped to 0..1.",)
    FUNCTION = "restore"
    CATEGORY = "Image-Filters/image"

    def restore(self, images, detail, mode, blur_type, blur_size, factor,
                match_original_approx=True):
        device = comfy.model_management.get_torch_device()
        out_dev = comfy.model_management.intermediate_device()
        B, d = images.shape[0], blur_size * 2 + 1
        out = torch.empty_like(images, device=out_dev)
        pbar = comfy.utils.ProgressBar(B)

        static_ref = detail.shape[0] < B      # el original repite el frame 0
        ref_c = refb_c = None

        with torch.inference_mode():
            for i in range(B):
                base = images[i:i + 1].movedim(-1, 1).to(device).float()

                if static_ref and ref_c is not None:
                    ref, ref_b = ref_c, refb_c
                else:
                    j = 0 if static_ref else i
                    ref = detail[j:j + 1].movedim(-1, 1).to(device).float() + 0.1
                    ref_b = _blur_like_pack(ref, d, blur_type, match_original_approx)
                    if static_ref:
                        ref_c, refb_c = ref, ref_b

                t = base + 0.1
                t_b = _blur_like_pack(t, d, blur_type, match_original_approx)

                res = (ref / ref_b) * t_b if mode == "multiply" else (ref - ref_b) + t_b
                res = torch.clamp(torch.lerp(base, res - 0.1, factor), 0, 1)

                out[i:i + 1] = res.movedim(1, -1).to(out_dev, images.dtype)
                pbar.update(1)
                del base, t, t_b, res

        return (out,)


NODE_CLASS_MAPPINGS = {"RestoreDetailGPU": RestoreDetailGPU}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoreDetailGPU": "Restore Details (GPU)"}
