"""
A/B Test: Spot detection sensitivity comparison on a single CZI.

Configs tested:
  A – Current code:  sigma_ratio=1.1, threshold compensated by 0.1/0.6,
                     auto_thr = median*0.4, clip [0.03, 0.08]
  B – sigma_ratio=1.6 only: same auto_thr formula as A, no compensation
  C – b689dd7 style:  sigma_ratio=1.6, auto_thr = median + 7*MAD, clip [0.02, 0.12]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aicspylibczi
from skimage.feature import blob_dog
from skimage.filters import gaussian
from scipy.ndimage import distance_transform_edt, zoom

# ── Config ────────────────────────────────────────────────────────────────────
CZI_PATH    = "/app/0714M-HF/0714M-HF-03.czi"
BTX_CH      = 3
PIXEL_SIZE  = 0.08234957298136651   # µm / px  (from channel_mapping_config.json)
MIN_DIAM_UM = 5.0
MAX_DIAM_UM = 12.0
OUT_PNG     = "/app/0714M-HF/AB_test_detection.png"
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers (mirrored from BTX_batch.py) ─────────────────────────────────────

def load_btx_channel(path, btx_ch):
    czi = aicspylibczi.CziFile(path)
    dims0 = czi.get_dims_shape()[0]
    z_range = dims0.get("Z", (0, 1))
    planes = []
    for z in range(z_range[0], z_range[1]):
        tile, _ = czi.read_image(C=btx_ch, Z=z)
        planes.append(np.squeeze(tile).astype(np.float32))
    img = np.max(np.stack(planes, axis=0), axis=0) if len(planes) > 1 else planes[0]
    return img


def remove_muscle_haze(img, pixel_size_um, max_spot_diameter_um=12.0):
    px = max(float(pixel_size_um), 1e-9)
    bg_sigma_um = max(50.0, float(max_spot_diameter_um) * 5.0)
    bg_sigma_px = bg_sigma_um / px
    background = gaussian(img, sigma=bg_sigma_px, preserve_range=True)
    result = img.astype(np.float32) - background.astype(np.float32)
    return np.clip(result, 0.0, None).astype(img.dtype)


def compute_sigma_bounds_px(min_sigma_um, max_sigma_um, pixel_size_um, image_shape):
    px = max(float(pixel_size_um), 1e-9)
    min_px = max(0.5, float(min_sigma_um) / px)
    max_px = max(min_px + 0.1, float(max_sigma_um) / px)
    h, w = image_shape[:2]
    sigma_cap = max(2.0, min(64.0, min(h, w) / 12.0))
    if min_px > sigma_cap:
        return None, None, sigma_cap
    max_px = min(max_px, sigma_cap)
    return min_px, max_px, sigma_cap


def run_dog(img_norm, pixel_size_um, min_diam, max_diam, auto_thr,
            sigma_ratio, apply_ratio_compensation):
    """Run DoG with given sigma_ratio and threshold strategy."""
    min_sigma_um = float(min_diam) / (2.0 * np.sqrt(2.0))
    max_sigma_um = float(max_diam) / (2.0 * np.sqrt(2.0))
    px = max(float(pixel_size_um), 1e-9)
    max_sigma_px_raw = max_sigma_um / px
    dog_scale = 1.0
    dog_sigma_target = 48.0

    if max_sigma_px_raw > dog_sigma_target:
        dog_scale = max(0.1, dog_sigma_target / max_sigma_px_raw)
        img_for_dog = zoom(img_norm, zoom=dog_scale, order=1)
    else:
        img_for_dog = img_norm

    px_for_dog = px / dog_scale
    min_sigma_px, max_sigma_px, _ = compute_sigma_bounds_px(
        min_sigma_um, max_sigma_um, px_for_dog, img_for_dog.shape)
    if min_sigma_px is None:
        return np.empty((0, 3))

    threshold_for_dog = float(auto_thr)
    if dog_scale < 1.0:
        threshold_for_dog *= max(0.6, dog_scale)
    if apply_ratio_compensation:
        _sk_default = 1.6
        threshold_for_dog *= (sigma_ratio - 1.0) / (_sk_default - 1.0)

    blobs = blob_dog(img_for_dog,
                     min_sigma=min_sigma_px, max_sigma=max_sigma_px,
                     threshold=threshold_for_dog, sigma_ratio=sigma_ratio)
    if len(blobs) == 0:
        return blobs

    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    if dog_scale != 1.0:
        blobs[:, :2] /= dog_scale
        blobs[:, 2]  /= dog_scale

    # Physical diameter filter
    r_um = blobs[:, 2] * px
    d_um = 2.0 * r_um
    ok = (d_um >= float(min_diam)) & (d_um <= float(max_diam))
    return blobs[ok]


def auto_thr_current(img_norm):
    """Current formula: median*0.4 clip [0.03, 0.08]"""
    sample = np.asarray(img_norm, dtype=np.float32)[::4, ::4].ravel()
    pos = sample[sample > 0.02]
    if pos.size < 50:
        return 0.05
    return float(np.clip(float(np.median(pos)) * 0.4, 0.03, 0.08))


def auto_thr_b689dd7(img_norm):
    """b689dd7 formula: median + 7*MAD  clip [0.02, 0.12]"""
    sample = np.asarray(img_norm, dtype=np.float32)[::4, ::4].ravel()
    pos = sample[sample > 0.005]
    if pos.size < 50:
        return 0.05
    median = float(np.median(pos))
    mad    = float(np.median(np.abs(pos - median)))
    return float(np.clip(median + 7.0 * 1.4826 * mad, 0.02, 0.12))


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading BTX channel …")
img_raw = load_btx_channel(CZI_PATH, BTX_CH)
print(f"  image shape: {img_raw.shape}, dtype: {img_raw.dtype}")

print("Applying background subtraction …")
img_haze = remove_muscle_haze(img_raw, PIXEL_SIZE, MAX_DIAM_UM)

p_high = float(np.percentile(img_haze, 99.9))
if p_high <= 0:
    p_high = 1e-5
img_norm = np.clip(img_haze.astype(np.float32) / p_high, 0.0, 1.0)

# Compute auto-thresholds
thr_A = auto_thr_current(img_norm)
thr_B = auto_thr_current(img_norm)   # same formula, different sigma_ratio
thr_C = auto_thr_b689dd7(img_norm)

print(f"\nAuto-thresholds (before compensation):")
print(f"  A (current,  sigma=1.1 + compensation): raw thr={thr_A:.4f}  →  effective={thr_A*(0.1/0.6):.4f}")
print(f"  B (sigma=1.6, no compensation):          raw thr={thr_B:.4f}  →  effective={thr_B:.4f}")
print(f"  C (b689dd7,  sigma=1.6, no compensation): raw thr={thr_C:.4f}  →  effective={thr_C:.4f}")

configs = [
    dict(label="A – Current\n(σ_ratio=1.1 + compensation\nauto: median×0.4)",
         sigma_ratio=1.1, thr=thr_A, compensate=True,  color="red"),
    dict(label="B – σ_ratio=1.6\n(no compensation\nauto: median×0.4)",
         sigma_ratio=1.6, thr=thr_B, compensate=False, color="orange"),
    dict(label="C – b689dd7 style\n(σ_ratio=1.6, no compensation\nauto: median+7×MAD)",
         sigma_ratio=1.6, thr=thr_C, compensate=False, color="cyan"),
]

results = []
for cfg in configs:
    print(f"\nRunning {cfg['label'].split(chr(10))[0]} …")
    blobs = run_dog(img_norm, PIXEL_SIZE, MIN_DIAM_UM, MAX_DIAM_UM,
                    cfg["thr"], cfg["sigma_ratio"], cfg["compensate"])
    n = len(blobs)
    print(f"  → {n} spots detected")
    results.append((cfg, blobs))

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
fig.suptitle(
    f"A/B Detection Test — 0714M-HF-03.czi  (BTX ch={BTX_CH}, px={PIXEL_SIZE:.4f} µm)\n"
    f"Diameter range: {MIN_DIAM_UM}–{MAX_DIAM_UM} µm",
    fontsize=13, fontweight="bold"
)

display = np.sqrt(img_norm.astype(np.float32))   # gamma stretch for visibility

for ax, (cfg, blobs) in zip(axes, results):
    ax.imshow(display, cmap="gray", interpolation="nearest",
              vmin=0, vmax=float(np.percentile(display, 99.5)))
    for blob in blobs:
        y, x, r = blob
        circle = plt.Circle((x, y), r, color=cfg["color"],
                             linewidth=0.8, fill=False, alpha=0.7)
        ax.add_patch(circle)
    effective_thr = cfg["thr"] * (0.1/0.6) if cfg["compensate"] else cfg["thr"]
    ax.set_title(
        f"{cfg['label']}\n"
        f"n = {len(blobs)} spots\n"
        f"raw thr={cfg['thr']:.4f}  eff={effective_thr:.4f}",
        fontsize=9, pad=4
    )
    ax.axis("off")

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PNG}")
