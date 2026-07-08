"""Streamlit-free per-image NMJ analysis and NMJ_Plot PNG rendering (shared by batch UI and CLI regen)."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import distance_transform_edt, zoom
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops

from nmj_master_dashboard import (
    BTX_CLASS_EARLY_NMJ,
    BTX_CLASS_MUSCLE,
    BTX_CLASS_NEURON,
    BTX_CLASS_ORPHANED,
    BTX_SIGNAL_CLASS_ORDER,
    BTX_SIGNAL_CLASS_PALETTE,
    MIN_PIXELS_FOR_SHAPE,
    RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL,
    ROUNDNESS_KRUSKAL_CLASSES,
    SPOT_DENSITY_PER_MM2_LABEL,
    UM2_PER_MM2,
    dataframe_for_roundness_kde_and_kruskal,
    draw_proximity_joint,
    load_confocal_image,
    normalize_btx_signal_classes,
    nmj_vs_orphan_intensity_mannwhitney_title,
    proximity_joint_axes,
    roundness_3way_kruskal_title,
    tissue_mask_verification_side_by_side,
    tissue_mask_verification_title,
    total_image_area_um2_from_metadata,
)

DOG_SIGMA_RATIO_CONSERVATIVE = 1.6
DOG_SIGMA_RATIO_HIGH = 1.3
AUTO_DOG_THRESHOLD_MAD_K = 3.0


def dog_sigma_ratio_from_sensitivity(sensitivity: str) -> float:
    return DOG_SIGMA_RATIO_HIGH if sensitivity == "High" else DOG_SIGMA_RATIO_CONSERVATIVE


def compute_sigma_bounds_px(min_sigma_um, max_sigma_um, pixel_size_um, image_shape):
    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    min_px_raw = float(min_sigma_um) / pixel_size_safe
    max_px_raw = float(max_sigma_um) / pixel_size_safe
    min_px = max(0.5, min_px_raw)
    max_px = max(min_px + 0.1, max_px_raw)
    h, w = image_shape[:2]
    sigma_cap = max(2.0, min(64.0, min(h, w) / 12.0))
    if min_px > sigma_cap:
        return None, None, sigma_cap
    max_px = min(max_px, sigma_cap)
    return min_px, max_px, sigma_cap


def remove_muscle_haze(img, pixel_size_um, bg_sigma_um):
    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    sigma_um = max(1e-9, float(bg_sigma_um))
    bg_sigma_px = float(sigma_um) / pixel_size_safe
    background = gaussian(img, sigma=bg_sigma_px, preserve_range=True)
    result = img.astype(np.float32, copy=False) - background.astype(np.float32, copy=False)
    return np.clip(result, 0.0, None).astype(img.dtype, copy=False)


def threshold_raw_for_spot_crop(threshold_used, p_high, window_btx):
    th_raw = float(threshold_used) * float(p_high)
    if window_btx.size == 0:
        return th_raw
    wmax = float(np.max(window_btx))
    if th_raw <= 0 or (wmax > 0 and th_raw >= wmax):
        try:
            th_raw = float(threshold_otsu(window_btx))
        except ValueError:
            th_raw = wmax * 0.5 if wmax > 0 else 1e-9
    return th_raw


def detect_blobs_stable(
    img_btx_norm, min_diameter_um, max_diameter_um, pixel_size_um, threshold, sigma_ratio=None
):
    if sigma_ratio is None:
        sigma_ratio = DOG_SIGMA_RATIO_CONSERVATIVE
    min_sigma_um = float(min_diameter_um) / (2.0 * np.sqrt(2.0))
    max_sigma_um = float(max_diameter_um) / (2.0 * np.sqrt(2.0))
    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    max_sigma_px_raw = float(max_sigma_um) / pixel_size_safe
    dog_scale = 1.0
    dog_sigma_target = 48.0
    if max_sigma_px_raw > dog_sigma_target:
        dog_scale = max(0.1, dog_sigma_target / max_sigma_px_raw)
        img_for_dog = zoom(img_btx_norm, zoom=dog_scale, order=1)
    else:
        img_for_dog = img_btx_norm
    pixel_size_for_dog = pixel_size_safe / dog_scale
    min_sigma_px, max_sigma_px, sigma_cap = compute_sigma_bounds_px(
        min_sigma_um=min_sigma_um,
        max_sigma_um=max_sigma_um,
        pixel_size_um=pixel_size_for_dog,
        image_shape=img_for_dog.shape,
    )
    if min_sigma_px is None:
        return None, dog_scale, sigma_cap
    threshold_for_dog = float(threshold)
    if dog_scale < 1.0:
        threshold_for_dog *= max(0.6, dog_scale)
    blobs = blob_dog(
        img_for_dog,
        min_sigma=min_sigma_px,
        max_sigma=max_sigma_px,
        sigma_ratio=float(sigma_ratio),
        threshold=threshold_for_dog,
    )
    if len(blobs) == 0:
        return blobs, dog_scale, sigma_cap
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    if dog_scale != 1.0:
        blobs[:, :2] = blobs[:, :2] / dog_scale
        blobs[:, 2] = blobs[:, 2] / dog_scale
    r_um = blobs[:, 2] * pixel_size_safe
    d_um = 2.0 * r_um
    ok = (d_um >= float(min_diameter_um)) & (d_um <= float(max_diameter_um))
    blobs = blobs[ok]
    return blobs, dog_scale, sigma_cap


def estimate_auto_threshold(img_btx_norm):
    sample = np.asarray(img_btx_norm, dtype=np.float32)[::4, ::4].ravel()
    if sample.size == 0:
        return 0.05
    pos = sample[sample > 0.005]
    if pos.size < 50:
        return 0.05
    median = float(np.median(pos))
    mad = float(np.median(np.abs(pos - median)))
    std_est = 1.4826 * mad
    k = float(AUTO_DOG_THRESHOLD_MAD_K)
    return float(np.clip(median + k * std_est, 0.02, 0.12))


@dataclass
class NmjPlotContext:
    czi_file: str
    df_spots: pd.DataFrame
    blobs: np.ndarray
    img_muscle: np.ndarray
    img_neuron: np.ndarray
    img_btx: np.ndarray
    img_btx_raw: np.ndarray
    muscle_mask: np.ndarray
    neuron_mask: np.ndarray
    m_thresh: float
    n_thresh: float
    m_thresh_mult: float
    n_thresh_mult: float
    distance_threshold_um: float
    zone_densities: tuple[float, float, float, float]
    zone_areas: tuple[float, float, float, float]
    threshold_used: float
    total_image_area_um2: float


def analyze_image_for_nmj_plot(
    image_path: str,
    *,
    muscle_idx: int,
    neuron_idx: int,
    btx_idx: int,
    pixel_size: float,
    min_diameter_um: float,
    max_diameter_um: float,
    auto_threshold: bool,
    threshold: float | None,
    dog_sigma_ratio: float,
    btx_bg_radius_um: float,
    m_thresh_mult: float,
    n_thresh_mult: float,
    distance_threshold_um: float,
) -> NmjPlotContext | None:
    total_image_area_um2 = total_image_area_um2_from_metadata(image_path)
    channels = load_confocal_image(
        image_path,
        channel_indices=[muscle_idx, neuron_idx, btx_idx],
    )
    if isinstance(channels, dict):
        img_muscle = channels[muscle_idx]
        img_neuron = channels[neuron_idx]
        img_btx = channels[btx_idx]
    else:
        img_muscle = channels[muscle_idx]
        img_neuron = channels[neuron_idx]
        img_btx = channels[btx_idx]
    channels = None
    gc.collect()

    img_btx_raw = img_btx.copy()
    img_btx = remove_muscle_haze(img_btx, pixel_size, btx_bg_radius_um)
    p_high = float(np.percentile(img_btx, 99.9))
    if p_high <= 0:
        p_high = 1e-5
    img_btx_norm = np.clip(img_btx.astype(np.float32, copy=False) / p_high, 0.0, 1.0)
    threshold_used = estimate_auto_threshold(img_btx_norm) if auto_threshold else float(threshold)
    blobs, _dog_scale, _sigma_cap = detect_blobs_stable(
        img_btx_norm=img_btx_norm,
        min_diameter_um=min_diameter_um,
        max_diameter_um=max_diameter_um,
        pixel_size_um=pixel_size,
        threshold=threshold_used,
        sigma_ratio=dog_sigma_ratio,
    )
    if blobs is None:
        return None
    if auto_threshold and len(blobs) == 0:
        threshold_retry = max(0.02, float(threshold_used) * 0.8)
        blobs_retry, _, _ = detect_blobs_stable(
            img_btx_norm=img_btx_norm,
            min_diameter_um=min_diameter_um,
            max_diameter_um=max_diameter_um,
            pixel_size_um=pixel_size,
            threshold=threshold_retry,
            sigma_ratio=dog_sigma_ratio,
        )
        if blobs_retry is not None and len(blobs_retry) > 0:
            blobs = blobs_retry
            threshold_used = float(threshold_retry)
    if blobs is None or len(blobs) == 0:
        return None

    m_thresh = threshold_otsu(img_muscle) * m_thresh_mult
    n_thresh = threshold_otsu(img_neuron) * n_thresh_mult
    muscle_mask = img_muscle > m_thresh
    neuron_mask = img_neuron > n_thresh
    edt_muscle_um = (
        distance_transform_edt(muscle_mask == 0).astype(np.float32, copy=False) * np.float32(pixel_size)
    )
    edt_neuron_um = (
        distance_transform_edt(neuron_mask == 0).astype(np.float32, copy=False) * np.float32(pixel_size)
    )

    spots_data: list[dict[str, Any]] = []
    for index, blob in enumerate(blobs):
        y, x, r = blob
        y_idx, x_idx = int(round(y)), int(round(x))
        y_idx = np.clip(y_idx, 0, edt_muscle_um.shape[0] - 1)
        x_idx = np.clip(x_idx, 0, edt_muscle_um.shape[1] - 1)
        d_m_center = float(edt_muscle_um[y_idx, x_idx])
        d_n_center = float(edt_neuron_um[y_idx, x_idx])
        r_um = float(r * pixel_size)
        d_m_um = max(0.0, d_m_center - r_um)
        d_n_um = max(0.0, d_n_center - r_um)

        roundness = np.nan
        area_px_spot = np.nan
        mean_intensity = 0.0
        overlap_ratio = 0.0
        r_int = max(3, int(r * 2))
        box_y1 = max(0, y_idx - r_int)
        box_y2 = min(img_btx.shape[0], y_idx + r_int)
        box_x1 = max(0, x_idx - r_int)
        box_x2 = min(img_btx.shape[1], x_idx + r_int)
        window_btx = img_btx[box_y1:box_y2, box_x1:box_x2]
        window_neuron = neuron_mask[box_y1:box_y2, box_x1:box_x2]
        if window_btx.size >= 4:
            try:
                th = threshold_raw_for_spot_crop(threshold_used, p_high, window_btx)
                labeled = label(window_btx > th)
                center_y, center_x = y_idx - box_y1, x_idx - box_x1
                spot_label = labeled[center_y, center_x]
                if spot_label == 0:
                    spot_label = labeled[np.unravel_index(np.argmax(window_btx), window_btx.shape)]
                if spot_label > 0:
                    props_list = regionprops(labeled)
                    props_dict = {p.label: p for p in props_list}
                    if spot_label in props_dict:
                        prop = props_dict[spot_label]
                    elif props_list:
                        prop = max(props_list, key=lambda x: x.area)
                    else:
                        prop = None
                    if prop is not None:
                        spot_mask = labeled == prop.label
                        area_px_spot = float(prop.area)
                        if prop.area > MIN_PIXELS_FOR_SHAPE:
                            roundness = float(np.clip(1.0 - float(prop.eccentricity), 0.0, 1.0))
                        mean_intensity = float(np.mean(window_btx[spot_mask]))
                        overlap_pixels = np.sum(spot_mask & window_neuron)
                        if prop.area > 0:
                            overlap_ratio = float(overlap_pixels / prop.area) * 100.0
            except Exception:
                pass

        spots_data.append(
            {
                "SPOT_ID": index,
                "POSITION_X": x * pixel_size,
                "POSITION_Y": y * pixel_size,
                "RADIUS": r * pixel_size,
                "CIRCULARITY": roundness,
                "ROUNDNESS": roundness,
                "AREA_PX": area_px_spot,
                "MEAN_INTENSITY": mean_intensity,
                "INNERVATION_OVERLAP_PCT": overlap_ratio,
                "Dist_to_Muscle_um": d_m_um,
                "Dist_to_Neuron_um": d_n_um,
                "Dist_to_Muscle_center_um": d_m_center,
                "Dist_to_Neuron_center_um": d_n_center,
                "DETECTION_THRESHOLD_USED": threshold_used,
                "is_NMJ": (d_m_um <= distance_threshold_um) and (d_n_um <= distance_threshold_um),
            }
        )

    df_spots = pd.DataFrame(spots_data)
    if df_spots.empty:
        return None

    def classify_quadrant(row):
        if row["Dist_to_Muscle_um"] <= distance_threshold_um and row["Dist_to_Neuron_um"] <= distance_threshold_um:
            return BTX_CLASS_EARLY_NMJ
        if row["Dist_to_Muscle_um"] <= distance_threshold_um:
            return BTX_CLASS_MUSCLE
        if row["Dist_to_Neuron_um"] <= distance_threshold_um:
            return BTX_CLASS_NEURON
        return BTX_CLASS_ORPHANED

    df_spots["BTX signal class"] = df_spots.apply(classify_quadrant, axis=1)
    df_spots = normalize_btx_signal_classes(df_spots)
    df_spots["TOTAL_IMAGE_AREA_um2"] = total_image_area_um2
    df_spots["Resolution_Class"] = np.where(
        float(pixel_size) > RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL,
        "Low-Res",
        "High-Res",
    )

    mask_m_zone = edt_muscle_um <= distance_threshold_um
    mask_n_zone = edt_neuron_um <= distance_threshold_um
    um2_per_px = float(pixel_size**2)
    area_nmj_um2 = float(np.sum(mask_m_zone & mask_n_zone)) * um2_per_px
    area_m_um2 = float(np.sum(mask_m_zone & ~mask_n_zone)) * um2_per_px
    area_n_um2 = float(np.sum(mask_n_zone & ~mask_m_zone)) * um2_per_px
    area_o_um2 = float(np.sum(~mask_m_zone & ~mask_n_zone)) * um2_per_px
    nmj_count = int(df_spots["is_NMJ"].sum())
    near_m_only = int(
        len(
            df_spots[
                (df_spots["Dist_to_Muscle_um"] <= distance_threshold_um)
                & (df_spots["Dist_to_Neuron_um"] > distance_threshold_um)
            ]
        )
    )
    near_n_only = int(
        len(
            df_spots[
                (df_spots["Dist_to_Neuron_um"] <= distance_threshold_um)
                & (df_spots["Dist_to_Muscle_um"] > distance_threshold_um)
            ]
        )
    )
    c_orphan = int(
        len(
            df_spots[
                (df_spots["Dist_to_Muscle_um"] > distance_threshold_um)
                & (df_spots["Dist_to_Neuron_um"] > distance_threshold_um)
            ]
        )
    )
    dens_nmj = (nmj_count / area_nmj_um2 * UM2_PER_MM2) if area_nmj_um2 > 0 else 0.0
    dens_m = (near_m_only / area_m_um2 * UM2_PER_MM2) if area_m_um2 > 0 else 0.0
    dens_n = (near_n_only / area_n_um2 * UM2_PER_MM2) if area_n_um2 > 0 else 0.0
    dens_o = (c_orphan / area_o_um2 * UM2_PER_MM2) if area_o_um2 > 0 else 0.0

    return NmjPlotContext(
        czi_file=os.path.basename(image_path),
        df_spots=df_spots,
        blobs=blobs,
        img_muscle=img_muscle,
        img_neuron=img_neuron,
        img_btx=img_btx,
        img_btx_raw=img_btx_raw,
        muscle_mask=muscle_mask,
        neuron_mask=neuron_mask,
        m_thresh=float(m_thresh),
        n_thresh=float(n_thresh),
        m_thresh_mult=float(m_thresh_mult),
        n_thresh_mult=float(n_thresh_mult),
        distance_threshold_um=float(distance_threshold_um),
        zone_densities=(dens_nmj, dens_m, dens_n, dens_o),
        zone_areas=(area_nmj_um2, area_m_um2, area_n_um2, area_o_um2),
        threshold_used=float(threshold_used),
        total_image_area_um2=float(total_image_area_um2),
    )


def build_nmj_plot_figure(ctx: NmjPlotContext):
    """Build the 12-panel per-image NMJ figure and return it (caller owns saving/closing)."""
    df_spots = ctx.df_spots
    blobs = ctx.blobs
    czi_file = ctx.czi_file
    dens_nmj, dens_m, dens_n, dens_o = ctx.zone_densities
    distance_threshold_um = ctx.distance_threshold_um

    def auto_contrast(img):
        p_low, p_high = np.percentile(img, (5, 99.5))
        return rescale_intensity(img, in_range=(p_low, p_high), out_range=(0.0, 1.0))

    img_m_norm = auto_contrast(ctx.img_muscle)
    img_n_norm = auto_contrast(ctx.img_neuron)
    img_b_norm = auto_contrast(ctx.img_btx)

    def robust_minmax(img):
        p_low, p_high = np.percentile(img, (1, 99.8))
        if p_high <= p_low:
            p_high = p_low + 1e-6
        return float(p_low), float(p_high)

    btx_clean_vis = ctx.img_btx.astype(np.float32)
    disp_low, disp_high = robust_minmax(ctx.img_btx_raw.astype(np.float32))
    img_btx_raw_vis = rescale_intensity(
        ctx.img_btx_raw.astype(np.float32), in_range=(disp_low, disp_high), out_range=(0.0, 1.0)
    )
    img_btx_clean_vis = rescale_intensity(btx_clean_vis, in_range=(disp_low, disp_high), out_range=(0.0, 1.0))
    raw_clean_side_by_side = np.concatenate([img_btx_raw_vis, img_btx_clean_vis], axis=1)
    comp_r = np.clip(img_n_norm + (img_b_norm * 1.2), 0, 1)
    comp_g = np.clip(img_m_norm + (img_b_norm * 1.2), 0, 1)
    comp_b = np.zeros_like(img_m_norm)
    composite_rgb = np.stack([comp_r, comp_g, comp_b], axis=-1)

    fig = plt.figure(figsize=(24, 30))
    outer = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    ax_scatter, ax_prox_kde_x, ax_prox_kde_y, ax_prox_title = proximity_joint_axes(
        fig, outer[0, 0], title_first=True
    )
    ax_size_kde = fig.add_subplot(outer[0, 1])
    ax_circ_kde = fig.add_subplot(outer[0, 2])
    ax_overlap_kde = fig.add_subplot(outer[1, 0])
    ax_intensity_kde = fig.add_subplot(outer[1, 1])
    ax_btx_clean = fig.add_subplot(outer[1, 2])
    ax_btx_only = fig.add_subplot(outer[2, 0])
    ax_btx_marked = fig.add_subplot(outer[2, 1])
    ax_comp_marked = fig.add_subplot(outer[2, 2])
    ax_comp_arrows = fig.add_subplot(outer[3, 0])
    ax_density = fig.add_subplot(outer[3, 1])
    ax_tissue_mask = fig.add_subplot(outer[3, 2])

    _roundness_order = list(ROUNDNESS_KRUSKAL_CLASSES)
    df_shape = dataframe_for_roundness_kde_and_kruskal(df_spots)
    if len(df_shape) > 0:
        sns.kdeplot(
            data=df_shape,
            x="ROUNDNESS",
            hue="BTX signal class",
            hue_order=_roundness_order,
            palette=BTX_SIGNAL_CLASS_PALETTE,
            ax=ax_circ_kde,
            common_norm=False,
            fill=True,
            clip=(0, 1),
            warn_singular=False,
        )
    ax_circ_kde.set_title(
        roundness_3way_kruskal_title(
            df_spots,
            label_base=f"3. {BTX_CLASS_EARLY_NMJ} Roundness KDE (1 − eccentricity)",
        )
    )
    ax_circ_kde.set_xlabel("Roundness (1 = circle)")
    ax_circ_kde.set_ylabel("Probability Density")
    ax_circ_kde.set_xlim(0, 1)

    if len(df_spots) > 0:
        sns.kdeplot(
            data=df_spots,
            x="RADIUS",
            hue="BTX signal class",
            hue_order=BTX_SIGNAL_CLASS_ORDER,
            palette=BTX_SIGNAL_CLASS_PALETTE,
            ax=ax_size_kde,
            common_norm=False,
            fill=True,
            warn_singular=False,
        )
    ax_size_kde.set_title("2. BTX Size KDE")
    ax_size_kde.set_xlabel("Radius (μm)")
    ax_size_kde.set_ylabel("Probability Density")

    df_innervation_img = (
        df_spots[df_spots["BTX signal class"] == BTX_CLASS_EARLY_NMJ] if len(df_spots) > 0 else df_spots
    )
    if len(df_innervation_img) > 0:
        sns.histplot(
            data=df_innervation_img,
            x="INNERVATION_OVERLAP_PCT",
            color=BTX_SIGNAL_CLASS_PALETTE[BTX_CLASS_EARLY_NMJ],
            ax=ax_overlap_kde,
        )
    ax_overlap_kde.set_title(f"4. {BTX_CLASS_EARLY_NMJ} Innervation Distribution")
    ax_overlap_kde.set_xlabel(f"{BTX_CLASS_EARLY_NMJ} Innervation (%)")
    ax_overlap_kde.set_ylabel("Count")

    if len(df_spots) > 0:
        _int_vals_img = (
            df_spots["MEAN_INTENSITY"].dropna() if "MEAN_INTENSITY" in df_spots.columns else pd.Series(dtype=float)
        )
        _int_max_img = float(_int_vals_img.quantile(0.999)) if len(_int_vals_img) > 0 else None
        sns.kdeplot(
            data=df_spots,
            x="MEAN_INTENSITY",
            hue="BTX signal class",
            hue_order=BTX_SIGNAL_CLASS_ORDER,
            palette=BTX_SIGNAL_CLASS_PALETTE,
            ax=ax_intensity_kde,
            common_norm=False,
            fill=True,
            warn_singular=False,
            clip=(0, _int_max_img) if _int_max_img is not None else None,
        )
        if _int_max_img is not None:
            ax_intensity_kde.set_xlim(0, _int_max_img * 1.05)
    intensity_title_img, _ = nmj_vs_orphan_intensity_mannwhitney_title(
        df_spots.assign(SOURCE_IMAGE=czi_file) if len(df_spots) else df_spots,
        label_base="5. Receptor Intensity KDE",
    )
    ax_intensity_kde.set_title(intensity_title_img)
    ax_intensity_kde.set_xlabel("Mean Fluorescence Intensity")
    ax_intensity_kde.set_ylabel("Probability Density")

    draw_proximity_joint(
        ax_scatter,
        ax_prox_kde_x,
        ax_prox_kde_y,
        df_spots,
        distance_threshold_um,
        "1. BTX Proximity Analysis",
        marginal_combined_black=True,
        title_ax=ax_prox_title,
    )

    ax_btx_clean.imshow(raw_clean_side_by_side, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    pane_w = img_btx_raw_vis.shape[1]
    ax_btx_clean.axvline(x=pane_w - 0.5, color="yellow", linewidth=2.5)
    ax_btx_clean.set_title("6. Raw BTX (L) | Cleaned BTX (R)")
    ax_btx_clean.axis("off")

    ax_btx_only.imshow(img_btx_clean_vis, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    ax_btx_only.set_title("7. Cleaned BTX")
    ax_btx_only.axis("off")

    ax_btx_marked.imshow(img_btx_clean_vis, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    ax_btx_marked.set_title("8. Cleaned BTX + Detected Spots")
    ax_btx_marked.axis("off")

    ax_comp_marked.imshow(composite_rgb, aspect="auto")
    ax_comp_marked.set_title("9. Composite + All Detected Spots")
    ax_comp_marked.axis("off")

    ax_comp_arrows.imshow(composite_rgb, aspect="auto")
    ax_comp_arrows.set_title(f"10. Composite + {BTX_CLASS_EARLY_NMJ} Only")
    ax_comp_arrows.axis("off")

    dens_data = pd.DataFrame(
        {
            "Zone": [BTX_CLASS_EARLY_NMJ, BTX_CLASS_MUSCLE, BTX_CLASS_NEURON, BTX_CLASS_ORPHANED],
            "Density": [dens_nmj, dens_m, dens_n, dens_o],
        }
    )
    sns.barplot(
        data=dens_data,
        x="Zone",
        y="Density",
        hue="Zone",
        palette=["red", "green", "blue", "gray"],
        legend=False,
        ax=ax_density,
    )
    ax_density.set_title(f"11. BTX Density ({SPOT_DENSITY_PER_MM2_LABEL})")
    ax_density.set_ylabel(SPOT_DENSITY_PER_MM2_LABEL)
    ax_density.set_xlabel("")

    tissue_mask_vis = tissue_mask_verification_side_by_side(
        img_m_norm, ctx.muscle_mask, img_n_norm, ctx.neuron_mask
    )
    ax_tissue_mask.imshow(tissue_mask_vis, aspect="auto")
    pane_w_mask = tissue_mask_vis.shape[1] // 2
    ax_tissue_mask.axvline(x=pane_w_mask - 0.5, color="yellow", linewidth=2.5)
    ax_tissue_mask.set_title(
        tissue_mask_verification_title(
            m_thresh=ctx.m_thresh,
            n_thresh=ctx.n_thresh,
            m_thresh_mult=ctx.m_thresh_mult,
            n_thresh_mult=ctx.n_thresh_mult,
        )
    )
    ax_tissue_mask.axis("off")

    by_spot_id = df_spots.set_index("SPOT_ID")
    for index, blob in enumerate(blobs):
        y, x, r = blob
        c1 = plt.Circle((x, y), r, color="yellow", linewidth=1, fill=False)
        c2 = plt.Circle((x, y), r, color="yellow", linewidth=1, fill=False)
        ax_btx_marked.add_patch(c1)
        ax_comp_marked.add_patch(c2)
        if index in by_spot_id.index and bool(by_spot_id.at[index, "is_NMJ"]):
            target_x = x + r + 2
            target_y = y - r - 2
            start_x = target_x + 80
            start_y = target_y - 80
            ax_comp_arrows.annotate(
                "",
                xy=(target_x, target_y),
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
            )

    return fig


def save_nmj_plot_png(ctx: NmjPlotContext, out_path: str) -> None:
    """Render the per-image NMJ figure and write it to ``out_path``."""
    fig = build_nmj_plot_figure(ctx)
    fig.savefig(out_path, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
