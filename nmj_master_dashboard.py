"""
Streamlit-free plotting helpers for aggregate (master) dashboards and optional regeneration
from saved ``ALL_FOLDERS_MASTER_RESULTS*.csv`` via ``regenerate_all_folders_panel_pdfs.py``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox as MplBbox
import numpy as np
import pandas as pd
import seaborn as sns

BTX_SIGNAL_CLASS_ORDER = ("NMJ", "Aneural AChR clusters", "Neuron-associated BTX signal", "Orphaned")
BTX_SIGNAL_CLASS_PALETTE = {
    "NMJ": "red",
    "Aneural AChR clusters": "green",
    "Neuron-associated BTX signal": "blue",
    "Orphaned": "gray",
}

BTX_SIGNAL_CLASS_LEGACY_ALIASES = {
    "Muscle Only": "Aneural AChR clusters",
    "Muscle only": "Aneural AChR clusters",
    "Neuron Only": "Neuron-associated BTX signal",
    "Neuron only": "Neuron-associated BTX signal",
}

MIN_PIXELS_FOR_SHAPE = 20
RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL = 0.5
ROUNDNESS_KRUSKAL_CLASSES = ("NMJ", "Aneural AChR clusters", "Neuron-associated BTX signal")


def dataframe_for_roundness_kde_and_kruskal(df):
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
        return df.iloc[0:0]
    required = {"AREA_PX", "BTX signal class", "ROUNDNESS"}
    if not required <= set(df.columns):
        return df.iloc[0:0]
    return df[
        (df["AREA_PX"] >= MIN_PIXELS_FOR_SHAPE)
        & (df["BTX signal class"].isin(ROUNDNESS_KRUSKAL_CLASSES))
    ].dropna(subset=["ROUNDNESS"])


def normalize_btx_signal_classes(df):
    if df is None or len(df) == 0 or "BTX signal class" not in df.columns:
        return df
    out = df.copy()
    out["BTX signal class"] = (
        out["BTX signal class"].astype(str).str.strip().replace(BTX_SIGNAL_CLASS_LEGACY_ALIASES)
    )
    return out


def ensure_roundness_column(df):
    if df is None or len(df) == 0:
        return df
    if "ROUNDNESS" in df.columns:
        return df
    if "CIRCULARITY" in df.columns:
        out = df.copy()
        out["ROUNDNESS"] = out["CIRCULARITY"]
        return out
    return df


def nmj_vs_orphan_intensity_mannwhitney_title(df, *, label_base="5. Global Receptor Intensity"):
    from scipy.stats import mannwhitneyu

    if df is None or len(df) == 0:
        return f"{label_base} (No Data)", None
    required = {"BTX signal class", "MEAN_INTENSITY"}
    if not required <= set(df.columns):
        return f"{label_base} (Missing Columns)", None

    nmj = df[df["BTX signal class"] == "NMJ"]["MEAN_INTENSITY"].dropna()
    orphan = df[df["BTX signal class"] == "Orphaned"]["MEAN_INTENSITY"].dropna()
    if len(nmj) < 3 or len(orphan) < 3:
        return f"{label_base} (Insufficient Clusters)", None

    try:
        _mw_stat, p_val = mannwhitneyu(nmj, orphan, alternative="greater")
    except ValueError:
        return f"{label_base} (Test Failed)", None

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    med_nmj = float(nmj.median())
    med_orphan = float(orphan.median())
    title = (
        f"{label_base} (Mann-Whitney P = {p_val:.4g} {sig} | "
        f"NMJ: {med_nmj:.2f} vs Orphaned: {med_orphan:.2f})"
    )
    summary = {
        "p_val": float(p_val),
        "nmj_median": med_nmj,
        "orphan_median": med_orphan,
        "fold_change": (med_nmj / med_orphan) if med_orphan > 0 else np.nan,
    }
    return title, summary


def roundness_3way_kruskal_title(df, label_base="3. Global NMJ Roundness Analysis"):
    from scipy.stats import kruskal

    if df is None or len(df) == 0:
        return f"{label_base} (No Data)"
    required = {"AREA_PX", "BTX signal class", "ROUNDNESS"}
    if not required <= set(df.columns):
        return f"{label_base} (Missing Columns)"

    df_valid = dataframe_for_roundness_kde_and_kruskal(df)

    g_nmj = df_valid[df_valid["BTX signal class"] == "NMJ"]["ROUNDNESS"]
    g_aneural = df_valid[df_valid["BTX signal class"] == "Aneural AChR clusters"]["ROUNDNESS"]
    g_neuron = df_valid[df_valid["BTX signal class"] == "Neuron-associated BTX signal"]["ROUNDNESS"]

    if any(len(g) < 3 for g in [g_nmj, g_aneural, g_neuron]):
        return f"{label_base} (Insufficient group sizes for 3-way test)"

    try:
        _stat, p_val = kruskal(g_nmj, g_aneural, g_neuron)
    except ValueError:
        return f"{label_base} (Test Failed)"

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    return (
        f"{label_base} (Kruskal P = {p_val:.4g} {sig})\n"
        f"Medians - NMJ: {g_nmj.median():.2f} | Aneural: {g_aneural.median():.2f} | "
        f"Neuron-Assoc: {g_neuron.median():.2f}"
    )


BTX_SIGNAL_CLASS_HISTOGRAM_LABELS = {
    "NMJ": "NMJ",
    "Aneural AChR clusters": "Muscle-only",
    "Neuron-associated BTX signal": "Neuron-only",
    "Orphaned": "Orphaned",
}


def global_btx_intensity_otsu_threshold(intensities):
    """Otsu on pooled spot ``MEAN_INTENSITY`` values (all classes combined)."""
    from skimage.filters import threshold_otsu

    arr = np.asarray(intensities, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(threshold_otsu(arr))


GLOBAL_BTX_INTENSITY_OTSU_COL = "GLOBAL_BTX_INTENSITY_OTSU"


def annotate_global_btx_intensity_otsu(df, *, otsu_value=None):
    """Append dataset-global BTX intensity Otsu as the last CSV column (same value on every row)."""
    if df is None:
        return df
    out = df.copy()
    if GLOBAL_BTX_INTENSITY_OTSU_COL in out.columns:
        out = out.drop(columns=[GLOBAL_BTX_INTENSITY_OTSU_COL])
    if otsu_value is None:
        if "MEAN_INTENSITY" not in out.columns or len(out) == 0:
            otsu_value = np.nan
        else:
            otsu_value = global_btx_intensity_otsu_threshold(out["MEAN_INTENSITY"])
    elif not np.isfinite(otsu_value):
        otsu_value = np.nan
    else:
        otsu_value = float(otsu_value)
    out[GLOBAL_BTX_INTENSITY_OTSU_COL] = otsu_value
    return out


def _draw_otsu_vline(ax, otsu_th, *, show_label=False, ymax_frac=0.95):
    if not np.isfinite(otsu_th):
        return
    ax.axvline(otsu_th, color="black", linestyle="--", linewidth=1.4, zorder=5)
    if not show_label:
        return
    y_top = ax.get_ylim()[1]
    ax.text(
        otsu_th,
        y_top * ymax_frac,
        f" Otsu\n {otsu_th:.1f}",
        fontsize=8,
        va="top",
        ha="left",
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
        zorder=6,
    )


def add_global_btx_intensity_histogram_with_otsu_row(fig, outer, row_idx, master_df, *, panel_num):
    """Full-width histogram row: combined KDE + per-class histograms with shared Otsu line."""
    axes_out = []
    gs_hist = outer[row_idx, :].subgridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)
    ax_combined = fig.add_subplot(gs_hist[0, :])
    class_axes = [fig.add_subplot(gs_hist[1, col]) for col in range(4)]
    axes_out = [ax_combined, *class_axes]

    master_df = normalize_btx_signal_classes(master_df)
    if master_df is None or len(master_df) == 0 or "MEAN_INTENSITY" not in master_df.columns:
        ax_combined.text(0.5, 0.5, "No intensity data", ha="center", va="center")
        ax_combined.set_axis_off()
        for ax in class_axes:
            ax.set_axis_off()
        ax_combined.set_title(f"{panel_num}. Global BTX Intensity Histograms (No Data)")
        return axes_out, np.nan

    intensities = master_df["MEAN_INTENSITY"].dropna().to_numpy(dtype=np.float64)
    otsu_th = global_btx_intensity_otsu_threshold(intensities)
    int_max = float(np.quantile(intensities, 0.999)) if intensities.size else None
    xlim = (0, int_max * 1.05) if int_max is not None and int_max > 0 else None

    sns.kdeplot(
        data=master_df,
        x="MEAN_INTENSITY",
        hue="BTX signal class",
        hue_order=BTX_SIGNAL_CLASS_ORDER,
        palette=BTX_SIGNAL_CLASS_PALETTE,
        ax=ax_combined,
        common_norm=False,
        fill=True,
        warn_singular=False,
        clip=xlim,
    )
    _draw_otsu_vline(ax_combined, otsu_th, show_label=True)
    otsu_label = f"{otsu_th:.1f} A.U." if np.isfinite(otsu_th) else "n/a"
    ax_combined.set_title(
        f"{panel_num}. Global BTX Intensity Histograms by Class (Global Otsu = {otsu_label})"
    )
    ax_combined.set_xlabel("Mean Fluorescence Intensity (A.U.)")
    ax_combined.set_ylabel("Probability Density")
    if xlim is not None:
        ax_combined.set_xlim(*xlim)

    for ax, btx_class in zip(class_axes, BTX_SIGNAL_CLASS_ORDER):
        class_df = master_df[master_df["BTX signal class"] == btx_class]
        label = BTX_SIGNAL_CLASS_HISTOGRAM_LABELS.get(btx_class, btx_class)
        color = BTX_SIGNAL_CLASS_PALETTE.get(btx_class, "gray")
        if len(class_df) > 0:
            sns.histplot(
                data=class_df,
                x="MEAN_INTENSITY",
                color=color,
                ax=ax,
                kde=True,
                stat="count",
                bins=30,
                edgecolor="white",
                linewidth=0.4,
            )
            _draw_otsu_vline(ax, otsu_th)
            vals = class_df["MEAN_INTENSITY"].dropna()
            stats_txt = (
                f"n = {len(vals)}\n"
                f"Med: {float(vals.median()):.1f}\n"
                f"Mean: {float(vals.mean()):.1f}"
            )
            ax.text(
                0.97,
                0.97,
                stats_txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.7"),
            )
        else:
            ax.text(0.5, 0.5, "No spots", ha="center", va="center", fontsize=9)
        ax.set_title(label, color=color, fontweight="bold")
        ax.set_xlabel("Mean Fluorescence Intensity (A.U.)")
        ax.set_ylabel("Spot Count")
        if xlim is not None:
            ax.set_xlim(*xlim)

    return axes_out, otsu_th


def proximity_joint_axes(
    fig, outer_cell, hspace=0.08, wspace=0.08, title_first=False, *, large_main_panel=False
):
    if title_first:
        if large_main_panel:
            height_ratios = [0.25, 1.0, 5.0]
            width_ratios = [5.0, 1.0]
        else:
            height_ratios = [0.28, 1, 4]
            width_ratios = [4, 1]

        inner = outer_cell.subgridspec(
            3,
            2,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=0.01,
            wspace=0.01,
        )
        ax_title = fig.add_subplot(inner[0, :])
        ax_title.axis("off")
        ax_kde_x = fig.add_subplot(inner[1, 0])
        ax_corner = fig.add_subplot(inner[1, 1])
        ax_corner.axis("off")
        ax_main = fig.add_subplot(inner[2, 0], sharex=ax_kde_x)
        ax_kde_y = fig.add_subplot(inner[2, 1], sharey=ax_main)
        ax_kde_x.tick_params(labelbottom=False)
        ax_kde_y.tick_params(labelleft=False)
        return ax_main, ax_kde_x, ax_kde_y, ax_title
    inner = outer_cell.subgridspec(
        2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=hspace, wspace=wspace
    )
    ax_kde_x = fig.add_subplot(inner[0, 0])
    ax_corner = fig.add_subplot(inner[0, 1])
    ax_corner.axis("off")
    ax_main = fig.add_subplot(inner[1, 0], sharex=ax_kde_x)
    ax_kde_y = fig.add_subplot(inner[1, 1], sharey=ax_main)
    ax_kde_x.tick_params(labelbottom=False)
    ax_kde_y.tick_params(labelleft=False)
    return ax_main, ax_kde_x, ax_kde_y


def _scatter_dataframe_with_clip_jitter(df, sigma_um=0.02, seed=42):
    if df is None or len(df) == 0:
        return df
    if "Dist_to_Muscle_um" not in df.columns or "Dist_to_Neuron_um" not in df.columns:
        return df
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    jx = np.abs(rng.normal(0.0, sigma_um, size=n))
    jy = np.abs(rng.normal(0.0, sigma_um, size=n))
    mx = out["Dist_to_Muscle_um"].to_numpy(copy=True).astype(float)
    my = out["Dist_to_Neuron_um"].to_numpy(copy=True).astype(float)
    clipped_x = mx <= 1e-9
    clipped_y = my <= 1e-9
    mx[clipped_x] = mx[clipped_x] + jx[clipped_x]
    my[clipped_y] = my[clipped_y] + jy[clipped_y]
    out["Dist_to_Muscle_um"] = mx
    out["Dist_to_Neuron_um"] = my
    return out


def _mannwhitney_neuron_distance_nmij_vs_aneural(df, min_per_group=3):
    from scipy.stats import mannwhitneyu

    if df is None or len(df) == 0:
        return None
    if "BTX signal class" not in df.columns or "Dist_to_Neuron_um" not in df.columns:
        return None
    dist_nmj = df[df["BTX signal class"] == "NMJ"]["Dist_to_Neuron_um"].dropna()
    dist_aneural = df[df["BTX signal class"] == "Aneural AChR clusters"]["Dist_to_Neuron_um"].dropna()
    if len(dist_nmj) < min_per_group or len(dist_aneural) < min_per_group:
        return None
    try:
        _, p_val = mannwhitneyu(dist_nmj, dist_aneural, alternative="less")
    except ValueError:
        return None
    return {
        "p_val": float(p_val),
        "med_nmj": float(dist_nmj.median()),
        "med_aneural": float(dist_aneural.median()),
    }


def get_spatial_docking_title(df, label_base="1. Synaptic Docking Precision", n_spots=None, min_per_group=3):
    head = label_base if n_spots is None else f"{label_base} (n={n_spots})"
    if df is None or len(df) == 0:
        return f"{head}\n(No Data)"
    res = _mannwhitney_neuron_distance_nmij_vs_aneural(df, min_per_group=min_per_group)
    if res is None:
        return (
            f"{head}\n(Insufficient clusters for Mann-Whitney; need ≥{min_per_group} NMJ and ≥{min_per_group} Aneural)"
        )
    p_val = res["p_val"]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    return (
        f"{head}\n(Mann-Whitney P = {p_val:.4g} {sig} | "
        f"NMJ: {res['med_nmj']:.2f} μm vs Aneural: {res['med_aneural']:.2f} μm)"
    )


def spatial_docking_mannwhitneyu_p(df, min_per_group=3):
    res = _mannwhitney_neuron_distance_nmij_vs_aneural(df, min_per_group=min_per_group)
    return float(res["p_val"]) if res is not None else float("nan")


def draw_proximity_joint(
    ax_main,
    ax_kde_x,
    ax_kde_y,
    df,
    distance_threshold_um,
    title,
    *,
    marginal_alpha=0.35,
    scatter_alpha=0.65,
    scatter_size=None,
    marginal_combined_black=False,
    title_ax=None,
):
    if df is not None and len(df) > 0:
        if marginal_combined_black:
            sns.kdeplot(
                data=df,
                x="Dist_to_Muscle_um",
                ax=ax_kde_x,
                color="black",
                fill=True,
                alpha=marginal_alpha,
                warn_singular=False,
            )
            sns.kdeplot(
                data=df,
                y="Dist_to_Neuron_um",
                ax=ax_kde_y,
                color="black",
                fill=True,
                alpha=marginal_alpha,
                warn_singular=False,
            )
        else:
            sns.kdeplot(
                data=df,
                x="Dist_to_Muscle_um",
                hue="BTX signal class",
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE,
                ax=ax_kde_x,
                common_norm=False,
                fill=True,
                alpha=marginal_alpha,
                legend=False,
                warn_singular=False,
            )
            sns.kdeplot(
                data=df,
                y="Dist_to_Neuron_um",
                hue="BTX signal class",
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE,
                ax=ax_kde_y,
                common_norm=False,
                fill=True,
                alpha=marginal_alpha,
                legend=False,
                warn_singular=False,
            )
    df_scatter = _scatter_dataframe_with_clip_jitter(df)
    scatter_kw = dict(
        data=df_scatter,
        x="Dist_to_Muscle_um",
        y="Dist_to_Neuron_um",
        hue="BTX signal class",
        hue_order=BTX_SIGNAL_CLASS_ORDER,
        palette=BTX_SIGNAL_CLASS_PALETTE,
        ax=ax_main,
    )
    if scatter_size is not None:
        scatter_kw["s"] = scatter_size
        scatter_kw["alpha"] = scatter_alpha
    else:
        scatter_kw["alpha"] = 0.7
    sns.scatterplot(**scatter_kw)
    ax_main.axvline(x=distance_threshold_um, color="black", linestyle="--")
    ax_main.axhline(y=distance_threshold_um, color="black", linestyle="--")
    n_spots = int(len(df)) if df is not None else 0
    full_title = get_spatial_docking_title(df, label_base=title, n_spots=n_spots)
    if title_ax is not None:
        title_ax.clear()
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.5,
            full_title,
            transform=title_ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax_main.set_title("")
    else:
        ax_main.set_title(full_title)
    ax_main.set_xlabel("Distance to Muscle — spot edge (μm)")
    ax_main.set_ylabel("Distance to Neuron — spot edge (μm)")

    ax_kde_x.set_xlabel("")
    ax_kde_x.set_ylabel("")
    ax_kde_y.set_ylabel("")
    ax_kde_y.set_xlabel("")


def _export_figure_panels_to_pdfs(fig, panels, output_stem, *, pad_inches=0.08):
    written = []
    errors = []
    out_dir = os.path.dirname(output_stem) or "."
    os.makedirs(out_dir, exist_ok=True)

    try:
        fig.canvas.draw()
    except Exception as e_draw:
        errors.append(f"canvas.draw: {e_draw}")

    renderer = fig.canvas.get_renderer()
    if renderer is None:
        errors.append("get_renderer() returned None")
        return written, errors

    for suffix, axes_list in panels:
        axes_list = [a for a in axes_list if a is not None]
        if not axes_list:
            errors.append(f"{suffix}: no axes")
            continue
        bboxes = []
        for ax in axes_list:
            try:
                bb = ax.get_tightbbox(renderer)
                if bb is not None and np.isfinite(bb.bounds).all() and bb.width > 0 and bb.height > 0:
                    bboxes.append(bb)
            except Exception as e_bb:
                errors.append(f"{suffix} tightbbox: {e_bb}")
                continue
        if not bboxes:
            errors.append(f"{suffix}: no valid tightbbox")
            continue
        try:
            bbox = MplBbox.union(bboxes)
        except Exception:
            bbox = bboxes[0]
            for b in bboxes[1:]:
                bbox = bbox.union(b)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        if not np.isfinite(bbox_inches.bounds).all():
            errors.append(f"{suffix}: non-finite bbox_inches")
            continue
        out_pdf = os.path.abspath(f"{output_stem}_{suffix}.pdf")
        try:
            fig.savefig(
                out_pdf,
                format="pdf",
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                dpi=fig.dpi,
            )
            if os.path.isfile(out_pdf) and os.path.getsize(out_pdf) > 0:
                written.append(out_pdf)
            else:
                errors.append(f"{suffix}: save produced missing or empty file {out_pdf!r}")
        except Exception as e_save:
            errors.append(f"{suffix}: {e_save}")
    return written, errors


def save_all_folders_summary_png(
    master_df, out_png, distance_threshold_um, *, also_save_panel_pdfs=False
):
    master_df = normalize_btx_signal_classes(master_df)

    folder_stats = (
        master_df.groupby("SOURCE_FOLDER")
        .agg(
            total_spots=("is_NMJ", "size"),
            nmj_spots=("is_NMJ", "sum"),
            mean_radius_um=("RADIUS", "mean"),
            mean_overlap_pct=("INNERVATION_OVERLAP_PCT", "mean"),
            median_dist_muscle_um=("Dist_to_Muscle_um", "median"),
            median_dist_neuron_um=("Dist_to_Neuron_um", "median"),
        )
        .reset_index()
        .sort_values("SOURCE_FOLDER")
    )
    folder_stats["nmj_rate_pct"] = np.where(
        folder_stats["total_spots"] > 0,
        folder_stats["nmj_spots"] / folder_stats["total_spots"] * 100.0,
        0.0,
    )

    fig = plt.figure(figsize=(22, 24), constrained_layout=True)
    outer = fig.add_gridspec(3, 2)
    ax_nmj_rate = fig.add_subplot(outer[0, 0])
    ax_total_spots = fig.add_subplot(outer[0, 1])
    ax_radius = fig.add_subplot(outer[1, 0])
    ax_overlap = fig.add_subplot(outer[1, 1])
    ax_distance = fig.add_subplot(outer[2, 0])
    ax_prox_main, ax_prox_kde_x, ax_prox_kde_y, ax_prox_title = proximity_joint_axes(
        fig, outer[2, 1], title_first=True, large_main_panel=True
    )

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="nmj_rate_pct", ax=ax_nmj_rate, color="#d62728")
    ax_nmj_rate.set_title("1. NMJ Formation Rate by Folder")
    ax_nmj_rate.set_xlabel("Folder")
    ax_nmj_rate.set_ylabel("NMJ Rate (%)")
    ax_nmj_rate.tick_params(axis="x", rotation=45)

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="total_spots", ax=ax_total_spots, color="#1f77b4")
    ax_total_spots.set_title("2. Total BTX Spots by Folder")
    ax_total_spots.set_xlabel("Folder")
    ax_total_spots.set_ylabel("Count")
    ax_total_spots.tick_params(axis="x", rotation=45)

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="mean_radius_um", ax=ax_radius, color="#2ca02c")
    ax_radius.set_title("3. Mean Spot Radius by Folder")
    ax_radius.set_xlabel("Folder")
    ax_radius.set_ylabel("Radius (um)")
    ax_radius.tick_params(axis="x", rotation=45)

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="mean_overlap_pct", ax=ax_overlap, color="#9467bd")
    ax_overlap.set_title("4. Mean Innervation Overlap by Folder")
    ax_overlap.set_xlabel("Folder")
    ax_overlap.set_ylabel("Overlap (%)")
    ax_overlap.tick_params(axis="x", rotation=45)

    sns.scatterplot(
        data=folder_stats,
        x="median_dist_muscle_um",
        y="median_dist_neuron_um",
        hue="SOURCE_FOLDER",
        s=120,
        ax=ax_distance,
    )
    ax_distance.axvline(x=distance_threshold_um, color="black", linestyle="--")
    ax_distance.axhline(y=distance_threshold_um, color="black", linestyle="--")
    ax_distance.set_title("5. Folder Medians in Distance Space")
    ax_distance.set_xlabel("Median Dist to Muscle (um)")
    ax_distance.set_ylabel("Median Dist to Neuron (um)")

    draw_proximity_joint(
        ax_prox_main,
        ax_prox_kde_x,
        ax_prox_kde_y,
        master_df,
        distance_threshold_um,
        "6. All-Folders Proximity",
        scatter_alpha=0.35,
        scatter_size=18,
        marginal_combined_black=True,
        title_ax=ax_prox_title,
    )

    fig.savefig(out_png, bbox_inches="tight")
    if also_save_panel_pdfs:
        stem, _ext = os.path.splitext(out_png)
        _export_figure_panels_to_pdfs(
            fig,
            [
                ("panel01_nmj_rate_by_folder", [ax_nmj_rate]),
                ("panel02_total_spots_by_folder", [ax_total_spots]),
                ("panel03_mean_radius_by_folder", [ax_radius]),
                ("panel04_mean_innervation_by_folder", [ax_overlap]),
                ("panel05_folder_medians_distance_space", [ax_distance]),
                (
                    "panel06_all_folders_proximity",
                    [ax_prox_title, ax_prox_kde_x, ax_prox_main, ax_prox_kde_y],
                ),
            ],
            stem,
        )
    fig.clf()
    plt.close(fig)
    return folder_stats


def posthoc_conover_iman(stats_df_spec, value_vars, block_col="File"):
    from scipy.stats import rankdata, t, friedmanchisquare

    # Extract only the columns of interest and drop any rows with missing values
    df_clean = stats_df_spec[[block_col] + value_vars].dropna()
    N = len(df_clean)
    k = len(value_vars)

    if N < 2 or k < 2:
        return None, None

    # Rank each row (block)
    ranks = np.zeros((N, k), dtype=float)
    for idx in range(N):
        row_vals = df_clean[value_vars].iloc[idx].to_numpy()
        ranks[idx, :] = rankdata(row_vals)

    # Sum of squares of all ranks
    A = np.sum(ranks ** 2)

    # Sum of squares of treatment totals divided by N
    R_j = np.sum(ranks, axis=0)
    B = np.sum(R_j ** 2) / N

    mean_ranks = R_j / N

    # Friedman statistic (T1) using scipy
    t1_stat, _ = friedmanchisquare(*[df_clean[col] for col in value_vars])

    df = (N - 1) * (k - 1)

    # Standard error
    denom = (2.0 * (A - B) / (N * (N - 1) * (k - 1))) * (1.0 - t1_stat / (N * (k - 1)))
    if denom <= 0:
        denom = 1e-15
    se = np.sqrt(denom)

    # Compute pairwise t-statistics and p-values
    results = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = np.abs(mean_ranks[i] - mean_ranks[j])
            t_stat = diff / se
            p_val = 2.0 * (1.0 - t.cdf(t_stat, df))

            g1 = value_vars[i].replace("Density_", "").replace("Abundance_", "")
            g2 = value_vars[j].replace("Density_", "").replace("Abundance_", "")

            results.append({
                "group1": g1,
                "group2": g2,
                "t_stat": float(t_stat),
                "p_val": float(p_val),
                "mean_rank1": float(mean_ranks[i]),
                "mean_rank2": float(mean_ranks[j])
            })

    df_results = pd.DataFrame(results)

    # Holm-Bonferroni correction
    m = len(df_results)
    sorted_indices = np.argsort(df_results["p_val"].to_numpy())
    adj_p = np.zeros(m)
    current_max = 0.0
    for rank_idx, orig_idx in enumerate(sorted_indices):
        raw_p = df_results.loc[orig_idx, "p_val"]
        adj = raw_p * (m - rank_idx)
        current_max = max(current_max, min(1.0, adj))
        adj_p[orig_idx] = current_max

    df_results["p_val_adj"] = adj_p

    def sig_stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    df_results["sig"] = df_results["p_val_adj"].apply(sig_stars)

    return df_results, t1_stat


def build_aggregate_batch_dashboard_figure(master_df, distance_threshold_um, *, run_all, all_file_stats):
    """Batch end-card figure: same layout as ``BTX_batch`` post-batch dashboard.

    ``all_file_stats`` is a list of dicts with keys ``File``, zone area/count columns, etc.
    (as produced during a live batch). If empty, the abundance panel shows a placeholder.

    Returns ``(fig, panel_specs, meta)`` where ``meta`` contains optional Streamlit messaging fields.
    """
    from scipy.stats import friedmanchisquare

    master_df = normalize_btx_signal_classes(master_df)

    if run_all:
        fig = plt.figure(figsize=(24, 40), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.01, wspace=0.01)
    else:
        fig = plt.figure(figsize=(20, 38), constrained_layout=True)
    outer = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.35])

    ax_scatter, ax_prox_kde_x, ax_prox_kde_y, ax_prox_title = proximity_joint_axes(
        fig, outer[0, 0], title_first=True, large_main_panel=True
    )
    ax_size_kde = fig.add_subplot(outer[0, 1])
    ax_circ_kde = fig.add_subplot(outer[1, 0])
    ax_overlap_kde = fig.add_subplot(outer[1, 1])
    ax_intensity_kde = fig.add_subplot(outer[2, 0])
    ax_control = None
    if run_all and "SOURCE_FOLDER" in master_df.columns and "SOURCE_IMAGE" in master_df.columns:
        ax_control = fig.add_subplot(outer[2, 1])
    if run_all:
        ax_abundance = fig.add_subplot(outer[3, 0])
    else:
        ax_abundance = fig.add_subplot(outer[2, 1])

    stats_df_spec = pd.DataFrame(all_file_stats or [])

    draw_proximity_joint(
        ax_scatter,
        ax_prox_kde_x,
        ax_prox_kde_y,
        master_df,
        distance_threshold_um,
        "Global NMJ Proximity Analysis",
        marginal_combined_black=True,
        title_ax=ax_prox_title,
    )

    if len(master_df) > 0:
        sns.kdeplot(
            data=master_df,
            x="RADIUS",
            hue="BTX signal class",
            hue_order=BTX_SIGNAL_CLASS_ORDER,
            palette=BTX_SIGNAL_CLASS_PALETTE,
            ax=ax_size_kde,
            common_norm=False,
            fill=True,
            warn_singular=False,
        )
    ax_size_kde.set_title("Global NMJ Size KDE")
    ax_size_kde.set_xlabel("Radius (μm)")
    ax_size_kde.set_ylabel("Probability Density")

    _roundness_order_global = list(ROUNDNESS_KRUSKAL_CLASSES)
    master_shape = dataframe_for_roundness_kde_and_kruskal(master_df)
    if len(master_shape) > 0:
        sns.kdeplot(
            data=master_shape,
            x="ROUNDNESS",
            hue="BTX signal class",
            hue_order=_roundness_order_global,
            palette=BTX_SIGNAL_CLASS_PALETTE,
            ax=ax_circ_kde,
            common_norm=False,
            fill=True,
            clip=(0, 1),
            warn_singular=False,
        )
    roundness_title_global = roundness_3way_kruskal_title(
        master_df,
        label_base="Global NMJ Roundness KDE (1 − eccentricity)",
    )
    ax_circ_kde.set_title(roundness_title_global)
    ax_circ_kde.set_xlabel("Roundness (1 = circle)")
    ax_circ_kde.set_ylabel("Probability Density")
    ax_circ_kde.set_xlim(0, 1)

    master_innervation = master_df[master_df["BTX signal class"] == "NMJ"] if len(master_df) > 0 else master_df
    if len(master_innervation) > 0:
        sns.histplot(
            data=master_innervation,
            x="INNERVATION_OVERLAP_PCT",
            color=BTX_SIGNAL_CLASS_PALETTE["NMJ"],
            ax=ax_overlap_kde,
        )
    ax_overlap_kde.set_title("Global NMJ Innervation Distribution")
    ax_overlap_kde.set_xlabel("NMJ Innervation (%)")
    ax_overlap_kde.set_ylabel("Count")

    if len(master_df) > 0:
        _int_vals = (
            master_df["MEAN_INTENSITY"].dropna()
            if "MEAN_INTENSITY" in master_df.columns
            else pd.Series(dtype=float)
        )
        _int_max = float(_int_vals.quantile(0.999)) if len(_int_vals) > 0 else None
        sns.kdeplot(
            data=master_df,
            x="MEAN_INTENSITY",
            hue="BTX signal class",
            hue_order=BTX_SIGNAL_CLASS_ORDER,
            palette=BTX_SIGNAL_CLASS_PALETTE,
            ax=ax_intensity_kde,
            common_norm=False,
            fill=True,
            warn_singular=False,
            clip=(0, _int_max) if _int_max is not None else None,
        )
        if _int_max is not None:
            ax_intensity_kde.set_xlim(0, _int_max * 1.05)
    intensity_title, intensity_summary = nmj_vs_orphan_intensity_mannwhitney_title(
        master_df,
        label_base="Global Receptor Intensity",
    )
    ax_intensity_kde.set_title(intensity_title)
    ax_intensity_kde.set_xlabel("Mean Fluorescence Intensity")
    ax_intensity_kde.set_ylabel("Probability Density")

    ax_abundance.clear()
    p_friedman_abundance = None
    conover_abundance_results = None
    if len(stats_df_spec) >= 1 and {"Area_NMJ_um2", "Area_Muscle_um2", "Area_Neuron_um2", "Area_Orphan_um2",
                                    "NMJs (Both)", "Near Aneural AChR clusters",
                                    "Near Neuron-associated BTX signal", "Orphaned"}.issubset(stats_df_spec.columns):
        total_area = (
            stats_df_spec["Area_NMJ_um2"] + 
            stats_df_spec["Area_Muscle_um2"] + 
            stats_df_spec["Area_Neuron_um2"] + 
            stats_df_spec["Area_Orphan_um2"]
        )
        total_area = np.where(total_area <= 0, 1.0, total_area)
        
        stats_df_spec["Abundance_NMJ"] = stats_df_spec["NMJs (Both)"] / total_area * 1000
        stats_df_spec["Abundance_Muscle"] = stats_df_spec["Near Aneural AChR clusters"] / total_area * 1000
        stats_df_spec["Abundance_Neuron"] = stats_df_spec["Near Neuron-associated BTX signal"] / total_area * 1000
        stats_df_spec["Abundance_Orphan"] = stats_df_spec["Orphaned"] / total_area * 1000
        
        has_nmj = "Density_NMJ" in stats_df_spec.columns
        if has_nmj:
            abundance_vars = ["Abundance_NMJ", "Abundance_Muscle", "Abundance_Neuron", "Abundance_Orphan"]
            palette_ab = ["red", "green", "blue", "gray"]
        else:
            abundance_vars = ["Abundance_Muscle", "Abundance_Neuron", "Abundance_Orphan"]
            palette_ab = ["red", "blue", "gray"]
            
        melt_abundance = stats_df_spec.melt(
            id_vars=["File"],
            value_vars=abundance_vars,
            var_name="Zone",
            value_name="Abundance",
        )
        melt_abundance["Zone"] = melt_abundance["Zone"].str.replace("Abundance_", "", regex=False)
        
        sns.boxplot(
            data=melt_abundance,
            x="Zone",
            y="Abundance",
            hue="Zone",
            palette=palette_ab,
            legend=False,
            ax=ax_abundance,
            showfliers=False,
        )
        sns.stripplot(
            data=melt_abundance, x="Zone", y="Abundance", color="black", alpha=0.4, jitter=True, ax=ax_abundance
        )
        
        abundance_title_str = "Global BTX Abundance (Spots / 1000 μm² total area)"
        if len(stats_df_spec) >= 2:
            try:
                if has_nmj:
                    _stat_friedman_ab, p_friedman_abundance = friedmanchisquare(
                        stats_df_spec["Abundance_NMJ"],
                        stats_df_spec["Abundance_Muscle"],
                        stats_df_spec["Abundance_Neuron"],
                        stats_df_spec["Abundance_Orphan"],
                    )
                else:
                    _stat_friedman_ab, p_friedman_abundance = friedmanchisquare(
                        stats_df_spec["Abundance_Muscle"],
                        stats_df_spec["Abundance_Neuron"],
                        stats_df_spec["Abundance_Orphan"],
                    )
                sig_star_ab = (
                    "***" if p_friedman_abundance < 0.001 else "**" if p_friedman_abundance < 0.01 else "*" if p_friedman_abundance < 0.05 else "ns"
                )
                abundance_title_str = f"Global BTX Abundance (Friedman P = {p_friedman_abundance:.4g} {sig_star_ab})"
                
                try:
                    conover_abundance_results, _ = posthoc_conover_iman(stats_df_spec, abundance_vars)
                    if conover_abundance_results is not None:
                        nmj_comps_ab = []
                        for _, row in conover_abundance_results.iterrows():
                            g1, g2 = row["group1"], row["group2"]
                            if "NMJ" in (g1, g2):
                                other = g2 if g1 == "NMJ" else g1
                                short_other = "Mus" if "Mus" in other else "Neu" if "Neu" in other else "Orp" if "Orp" in other else other
                                nmj_comps_ab.append(f"NMJ-{short_other}:{row['sig']}")
                        if nmj_comps_ab:
                            abundance_title_str += f"\nConover: " + ", ".join(nmj_comps_ab)
                except Exception:
                    pass
            except ValueError:
                abundance_title_str = "Global BTX Abundance (Insufficient Variance)"
                p_friedman_abundance = 1.0
                
        ax_abundance.set_title(abundance_title_str)
        ax_abundance.set_ylabel("Spots / 1000 μm²")
        ax_abundance.set_xlabel("Target Tissue Zone")
    else:
        ax_abundance.text(0.5, 0.5, "Insufficient images\nfor abundance test", ha="center", va="center")
        ax_abundance.set_axis_off()

    if ax_control is not None:
        per_image = (
            master_df.groupby(["SOURCE_FOLDER", "SOURCE_IMAGE"])
            .agg(total_spots=("is_NMJ", "size"), nmj_spots=("is_NMJ", "sum"))
            .reset_index()
        )
        per_image["nmj_rate_pct"] = np.where(
            per_image["total_spots"] > 0,
            per_image["nmj_spots"] / per_image["total_spots"] * 100.0,
            0.0,
        )
        sns.stripplot(
            data=per_image,
            x="SOURCE_FOLDER",
            y="nmj_rate_pct",
            color="black",
            alpha=0.65,
            jitter=0.25,
            ax=ax_control,
        )
        sns.pointplot(
            data=per_image,
            x="SOURCE_FOLDER",
            y="nmj_rate_pct",
            estimator=np.mean,
            errorbar="sd",
            linestyle="none",
            color="red",
            markers="D",
            markersize=7,
            linewidth=1.5,
            ax=ax_control,
        )
        ax_control.set_title("Per-Image NMJ Rate Control Chart")
        ax_control.set_xlabel("Folder")
        ax_control.set_ylabel("NMJ Rate (%)")
        ax_control.tick_params(axis="x", rotation=45)

    hist_panel_num = 8 if (run_all and ax_control is not None) else 7
    hist_axes, global_otsu_th = add_global_btx_intensity_histogram_with_otsu_row(
        fig,
        outer,
        4,
        master_df,
        panel_num=hist_panel_num,
    )

    panel_specs = [
        ("panel01_global_proximity", [ax_prox_title, ax_prox_kde_x, ax_scatter, ax_prox_kde_y]),
        ("panel02_global_size_kde", [ax_size_kde]),
        ("panel03_global_roundness_kde", [ax_circ_kde]),
        ("panel04_global_innervation", [ax_overlap_kde]),
        ("panel05_global_intensity", [ax_intensity_kde]),
        ("panel06_btx_abundance", [ax_abundance]),
    ]
    if run_all and ax_control is not None:
        panel_specs.append(("panel07_per_image_nmj_control", [ax_control]))
    panel_specs.append(("panel08_global_intensity_histogram_otsu", hist_axes))

    meta = {
        "intensity_summary": intensity_summary,
        "friedman_p_abundance": float(p_friedman_abundance) if p_friedman_abundance is not None else None,
        "conover_abundance_results": conover_abundance_results,
        "global_btx_intensity_otsu": float(global_otsu_th) if np.isfinite(global_otsu_th) else None,
    }
    return fig, panel_specs, meta


# --- MULTI-FORMAT IMAGE READING UTILITIES ---

SUPPORTED_EXTENSIONS = ('.czi', '.tif', '.tiff', '.lif', '.nd2', '.oir', '.poir')


def collect_image_jobs(target_dirs):
    """Recursively search for supported confocal and TIFF image files in target_dirs.

    Returns sorted list of (abs_dirpath, filename) pairs.
    """
    out = []
    for target_d in target_dirs:
        root = os.path.normpath(os.path.abspath(target_d))
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Exclude hidden directories in-place so os.walk doesn't traverse them
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for f in filenames:
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    out.append((os.path.normpath(os.path.abspath(dirpath)), f))
    out.sort(key=lambda t: (t[0].lower(), t[1].lower()))
    return out


def _oir_pixel_size_um(oir) -> float:
    """Pixel size (µm/px) from an open :class:`oirfile.OirFile`."""
    scales = getattr(oir, "coord_scales", {}) or {}
    units = getattr(oir, "coord_units", {}) or {}
    px_x = scales.get("X")
    if px_x is not None and units.get("X", "micrometer") == "micrometer":
        return float(px_x)
    px_y = scales.get("Y")
    if px_y is not None and units.get("Y", "micrometer") == "micrometer":
        return float(px_y)
    sizes = oir.sizes or {}
    if "X" in sizes and sizes["X"] > 0:
        plx = getattr(oir, "_pixel_length_x", 0.0)
        if plx > 0:
            return float(plx) / float(sizes["X"])
    return 1.0


def _oir_metadata_from_handle(oir):
    sizes = oir.sizes or {}
    if "Y" not in sizes or "X" not in sizes:
        raise ValueError("OIR file missing Y/X dimensions")
    num_channels = int(sizes.get("C", sizes.get("S", 1)))
    shape_yx = (int(sizes["Y"]), int(sizes["X"]))
    return num_channels, _oir_pixel_size_um(oir), shape_yx


def _oir_projected_cyx(oir):
    """Max-project T/L/Z and return a (C, Y, X) array."""
    data = np.asarray(oir.asarray())
    dims = tuple(oir.dims)
    reduce_axes = tuple(i for i, d in enumerate(dims) if d in ("T", "L", "Z"))
    if reduce_axes:
        data = np.max(data, axis=reduce_axes)
        dims = tuple(d for d in dims if d not in ("T", "L", "Z"))

    if data.ndim == 2:
        data = data[np.newaxis, ...]
        dims = ("C", "Y", "X")
    elif "C" not in dims and "S" not in dims:
        data = data[np.newaxis, ...]
        dims = ("C",) + dims

    chan_label = "C" if "C" in dims else "S"
    data = np.transpose(data, (dims.index(chan_label), dims.index("Y"), dims.index("X")))
    return np.ascontiguousarray(data)


def _load_oir_image(path, channel_indices=None):
    from oirfile import OirFile, PoirFile

    ext = os.path.splitext(path)[1].lower()
    if ext == ".poir":
        with PoirFile(path, squeeze=True) as archive:
            oir = next(iter(archive.values()))
            stack = _oir_projected_cyx(oir)
    else:
        with OirFile(path, squeeze=True) as oir:
            stack = _oir_projected_cyx(oir)

    if channel_indices is not None:
        wanted = list(dict.fromkeys(int(c) for c in channel_indices))
        h, w = stack.shape[1], stack.shape[2]
        return {
            c_idx: (stack[c_idx].copy() if c_idx < stack.shape[0] else np.zeros((h, w), dtype=stack.dtype))
            for c_idx in wanted
        }
    return stack


def get_confocal_metadata(path):
    """Unified function to extract channel count, pixel size (um/pixel), and YX shape.

    Supported formats: ``.czi`` (Zeiss), ``.nd2`` (Nikon), ``.lif`` (Leica),
    ``.oir``/``.poir`` (Olympus/Evident), ``.tif``/``.tiff``.

    Returns (num_channels, pixel_size_um, shape_yx).
    See also :func:`total_image_area_um2_from_metadata`.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.czi':
        import aicspylibczi
        czi = aicspylibczi.CziFile(path)
        dims = czi.get_dims_shape()[0]
        cc = dims.get("C", [0, 1])
        num_channels = int(cc[1]) - int(cc[0])

        pixel_size_um = 1.0
        try:
            for dist in czi.meta.findall(".//Distance"):
                if dist.attrib.get("Id") == "X":
                    val = dist.find("Value")
                    if val is not None:
                        pixel_size_um = float(val.text) * 1e6
                        break
        except Exception:
            pass

        shape_yx = None
        y_rng = dims.get("Y")
        x_rng = dims.get("X")
        if y_rng is not None and x_rng is not None and len(y_rng) >= 2 and len(x_rng) >= 2:
            shape_yx = (int(y_rng[1]) - int(y_rng[0]), int(x_rng[1]) - int(x_rng[0]))
        return num_channels, pixel_size_um, shape_yx

    elif ext == '.nd2':
        import nd2
        with nd2.ND2File(path) as ndfile:
            sizes = ndfile.sizes
            num_channels = sizes.get('C', 1)
            pixel_size_um = 1.0
            try:
                vx = ndfile.voxel_size()
                if vx is not None and hasattr(vx, 'x'):
                    pixel_size_um = float(vx.x)
            except Exception:
                pass
            shape_yx = (sizes.get('Y', 512), sizes.get('X', 512))
            return num_channels, pixel_size_um, shape_yx

    elif ext == '.lif':
        import readlif.reader
        lif_file = readlif.reader.LifFile(path)
        if not lif_file.img_list:
            raise ValueError(f"No image series found in LIF file: {path}")
        lif_image = lif_file.get_image(0)
        dims = lif_image.info['dims']
        num_channels = dims.c

        pixel_size_um = 1.0
        try:
            scale_tuple = lif_image.info.get('scale')
            if scale_tuple and scale_tuple[0] > 0:
                pixel_size_um = 1.0 / float(scale_tuple[0])
        except Exception:
            pass
        shape_yx = (dims.y, dims.x)
        return num_channels, pixel_size_um, shape_yx

    elif ext in (".oir", ".poir"):
        from oirfile import OirFile, PoirFile

        if ext == ".poir":
            with PoirFile(path, squeeze=True) as archive:
                oir = next(iter(archive.values()), None)
                if oir is None:
                    raise ValueError(f"No OIR entries found in POIR archive: {path}")
                return _oir_metadata_from_handle(oir)
        with OirFile(path, squeeze=True) as oir:
            return _oir_metadata_from_handle(oir)

    elif ext in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            shape = list(series.shape)
            ndim = len(shape)
            axes = getattr(series, 'axes', None)

            if not axes:
                if ndim == 2:
                    num_channels = 1
                    shape_yx = tuple(shape)
                elif ndim == 3:
                    if shape[0] <= 10 and shape[0] < shape[1] and shape[0] < shape[2]:
                        num_channels = shape[0]
                        shape_yx = (shape[1], shape[2])
                    elif shape[2] <= 10 and shape[2] < shape[0] and shape[2] < shape[1]:
                        num_channels = shape[2]
                        shape_yx = (shape[0], shape[1])
                    else:
                        num_channels = 1
                        shape_yx = (shape[1], shape[2])
                elif ndim == 4:
                    dim0, dim1, h, w = shape[0], shape[1], shape[2], shape[3]
                    if dim0 < dim1:
                        num_channels = dim0
                    else:
                        num_channels = dim1
                    shape_yx = (h, w)
                else:
                    num_channels = 1
                    shape_yx = (shape[-2], shape[-1])
            else:
                c_idx = axes.find('C')
                y_idx = axes.find('Y')
                x_idx = axes.find('X')
                num_channels = shape[c_idx] if c_idx != -1 else 1
                h = shape[y_idx] if y_idx != -1 else shape[-2]
                w = shape[x_idx] if x_idx != -1 else shape[-1]
                shape_yx = (h, w)

            pixel_size_um = 1.0
            try:
                page = tif.pages[0]
                tags = page.tags
                if 'XResolution' in tags and 'ResolutionUnit' in tags:
                    x_res = tags['XResolution'].value
                    unit = tags['ResolutionUnit'].value
                    if isinstance(x_res, tuple) and len(x_res) == 2 and x_res[1] > 0:
                        res_val = x_res[0] / x_res[1]
                    else:
                        res_val = float(x_res)

                    if res_val > 0:
                        if unit == 2:
                            pixel_size_um = 25400.0 / res_val
                        elif unit == 3:
                            pixel_size_um = 10000.0 / res_val
                        else:
                            pixel_size_um = 1.0 / res_val
            except Exception:
                pass
            return num_channels, pixel_size_um, shape_yx

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def total_image_area_um2_from_metadata(path):
    """Physical field-of-view area (µm²) from file metadata: height × width × pixel_size².

    Uses :func:`get_confocal_metadata` (``.czi``, ``.nd2``, ``.lif``, ``.oir``/``.poir``, ``.tif``/``.tiff``).
    """
    _num_channels, pixel_size_um, shape_yx = get_confocal_metadata(path)
    if not shape_yx or len(shape_yx) != 2:
        raise ValueError(f"Could not read image shape from metadata: {path}")
    if pixel_size_um is None or float(pixel_size_um) <= 0:
        raise ValueError(f"Could not read valid pixel size from metadata: {path}")
    height, width = int(shape_yx[0]), int(shape_yx[1])
    return float(height) * float(width) * float(pixel_size_um) ** 2


def _czi_channel_zmax_2d(czi, c_idx, dims0):
    """Return a single (Y, X) plane as Z-max projection with one Z plane in RAM at a time."""
    z_rng = dims0.get("Z", (0, 1))
    z0, z1 = int(z_rng[0]), int(z_rng[1])
    if z1 - z0 <= 1:
        img, _ = czi.read_image(C=c_idx)
        arr = np.squeeze(np.asarray(img))
        while arr.ndim > 2:
            arr = np.max(arr, axis=0)
        return arr.copy() if not arr.flags.owndata else arr

    acc = None
    for zi in range(z0, z1):
        img, _ = czi.read_image(C=c_idx, Z=zi)
        plane = np.squeeze(np.asarray(img))
        while plane.ndim > 2:
            plane = np.max(plane, axis=0)
        if acc is None:
            acc = plane.copy()
        else:
            np.maximum(acc, plane, out=acc)
        del img
    return acc


def load_confocal_image(path, channel_indices=None):
    """Unified function to load confocal or TIFF Z-stacks as 2D Z-max projections per channel.

    If channel_indices is set, returns dictionary {c_idx: 2D array}.
    Else returns 3D numpy array of shape (C, Y, X).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == '.czi':
        import aicspylibczi
        czi = aicspylibczi.CziFile(path)
        try:
            if channel_indices is not None:
                wanted = list(dict.fromkeys(int(c) for c in channel_indices))
                try:
                    dims0 = czi.get_dims_shape()[0]
                    channels = {}
                    for c_idx in wanted:
                        channels[c_idx] = _czi_channel_zmax_2d(czi, c_idx, dims0)
                    return channels
                except Exception:
                    pass

            img, _ = czi.read_image()
            img_sq = np.squeeze(img)
            if img_sq.ndim == 4:
                img_sq = np.max(img_sq, axis=1)
            if img_sq.ndim < 3:
                img_sq = np.expand_dims(img_sq, axis=0)
            if img_sq.ndim == 3 and img_sq.shape[-1] <= 10 and img_sq.shape[-1] < img_sq.shape[0]:
                img_sq = np.moveaxis(img_sq, -1, 0)

            if channel_indices is not None:
                wanted = list(dict.fromkeys(int(c) for c in channel_indices))
                return {c_idx: img_sq[c_idx].copy() for c_idx in wanted}
            return img_sq
        finally:
            if hasattr(czi, "close"):
                try:
                    czi.close()
                except Exception:
                    pass

    elif ext == '.nd2':
        import nd2
        with nd2.ND2File(path) as ndfile:
            axes = ndfile.axes
            sizes = ndfile.sizes
            data = ndfile.to_dask()

            c_pos = axes.find('C')
            z_pos = axes.find('Z')

            if channel_indices is not None:
                wanted = list(dict.fromkeys(int(c) for c in channel_indices))
                channels = {}
                for c_idx in wanted:
                    idx = [slice(None)] * len(axes)
                    if c_pos != -1:
                        idx[c_pos] = c_idx
                    chan_data = data[tuple(idx)]

                    new_axes = axes.replace('C', '')
                    new_z_pos = new_axes.find('Z')
                    if new_z_pos != -1:
                        chan_2d = chan_data.max(axis=new_z_pos).compute()
                    else:
                        chan_2d = chan_data.compute()

                    chan_2d = np.squeeze(np.asarray(chan_2d))
                    channels[c_idx] = chan_2d
                return channels
            else:
                if z_pos != -1:
                    projected = data.max(axis=z_pos).compute()
                else:
                    projected = data.compute()

                remaining_axes = axes.replace('Z', '')
                c_pos_new = remaining_axes.find('C')
                y_pos_new = remaining_axes.find('Y')
                x_pos_new = remaining_axes.find('X')

                order = []
                if c_pos_new != -1:
                    order.append(c_pos_new)
                else:
                    projected = np.expand_dims(projected, axis=0)
                    order.append(0)
                    y_pos_new += 1
                    x_pos_new += 1

                order.append(y_pos_new if y_pos_new != -1 else len(order))
                order.append(x_pos_new if x_pos_new != -1 else len(order))

                final_arr = np.transpose(projected, order)
                return np.squeeze(final_arr) if final_arr.ndim > 3 else final_arr

    elif ext == '.lif':
        import readlif.reader
        lif_file = readlif.reader.LifFile(path)
        if not lif_file.img_list:
            raise ValueError(f"No image series found in LIF file: {path}")
        lif_image = lif_file.get_image(0)
        dims = lif_image.info['dims']
        num_slices = dims.z
        num_channels = dims.c

        if channel_indices is not None:
            wanted = list(dict.fromkeys(int(c) for c in channel_indices))
            channels = {}
            for c_idx in wanted:
                acc = None
                for zi in range(num_slices):
                    img = lif_image.get_frame(z=zi, c=c_idx)
                    plane = np.array(img)
                    if acc is None:
                        acc = plane.copy()
                    else:
                        np.maximum(acc, plane, out=acc)
                channels[c_idx] = acc
            return channels
        else:
            all_ch = []
            for c_idx in range(num_channels):
                acc = None
                for zi in range(num_slices):
                    img = lif_image.get_frame(z=zi, c=c_idx)
                    plane = np.array(img)
                    if acc is None:
                        acc = plane.copy()
                    else:
                        np.maximum(acc, plane, out=acc)
                all_ch.append(acc)
            return np.stack(all_ch, axis=0)

    elif ext in (".oir", ".poir"):
        return _load_oir_image(path, channel_indices=channel_indices)

    elif ext in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            data = series.asarray()
            ndim = data.ndim
            axes = getattr(series, 'axes', None)

            if not axes:
                if ndim == 2:
                    std_data = data[np.newaxis, np.newaxis, :, :]
                elif ndim == 3:
                    shape = data.shape
                    if shape[0] <= 10 and shape[0] < shape[1] and shape[0] < shape[2]:
                        std_data = data[:, np.newaxis, :, :]
                    elif shape[2] <= 10 and shape[2] < shape[0] and shape[2] < shape[1]:
                        std_data = np.moveaxis(data, -1, 0)[:, np.newaxis, :, :]
                    else:
                        std_data = data[np.newaxis, :, :, :]
                elif ndim == 4:
                    dim0, dim1 = data.shape[0], data.shape[1]
                    if dim0 < dim1:
                        std_data = data
                    else:
                        std_data = np.moveaxis(data, 1, 0)
                else:
                    std_data = data.reshape((-1, 1, data.shape[-2], data.shape[-1]))
            else:
                c_idx = axes.find('C')
                z_idx = axes.find('Z')
                y_idx = axes.find('Y')
                x_idx = axes.find('X')

                src_indices = [c_idx, z_idx, y_idx, x_idx]
                temp_data = data.copy()
                final_axes_order = []
                for i, idx in enumerate(src_indices):
                    if idx == -1:
                        temp_data = np.expand_dims(temp_data, axis=-1)
                        final_axes_order.append(temp_data.ndim - 1)
                    else:
                        final_axes_order.append(idx)
                std_data = np.transpose(temp_data, final_axes_order)

            if channel_indices is not None:
                wanted = list(dict.fromkeys(int(c) for c in channel_indices))
                channels = {}
                for c_idx in wanted:
                    if c_idx < std_data.shape[0]:
                        chan_2d = np.max(std_data[c_idx], axis=0)
                        channels[c_idx] = chan_2d
                    else:
                        channels[c_idx] = np.zeros((std_data.shape[2], std_data.shape[3]))
                return channels
            else:
                all_ch = []
                for c_idx in range(std_data.shape[0]):
                    chan_2d = np.max(std_data[c_idx], axis=0)
                    all_ch.append(chan_2d)
                return np.stack(all_ch, axis=0)

