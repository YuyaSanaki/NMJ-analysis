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


def build_aggregate_batch_dashboard_figure(master_df, distance_threshold_um, *, run_all, all_file_stats):
    """Batch end-card figure: same layout as ``BTX_batch`` post-batch dashboard.

    ``all_file_stats`` is a list of dicts with keys ``File``, ``Density_Muscle``, ``Density_Neuron``,
    ``Density_Orphan`` (as produced during a live batch). If empty, panel 6 shows the
    insufficient-data placeholder.

    Returns ``(fig, panel_specs, meta)`` where ``meta`` contains optional Streamlit messaging fields.
    """
    from scipy.stats import friedmanchisquare

    master_df = normalize_btx_signal_classes(master_df)

    if run_all:
        fig = plt.figure(figsize=(24, 34), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.01, wspace=0.01)
        outer = fig.add_gridspec(4, 2)
    else:
        fig = plt.figure(figsize=(20, 24), constrained_layout=True)
        outer = fig.add_gridspec(3, 2)

    ax_scatter, ax_prox_kde_x, ax_prox_kde_y, ax_prox_title = proximity_joint_axes(
        fig, outer[0, 0], title_first=True, large_main_panel=run_all
    )
    ax_size_kde = fig.add_subplot(outer[0, 1])
    ax_circ_kde = fig.add_subplot(outer[1, 0])
    ax_overlap_kde = fig.add_subplot(outer[1, 1])
    ax_intensity_kde = fig.add_subplot(outer[2, 0])
    ax_spec = fig.add_subplot(outer[2, 1])

    draw_proximity_joint(
        ax_scatter,
        ax_prox_kde_x,
        ax_prox_kde_y,
        master_df,
        distance_threshold_um,
        "1. Global NMJ Proximity Analysis",
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
    ax_size_kde.set_title("2. Global NMJ Size KDE")
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
        label_base="3. Global NMJ Roundness KDE (1 − eccentricity)",
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
    ax_overlap_kde.set_title("4. Global NMJ Innervation Distribution")
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
        label_base="5. Global Receptor Intensity",
    )
    ax_intensity_kde.set_title(intensity_title)
    ax_intensity_kde.set_xlabel("Mean Fluorescence Intensity")
    ax_intensity_kde.set_ylabel("Probability Density")

    ax_spec.clear()
    stats_df_spec = pd.DataFrame(all_file_stats or [])
    p_friedman = None
    n_spec_images = 0
    if len(stats_df_spec) >= 2 and {"Density_Muscle", "Density_Neuron", "Density_Orphan"} <= set(
        stats_df_spec.columns
    ):
        n_spec_images = len(stats_df_spec)
        melt_df = stats_df_spec.melt(
            id_vars=["File"],
            value_vars=["Density_Muscle", "Density_Neuron", "Density_Orphan"],
            var_name="Zone",
            value_name="Density",
        )
        melt_df["Zone"] = melt_df["Zone"].str.replace("Density_", "", regex=False)
        try:
            _stat_friedman, p_friedman = friedmanchisquare(
                stats_df_spec["Density_Muscle"],
                stats_df_spec["Density_Neuron"],
                stats_df_spec["Density_Orphan"],
            )
            sig_star = (
                "***" if p_friedman < 0.001 else "**" if p_friedman < 0.01 else "*" if p_friedman < 0.05 else "ns"
            )
            title_str = f"6. BTX Enrichment (Friedman P = {p_friedman:.4g} {sig_star})"
        except ValueError:
            title_str = "6. BTX Enrichment (Insufficient Variance)"
            p_friedman = 1.0

        sns.boxplot(
            data=melt_df,
            x="Zone",
            y="Density",
            hue="Zone",
            palette=["red", "blue", "gray"],
            legend=False,
            ax=ax_spec,
            showfliers=False,
        )
        sns.stripplot(
            data=melt_df, x="Zone", y="Density", color="black", alpha=0.4, jitter=True, ax=ax_spec
        )
        ax_spec.set_title(title_str)
        ax_spec.set_ylabel("Spots / 1000 μm²")
        ax_spec.set_xlabel("Target Tissue Zone")
        if len(stats_df_spec) < 5:
            ax_spec.text(
                0.95,
                0.05,
                f"Low N ({len(stats_df_spec)}) limits power",
                transform=ax_spec.transAxes,
                ha="right",
                fontsize=9,
                alpha=0.7,
            )
    else:
        if len(stats_df_spec) > 0 and not {"Density_Muscle", "Density_Neuron", "Density_Orphan"} <= set(
            stats_df_spec.columns
        ):
            ax_spec.text(
                0.5,
                0.5,
                "Panel 6: density columns missing\n"
                "(need File, Density_Muscle,\nDensity_Neuron, Density_Orphan)",
                ha="center",
                va="center",
                fontsize=10,
            )
        else:
            ax_spec.text(0.5, 0.5, "Insufficient images\nfor specificity test", ha="center", va="center")
        ax_spec.set_axis_off()

    ax_control = None
    if run_all and "SOURCE_FOLDER" in master_df.columns and "SOURCE_IMAGE" in master_df.columns:
        ax_control = fig.add_subplot(outer[3, :])
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
        ax_control.set_title("7. Per-Image NMJ Rate Control Chart")
        ax_control.set_xlabel("Folder")
        ax_control.set_ylabel("NMJ Rate (%)")
        ax_control.tick_params(axis="x", rotation=45)

    panel_specs = [
        ("panel01_global_proximity", [ax_prox_title, ax_prox_kde_x, ax_scatter, ax_prox_kde_y]),
        ("panel02_global_size_kde", [ax_size_kde]),
        ("panel03_global_roundness_kde", [ax_circ_kde]),
        ("panel04_global_innervation", [ax_overlap_kde]),
        ("panel05_global_intensity", [ax_intensity_kde]),
        ("panel06_btx_enrichment", [ax_spec]),
    ]
    if run_all and ax_control is not None:
        panel_specs.append(("panel07_per_image_nmj_control", [ax_control]))

    meta = {
        "intensity_summary": intensity_summary,
        "friedman_p": float(p_friedman) if p_friedman is not None else None,
        "n_spec_images": int(n_spec_images),
    }
    return fig, panel_specs, meta
