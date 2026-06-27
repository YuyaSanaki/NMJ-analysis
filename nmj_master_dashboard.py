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

BTX_CLASS_EARLY_NMJ = "early NMJ-like"
BTX_CLASS_MUSCLE = "Muscle-associated"
BTX_CLASS_NEURON = "Neuron-associated"
BTX_CLASS_ORPHANED = "Orphaned"

BTX_SIGNAL_CLASS_ORDER = (
    BTX_CLASS_EARLY_NMJ,
    BTX_CLASS_MUSCLE,
    BTX_CLASS_NEURON,
    BTX_CLASS_ORPHANED,
)
BTX_SIGNAL_CLASS_PALETTE = {
    BTX_CLASS_EARLY_NMJ: "red",
    BTX_CLASS_MUSCLE: "green",
    BTX_CLASS_NEURON: "blue",
    BTX_CLASS_ORPHANED: "gray",
}

BTX_SIGNAL_CLASS_LEGACY_ALIASES = {
    "NMJ": BTX_CLASS_EARLY_NMJ,
    "Aneural AChR clusters": BTX_CLASS_MUSCLE,
    "Neuron-associated BTX signal": BTX_CLASS_NEURON,
    "Orphan": BTX_CLASS_ORPHANED,
    "Muscle Only": BTX_CLASS_MUSCLE,
    "Muscle only": BTX_CLASS_MUSCLE,
    "Neuron Only": BTX_CLASS_NEURON,
    "Neuron only": BTX_CLASS_NEURON,
}

BTX_CLASS_COMPARISON_4WAY = (
    f"{BTX_CLASS_EARLY_NMJ} vs {BTX_CLASS_MUSCLE} vs {BTX_CLASS_NEURON} vs {BTX_CLASS_ORPHANED}"
)
BTX_CLASS_COMPARISON_3WAY = (
    f"{BTX_CLASS_EARLY_NMJ} vs {BTX_CLASS_MUSCLE} vs {BTX_CLASS_NEURON}"
)
BTX_CLASS_COMPARISON_INTENSITY = f"{BTX_CLASS_EARLY_NMJ} vs {BTX_CLASS_ORPHANED}"
BTX_CLASS_COMPARISON_4WAY_ZONES = f"{BTX_CLASS_COMPARISON_4WAY} zones"

DENSITY_COL_EARLY_NMJ = "Density_early_NMJ_like"
DENSITY_COL_MUSCLE = "Density_Muscle_associated"
DENSITY_COL_NEURON = "Density_Neuron_associated"
DENSITY_COL_ORPHANED = "Density_Orphaned"

AREA_COL_EARLY_NMJ = "Area_early_NMJ_like_um2"
AREA_COL_MUSCLE = "Area_Muscle_associated_um2"
AREA_COL_NEURON = "Area_Neuron_associated_um2"
AREA_COL_ORPHANED = "Area_Orphaned_um2"

ABUNDANCE_COL_EARLY_NMJ = "Abundance_early_NMJ_like"
ABUNDANCE_COL_MUSCLE = "Abundance_Muscle_associated"
ABUNDANCE_COL_NEURON = "Abundance_Neuron_associated"
ABUNDANCE_COL_ORPHANED = "Abundance_Orphaned"

FILE_STATS_COLUMN_LEGACY_ALIASES = {
    "NMJs (Both)": BTX_CLASS_EARLY_NMJ,
    "Near Aneural AChR clusters": BTX_CLASS_MUSCLE,
    "Near Neuron-associated BTX signal": BTX_CLASS_NEURON,
    "Density_NMJ": DENSITY_COL_EARLY_NMJ,
    "Density_Muscle": DENSITY_COL_MUSCLE,
    "Density_Neuron": DENSITY_COL_NEURON,
    "Density_Orphan": DENSITY_COL_ORPHANED,
    "Area_NMJ_um2": AREA_COL_EARLY_NMJ,
    "Area_Muscle_um2": AREA_COL_MUSCLE,
    "Area_Neuron_um2": AREA_COL_NEURON,
    "Area_Orphan_um2": AREA_COL_ORPHANED,
    "Abundance_NMJ": ABUNDANCE_COL_EARLY_NMJ,
    "Abundance_Muscle": ABUNDANCE_COL_MUSCLE,
    "Abundance_Neuron": ABUNDANCE_COL_NEURON,
    "Abundance_Orphan": ABUNDANCE_COL_ORPHANED,
}

MIN_PIXELS_FOR_SHAPE = 20
RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL = 0.5
ROUNDNESS_KRUSKAL_CLASSES = (BTX_CLASS_EARLY_NMJ, BTX_CLASS_MUSCLE, BTX_CLASS_NEURON)


def normalize_file_stats_columns(df):
    if df is None or len(df) == 0:
        return df
    rename = {k: v for k, v in FILE_STATS_COLUMN_LEGACY_ALIASES.items() if k in df.columns}
    if not rename:
        return df
    return df.rename(columns=rename)


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


MASTER_RESULTS_LEADING_COLS = (
    "SOURCE_FOLDER",
    "SOURCE_IMAGE",
    "TOTAL_IMAGE_AREA_um2",
)


def prepare_spot_table_for_master(
    df,
    *,
    source_folder: str,
    source_image: str,
    total_image_area_um2: float,
):
    """Tag a per-image spot table for streaming into ``*_MASTER_RESULTS*.csv``."""
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    out["SOURCE_FOLDER"] = source_folder
    out["SOURCE_IMAGE"] = source_image
    out["TOTAL_IMAGE_AREA_um2"] = float(total_image_area_um2)
    cols = [c for c in MASTER_RESULTS_LEADING_COLS if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    return out[cols]


def finalize_master_results_dataframe(df, *, file_stats: list[dict] | None = None):
    """Normalize, backfill ``TOTAL_IMAGE_AREA_um2`` if needed, order columns, append global Otsu."""
    if df is None or len(df) == 0:
        return df
    out = ensure_roundness_column(normalize_btx_signal_classes(df))
    if "TOTAL_IMAGE_AREA_um2" not in out.columns:
        out["TOTAL_IMAGE_AREA_um2"] = np.nan
    if file_stats and {"SOURCE_FOLDER", "SOURCE_IMAGE"} <= set(out.columns):
        lookup = {
            (row.get("Folder"), row.get("File")): row.get("TOTAL_IMAGE_AREA_um2")
            for row in file_stats
            if row.get("TOTAL_IMAGE_AREA_um2") is not None
        }
        if lookup:
            missing = out["TOTAL_IMAGE_AREA_um2"].isna()
            if missing.any():
                out.loc[missing, "TOTAL_IMAGE_AREA_um2"] = out.loc[missing].apply(
                    lambda row: lookup.get((row["SOURCE_FOLDER"], row["SOURCE_IMAGE"]), np.nan),
                    axis=1,
                )
    cols = [c for c in MASTER_RESULTS_LEADING_COLS if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]
    return annotate_global_btx_intensity_otsu(out)


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

    nmj = df[df["BTX signal class"] == BTX_CLASS_EARLY_NMJ]["MEAN_INTENSITY"].dropna()
    orphan = df[df["BTX signal class"] == BTX_CLASS_ORPHANED]["MEAN_INTENSITY"].dropna()
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
        f"{BTX_CLASS_EARLY_NMJ}: {med_nmj:.2f} vs {BTX_CLASS_ORPHANED}: {med_orphan:.2f})"
    )
    summary = {
        "p_val": float(p_val),
        "nmj_median": med_nmj,
        "orphan_median": med_orphan,
        "fold_change": (med_nmj / med_orphan) if med_orphan > 0 else np.nan,
    }
    return title, summary


def roundness_3way_kruskal_title(df, label_base="3. Global early NMJ-like Roundness Analysis"):
    from scipy.stats import kruskal

    if df is None or len(df) == 0:
        return f"{label_base} (No Data)"
    required = {"AREA_PX", "BTX signal class", "ROUNDNESS"}
    if not required <= set(df.columns):
        return f"{label_base} (Missing Columns)"

    df_valid = dataframe_for_roundness_kde_and_kruskal(df)

    g_nmj = df_valid[df_valid["BTX signal class"] == BTX_CLASS_EARLY_NMJ]["ROUNDNESS"]
    g_aneural = df_valid[df_valid["BTX signal class"] == BTX_CLASS_MUSCLE]["ROUNDNESS"]
    g_neuron = df_valid[df_valid["BTX signal class"] == BTX_CLASS_NEURON]["ROUNDNESS"]

    if any(len(g) < 3 for g in [g_nmj, g_aneural, g_neuron]):
        return f"{label_base} (Insufficient group sizes for 3-way test)"

    try:
        _stat, p_val = kruskal(g_nmj, g_aneural, g_neuron)
    except ValueError:
        return f"{label_base} (Test Failed)"

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    return (
        f"{label_base} (Kruskal P = {p_val:.4g} {sig})\n"
        f"Medians - {BTX_CLASS_EARLY_NMJ}: {g_nmj.median():.2f} | {BTX_CLASS_MUSCLE}: {g_aneural.median():.2f} | "
        f"{BTX_CLASS_NEURON}: {g_neuron.median():.2f}"
    )


BTX_SIGNAL_CLASS_HISTOGRAM_LABELS = {
    BTX_CLASS_EARLY_NMJ: BTX_CLASS_EARLY_NMJ,
    BTX_CLASS_MUSCLE: BTX_CLASS_MUSCLE,
    BTX_CLASS_NEURON: BTX_CLASS_NEURON,
    BTX_CLASS_ORPHANED: BTX_CLASS_ORPHANED,
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


def _sig_stars(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


BTX_CLASS_TO_ABUNDANCE_COL = {
    BTX_CLASS_EARLY_NMJ: ABUNDANCE_COL_EARLY_NMJ,
    BTX_CLASS_MUSCLE: ABUNDANCE_COL_MUSCLE,
    BTX_CLASS_NEURON: ABUNDANCE_COL_NEURON,
    BTX_CLASS_ORPHANED: ABUNDANCE_COL_ORPHANED,
}

ZONE_ABUNDANCE_AREA_COLS = (
    AREA_COL_EARLY_NMJ,
    AREA_COL_MUSCLE,
    AREA_COL_NEURON,
    AREA_COL_ORPHANED,
)
ZONE_ABUNDANCE_COUNT_COLS = (
    BTX_CLASS_EARLY_NMJ,
    BTX_CLASS_MUSCLE,
    BTX_CLASS_NEURON,
    BTX_CLASS_ORPHANED,
)
ZONE_ABUNDANCE_VALUE_COLS = (
    ABUNDANCE_COL_EARLY_NMJ,
    ABUNDANCE_COL_MUSCLE,
    ABUNDANCE_COL_NEURON,
    ABUNDANCE_COL_ORPHANED,
)


def _append_zone_abundance_columns(stats_df_spec):
    """Add per-image zone abundance (spots / 1000 µm² total mask area) columns."""
    if stats_df_spec is None or len(stats_df_spec) == 0:
        return pd.DataFrame()
    df = normalize_file_stats_columns(stats_df_spec.copy())
    if not set(ZONE_ABUNDANCE_AREA_COLS) <= set(df.columns):
        return df
    if not set(ZONE_ABUNDANCE_COUNT_COLS) <= set(df.columns):
        return df
    total_area = (
        df[AREA_COL_EARLY_NMJ]
        + df[AREA_COL_MUSCLE]
        + df[AREA_COL_NEURON]
        + df[AREA_COL_ORPHANED]
    )
    total_area = np.where(total_area <= 0, 1.0, total_area)
    df[ABUNDANCE_COL_EARLY_NMJ] = df[BTX_CLASS_EARLY_NMJ] / total_area * 1000
    df[ABUNDANCE_COL_MUSCLE] = df[BTX_CLASS_MUSCLE] / total_area * 1000
    df[ABUNDANCE_COL_NEURON] = df[BTX_CLASS_NEURON] / total_area * 1000
    df[ABUNDANCE_COL_ORPHANED] = df[BTX_CLASS_ORPHANED] / total_area * 1000
    return df


def build_otsu_thresholded_abundance_stats(master_df, all_file_stats, otsu_th):
    """Per-image zone abundance using only spots with ``MEAN_INTENSITY >=`` global Otsu."""
    base = normalize_file_stats_columns(pd.DataFrame(all_file_stats or []))
    if base.empty or not np.isfinite(otsu_th):
        return pd.DataFrame()
    if not {"File", *ZONE_ABUNDANCE_AREA_COLS} <= set(base.columns):
        return pd.DataFrame()
    if master_df is None or len(master_df) == 0:
        return pd.DataFrame()
    if "MEAN_INTENSITY" not in master_df.columns or "SOURCE_IMAGE" not in master_df.columns:
        return pd.DataFrame()
    if "BTX signal class" not in master_df.columns:
        return pd.DataFrame()

    filtered = master_df[master_df["MEAN_INTENSITY"] >= float(otsu_th)]
    if len(filtered) == 0:
        return pd.DataFrame()

    counts = (
        filtered.groupby(["SOURCE_IMAGE", "BTX signal class"]).size().unstack(fill_value=0)
    )
    out = base[["File"]].copy()
    total_area = (
        base[AREA_COL_EARLY_NMJ]
        + base[AREA_COL_MUSCLE]
        + base[AREA_COL_NEURON]
        + base[AREA_COL_ORPHANED]
    )
    total_area = np.where(total_area <= 0, 1.0, total_area)

    for btx_class, ab_col in BTX_CLASS_TO_ABUNDANCE_COL.items():
        if btx_class in counts.columns:
            class_counts = base["File"].map(counts[btx_class]).fillna(0)
        else:
            class_counts = pd.Series(0.0, index=base.index)
        out[ab_col] = class_counts / total_area * 1000

    return out


def draw_zone_btx_abundance_panel(ax, stats_df_spec, *, title_base, include_nmj_zone=True):
    """Box/strip plot of zone abundance; returns ``(title, p_friedman, conover_df)``."""
    from scipy.stats import friedmanchisquare

    ax.clear()
    p_friedman_abundance = None
    conover_abundance_results = None

    if stats_df_spec is None or len(stats_df_spec) < 1:
        ax.text(0.5, 0.5, "Insufficient images\nfor abundance test", ha="center", va="center")
        ax.set_axis_off()
        return title_base, None, None

    stats_df_spec = _append_zone_abundance_columns(stats_df_spec)
    if include_nmj_zone and ABUNDANCE_COL_EARLY_NMJ in stats_df_spec.columns:
        abundance_vars = list(ZONE_ABUNDANCE_VALUE_COLS)
        palette_ab = ["red", "green", "blue", "gray"]
    else:
        abundance_vars = [ABUNDANCE_COL_MUSCLE, ABUNDANCE_COL_NEURON, ABUNDANCE_COL_ORPHANED]
        palette_ab = ["green", "blue", "gray"]

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
        ax=ax,
        showfliers=False,
    )
    sns.stripplot(
        data=melt_abundance,
        x="Zone",
        y="Abundance",
        color="black",
        alpha=0.4,
        jitter=True,
        ax=ax,
    )

    abundance_title_str = title_base
    if len(stats_df_spec) >= 2:
        try:
            if include_nmj_zone and ABUNDANCE_COL_EARLY_NMJ in stats_df_spec.columns:
                _stat_friedman_ab, p_friedman_abundance = friedmanchisquare(
                    stats_df_spec[ABUNDANCE_COL_EARLY_NMJ],
                    stats_df_spec[ABUNDANCE_COL_MUSCLE],
                    stats_df_spec[ABUNDANCE_COL_NEURON],
                    stats_df_spec[ABUNDANCE_COL_ORPHANED],
                )
            else:
                _stat_friedman_ab, p_friedman_abundance = friedmanchisquare(
                    stats_df_spec[ABUNDANCE_COL_MUSCLE],
                    stats_df_spec[ABUNDANCE_COL_NEURON],
                    stats_df_spec[ABUNDANCE_COL_ORPHANED],
                )
            sig_star_ab = _sig_stars(p_friedman_abundance)
            abundance_title_str = f"{title_base}\n(Friedman P = {p_friedman_abundance:.4g} {sig_star_ab})"
            try:
                conover_abundance_results, _ = posthoc_conover_iman(stats_df_spec, abundance_vars)
                if conover_abundance_results is not None:
                    nmj_comps_ab = []
                    for _, row in conover_abundance_results.iterrows():
                        g1, g2 = row["group1"], row["group2"]
                        early_nmj_key = ABUNDANCE_COL_EARLY_NMJ.replace("Abundance_", "")
                        if early_nmj_key in (g1, g2):
                            other = g2 if g1 == early_nmj_key else g1
                            short_other = (
                                "Mus" if "Mus" in other
                                else "Neu" if "Neu" in other
                                else "Orp" if "Orp" in other
                                else other
                            )
                            nmj_comps_ab.append(f"{BTX_CLASS_EARLY_NMJ}-{short_other}:{row['sig']}")
                    if nmj_comps_ab:
                        abundance_title_str += "\nConover: " + ", ".join(nmj_comps_ab)
            except Exception:
                pass
        except ValueError:
            abundance_title_str = f"{title_base}\n(Insufficient Variance)"
            p_friedman_abundance = 1.0

    ax.set_title(abundance_title_str)
    ax.set_ylabel("Spots / 1000 μm²")
    ax.set_xlabel("Target Tissue Zone")
    return abundance_title_str, p_friedman_abundance, conover_abundance_results


def draw_global_intensity_kde_panel(ax, master_df, *, title, otsu_th=None, otsu_filtered=False):
    """KDE of spot ``MEAN_INTENSITY``; optionally keep only spots at/above global Otsu."""
    ax.clear()
    plot_df = master_df
    if plot_df is None or len(plot_df) == 0:
        ax.text(0.5, 0.5, "No intensity data", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return title, None

    if otsu_filtered and np.isfinite(otsu_th):
        plot_df = plot_df[plot_df["MEAN_INTENSITY"] >= float(otsu_th)]
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, "No spots after Otsu filter", ha="center", va="center")
        ax.set_title(title)
        ax.set_axis_off()
        return title, None

    _int_vals = plot_df["MEAN_INTENSITY"].dropna()
    _int_max = float(_int_vals.quantile(0.999)) if len(_int_vals) > 0 else None
    sns.kdeplot(
        data=plot_df,
        x="MEAN_INTENSITY",
        hue="BTX signal class",
        hue_order=BTX_SIGNAL_CLASS_ORDER,
        palette=BTX_SIGNAL_CLASS_PALETTE,
        ax=ax,
        common_norm=False,
        fill=True,
        warn_singular=False,
        clip=(0, _int_max) if _int_max is not None else None,
    )
    if _int_max is not None:
        ax.set_xlim(0, _int_max * 1.05)
    if otsu_filtered and np.isfinite(otsu_th):
        _draw_otsu_vline(ax, otsu_th)

    intensity_summary = None
    if otsu_filtered:
        panel_title = title
    else:
        panel_title, intensity_summary = nmj_vs_orphan_intensity_mannwhitney_title(
            plot_df,
            label_base=title,
        )
    ax.set_title(panel_title)
    ax.set_xlabel("Mean Fluorescence Intensity")
    ax.set_ylabel("Probability Density")
    return panel_title, intensity_summary


def _roundness_kruskal_result(df):
    from scipy.stats import kruskal

    if df is None or len(df) == 0:
        return None
    df_valid = dataframe_for_roundness_kde_and_kruskal(df)
    g_nmj = df_valid[df_valid["BTX signal class"] == BTX_CLASS_EARLY_NMJ]["ROUNDNESS"]
    g_aneural = df_valid[df_valid["BTX signal class"] == BTX_CLASS_MUSCLE]["ROUNDNESS"]
    g_neuron = df_valid[df_valid["BTX signal class"] == BTX_CLASS_NEURON]["ROUNDNESS"]
    if any(len(g) < 3 for g in [g_nmj, g_aneural, g_neuron]):
        return None
    try:
        h_stat, p_val = kruskal(g_nmj, g_aneural, g_neuron)
    except ValueError:
        return None
    return {
        "statistic_name": "H",
        "statistic_value": float(h_stat),
        "p_value": float(p_val),
        "n_nmj": int(len(g_nmj)),
        "n_aneural": int(len(g_aneural)),
        "n_neuron": int(len(g_neuron)),
    }


def _intensity_mannwhitney_result(df):
    from scipy.stats import mannwhitneyu

    if df is None or len(df) == 0:
        return None
    nmj = df[df["BTX signal class"] == BTX_CLASS_EARLY_NMJ]["MEAN_INTENSITY"].dropna()
    orphan = df[df["BTX signal class"] == BTX_CLASS_ORPHANED]["MEAN_INTENSITY"].dropna()
    if len(nmj) < 3 or len(orphan) < 3:
        return None
    try:
        u_stat, p_val = mannwhitneyu(nmj, orphan, alternative="greater")
    except ValueError:
        return None
    return {
        "statistic_name": "U",
        "statistic_value": float(u_stat),
        "p_value": float(p_val),
        "n_nmj": int(len(nmj)),
        "n_orphan": int(len(orphan)),
        "nmj_median": float(nmj.median()),
        "orphan_median": float(orphan.median()),
    }


STAT_SUMMARY_COLUMNS = (
    "level",
    "folder",
    "file",
    "metric",
    "comparison",
    "test",
    "test_design",
    "statistic",
    "statistic_value",
    "p_value",
    "p_value_adjusted",
    "significance",
    "notes",
)


def build_batch_stat_summary_dataframe(
    master_df,
    *,
    distance_threshold_um,
    dash_meta,
    run_all,
):
    """Primary image-level tests + exploratory spot-pooled tests.

    Returns ``(stat_df, image_medians_df, otsu_dim_noise_df)``.
    """
    rows = []
    image_medians_df = build_per_image_all_class_medians_table(master_df)

    def add_row(**kwargs):
        row = {col: kwargs.get(col) for col in STAT_SUMMARY_COLUMNS}
        rows.append(row)

    scope = "all_folders" if run_all else "current_folder"
    otsu_th = dash_meta.get("global_btx_intensity_otsu") if dash_meta else None
    otsu_label = f"{float(otsu_th):.1f} A.U." if otsu_th is not None and np.isfinite(otsu_th) else "n/a"
    n_images = (
        int(master_df[_image_group_columns(master_df)].drop_duplicates().shape[0])
        if _image_group_columns(master_df) is not None
        else 0
    )

    # --- Primary inference: per-image class medians (unit of replication = image) ---
    img_prox = image_level_proximity_analysis_results(master_df)
    if img_prox is not None:
        for axis_key, axis_label in (("muscle_kruskal", "muscle mask"), ("neuron_kruskal", "neuron mask")):
            kw = img_prox.get(axis_key)
            if kw is None:
                continue
            med = kw["medians"]
            add_row(
                level="primary_image_level",
                folder=scope,
                file="",
                metric=f"Proximity — distance to {axis_label} (per-image class median)",
                comparison=BTX_CLASS_COMPARISON_4WAY,
                test="Kruskal-Wallis",
                test_design=(
                    f"Image-level class medians; NMJ boundary ≤ {distance_threshold_um} μm; "
                    f"n_images={n_images}"
                ),
                statistic=kw["statistic_name"],
                statistic_value=kw["statistic_value"],
                p_value=kw["p_value"],
                p_value_adjusted="",
                significance=_sig_stars(kw["p_value"]),
                notes=(
                    "Median of per-image medians (μm): "
                    f"{BTX_CLASS_EARLY_NMJ}={med.get(BTX_CLASS_EARLY_NMJ, float('nan')):.3f}, "
                    f"{BTX_CLASS_MUSCLE}={med.get(BTX_CLASS_MUSCLE, float('nan')):.3f}, "
                    f"{BTX_CLASS_NEURON}={med.get(BTX_CLASS_NEURON, float('nan')):.3f}, "
                    f"{BTX_CLASS_ORPHANED}={med.get(BTX_CLASS_ORPHANED, float('nan')):.3f}"
                ),
            )
        for axis_label, posthoc_df in (
            ("muscle mask", img_prox.get("posthoc_muscle")),
            ("neuron mask", img_prox.get("posthoc_neuron")),
        ):
            if posthoc_df is None or len(posthoc_df) == 0:
                continue
            for _, row in posthoc_df.iterrows():
                add_row(
                    level="primary_posthoc",
                    folder=scope,
                    file="",
                    metric=f"Proximity — distance to {axis_label} (per-image class median)",
                    comparison=f"{row['group1']} vs {row['group2']}",
                    test="Mann-Whitney U",
                    test_design=f"pairwise {BTX_CLASS_EARLY_NMJ} vs other class on image medians; Holm–Bonferroni adjusted",
                    statistic="U",
                    statistic_value=row.get("u_stat"),
                    p_value=row.get("p_val"),
                    p_value_adjusted=row.get("p_val_adj"),
                    significance=row.get("sig"),
                    notes="Follow-up to image-level Kruskal–Wallis on proximity axis",
                )

    img_round = image_level_roundness_kruskal_result(master_df)
    if img_round is not None:
        add_row(
            level="primary_image_level",
            folder=scope,
            file="",
            metric="Spot roundness (per-image class median)",
            comparison=BTX_CLASS_COMPARISON_3WAY,
            test="Kruskal-Wallis",
            test_design="Image-level medians; spots with AREA_PX ≥ MIN_PIXELS_FOR_SHAPE",
            statistic=img_round["statistic_name"],
            statistic_value=img_round["statistic_value"],
            p_value=img_round["p_value"],
            p_value_adjusted="",
            significance=_sig_stars(img_round["p_value"]),
            notes=f"n_images={img_round.get('n_images', n_images)}",
        )

    img_int_paired = image_level_intensity_paired_nmj_vs_orphan(master_df)
    if img_int_paired is not None:
        add_row(
            level="primary_image_level",
            folder=scope,
            file="",
            metric="BTX intensity (paired within image, class medians)",
            comparison=BTX_CLASS_COMPARISON_INTENSITY,
            test="Wilcoxon signed-rank",
            test_design=(
                "one-sided (greater): per-image early NMJ-like median vs Orphaned median "
                "in images containing both classes"
            ),
            statistic=img_int_paired["statistic_name"],
            statistic_value=img_int_paired["statistic_value"],
            p_value=img_int_paired["p_value"],
            p_value_adjusted="",
            significance=_sig_stars(img_int_paired["p_value"]),
            notes=(
                f"n_paired_images={img_int_paired['n_paired_images']}; "
                f"{BTX_CLASS_EARLY_NMJ} brighter in {img_int_paired['nmj_brighter_count']} images; "
                f"medians of image medians {img_int_paired['nmj_median_of_medians']:.2f} vs "
                f"{img_int_paired['orphan_median_of_medians']:.2f} A.U."
            ),
        )

    img_int = image_level_intensity_nmj_vs_orphan(master_df)
    if img_int is not None:
        add_row(
            level="primary_sensitivity",
            folder=scope,
            file="",
            metric="BTX intensity (unpaired image medians, all spots)",
            comparison=BTX_CLASS_COMPARISON_INTENSITY,
            test="Mann-Whitney U",
            test_design="sensitivity: unpaired one-sided (greater) on image-level medians",
            statistic=img_int["statistic_name"],
            statistic_value=img_int["statistic_value"],
            p_value=img_int["p_value"],
            p_value_adjusted="",
            significance=_sig_stars(img_int["p_value"]),
            notes=(
                f"n_images {BTX_CLASS_EARLY_NMJ}={img_int['n_nmj_images']}, "
                f"{BTX_CLASS_ORPHANED}={img_int['n_orphan_images']}; "
                f"medians {img_int['nmj_median']:.2f} vs {img_int['orphan_median']:.2f} A.U.; "
                "use paired Wilcoxon row for primary inference"
            ),
        )

    otsu_dim_noise_df = build_otsu_dim_noise_rejection_table(master_df, otsu_th=otsu_th)
    if otsu_th is not None and np.isfinite(otsu_th):
        img_frac_paired = image_level_paired_frac_above_otsu_nmj_vs_orphan(master_df, otsu_th)
        if img_frac_paired is not None:
            add_row(
                level="primary_image_level",
                folder=scope,
                file="",
                metric=f"Fraction above Otsu {otsu_label} (paired within image)",
                comparison=BTX_CLASS_COMPARISON_INTENSITY,
                test="Wilcoxon signed-rank",
                test_design=(
                    "one-sided (greater): per-image fraction of early NMJ-like spots ≥ Otsu vs "
                    "Orphaned fraction in images with both classes"
                ),
                statistic=img_frac_paired["statistic_name"],
                statistic_value=img_frac_paired["statistic_value"],
                p_value=img_frac_paired["p_value"],
                p_value_adjusted="",
                significance=_sig_stars(img_frac_paired["p_value"]),
                notes=(
                    f"n_paired_images={img_frac_paired['n_paired_images']}; "
                    f"higher in {img_frac_paired['nmj_higher_frac_count']} images; "
                    f"median fractions {img_frac_paired['nmj_median_frac']:.3f} vs "
                    f"{img_frac_paired['orphan_median_frac']:.3f}"
                ),
            )

        img_int_paired_otsu = image_level_intensity_paired_nmj_vs_orphan(master_df, otsu_th=otsu_th)
        if img_int_paired_otsu is not None:
            add_row(
                level="primary_image_level",
                folder=scope,
                file="",
                metric=f"BTX intensity (paired within image, spots ≥ Otsu {otsu_label})",
                comparison=BTX_CLASS_COMPARISON_INTENSITY,
                test="Wilcoxon signed-rank",
                test_design="one-sided (greater) on paired image medians after Otsu spot filter",
                statistic=img_int_paired_otsu["statistic_name"],
                statistic_value=img_int_paired_otsu["statistic_value"],
                p_value=img_int_paired_otsu["p_value"],
                p_value_adjusted="",
                significance=_sig_stars(img_int_paired_otsu["p_value"]),
                notes=(
                    f"n_paired_images={img_int_paired_otsu['n_paired_images']}; "
                    f"{BTX_CLASS_EARLY_NMJ} brighter in {img_int_paired_otsu['nmj_brighter_count']} images"
                ),
            )

        img_int_otsu = image_level_intensity_nmj_vs_orphan(master_df, otsu_th=otsu_th)
        if img_int_otsu is not None:
            add_row(
                level="primary_sensitivity",
                folder=scope,
                file="",
                metric=f"BTX intensity (unpaired image medians, spots ≥ Otsu {otsu_label})",
                comparison=BTX_CLASS_COMPARISON_INTENSITY,
                test="Mann-Whitney U",
                test_design="sensitivity: unpaired one-sided (greater) after Otsu spot filter",
                statistic=img_int_otsu["statistic_name"],
                statistic_value=img_int_otsu["statistic_value"],
                p_value=img_int_otsu["p_value"],
                p_value_adjusted="",
                significance=_sig_stars(img_int_otsu["p_value"]),
                notes=(
                    f"n_images {BTX_CLASS_EARLY_NMJ}={img_int_otsu['n_nmj_images']}, "
                    f"{BTX_CLASS_ORPHANED}={img_int_otsu['n_orphan_images']}; "
                    "use paired Wilcoxon row for primary inference"
                ),
            )

    p_friedman_all = dash_meta.get("friedman_p_abundance") if dash_meta else None
    if p_friedman_all is not None:
        add_row(
            level="primary_image_level",
            folder=scope,
            file="",
            metric="Zone BTX spot abundance (all detected spots)",
            comparison=BTX_CLASS_COMPARISON_4WAY_ZONES,
            test="Friedman",
            test_design="repeated measures per image; abundance = spots / 1000 μm² total mask area",
            statistic="chi-square",
            statistic_value="",
            p_value=p_friedman_all,
            p_value_adjusted="",
            significance=_sig_stars(p_friedman_all),
            notes="Per-image zone counts from EDT classification at NMJ boundary",
        )

    conover_all = dash_meta.get("conover_abundance_results") if dash_meta else None
    if conover_all is not None and len(conover_all) > 0:
        for _, row in conover_all.iterrows():
            add_row(
                level="primary_posthoc",
                folder=scope,
                file="",
                metric="Zone BTX spot abundance (all detected spots)",
                comparison=f"{row['group1']} vs {row['group2']}",
                test="Conover-Iman",
                test_design="pairwise post-hoc after Friedman; Holm–Bonferroni adjusted",
                statistic="t",
                statistic_value=row.get("t_stat"),
                p_value=row.get("p_val"),
                p_value_adjusted=row.get("p_val_adj"),
                significance=row.get("sig"),
                notes="Follow-up to per-image Friedman abundance test",
            )

    p_friedman_otsu = dash_meta.get("friedman_p_abundance_otsu") if dash_meta else None
    if p_friedman_otsu is not None:
        add_row(
            level="primary_image_level",
            folder=scope,
            file="",
            metric=f"Zone BTX spot abundance (spots ≥ global Otsu {otsu_label})",
            comparison=BTX_CLASS_COMPARISON_4WAY_ZONES,
            test="Friedman",
            test_design="repeated measures per image; Otsu-filtered counts / 1000 μm²",
            statistic="chi-square",
            statistic_value="",
            p_value=p_friedman_otsu,
            p_value_adjusted="",
            significance=_sig_stars(p_friedman_otsu),
            notes=f"Only spots with MEAN_INTENSITY ≥ {otsu_label}",
        )

    conover_otsu = dash_meta.get("conover_abundance_otsu_results") if dash_meta else None
    if conover_otsu is not None and len(conover_otsu) > 0:
        for _, row in conover_otsu.iterrows():
            add_row(
                level="primary_posthoc",
                folder=scope,
                file="",
                metric=f"Zone BTX spot abundance (spots ≥ global Otsu {otsu_label})",
                comparison=f"{row['group1']} vs {row['group2']}",
                test="Conover-Iman",
                test_design="pairwise post-hoc after Friedman; Holm–Bonferroni adjusted",
                statistic="t",
                statistic_value=row.get("t_stat"),
                p_value=row.get("p_val"),
                p_value_adjusted=row.get("p_val_adj"),
                significance=row.get("sig"),
                notes="Follow-up to Otsu-filtered Friedman abundance test",
            )

    # --- Exploratory: all spots pooled (visualization support; not primary inference) ---
    proximity = proximity_analysis_results(master_df)
    if proximity is not None:
        muscle_kw = proximity.get("muscle_kruskal")
        if muscle_kw is not None:
            med = muscle_kw["medians"]
            add_row(
                level="exploratory_spot_pooled",
                folder=scope,
                file="",
                metric="Proximity — distance to muscle mask (spot edge)",
                comparison=BTX_CLASS_COMPARISON_4WAY,
                test="Kruskal-Wallis",
                test_design="EXPLORATORY: all spots pooled (non-independent within image)",
                statistic=muscle_kw["statistic_name"],
                statistic_value=muscle_kw["statistic_value"],
                p_value=muscle_kw["p_value"],
                p_value_adjusted="",
                significance=_sig_stars(muscle_kw["p_value"]),
                notes=(
                    "Spot-pooled medians (μm): "
                    f"{BTX_CLASS_EARLY_NMJ}={med.get(BTX_CLASS_EARLY_NMJ, float('nan')):.3f}, "
                    f"{BTX_CLASS_MUSCLE}={med.get(BTX_CLASS_MUSCLE, float('nan')):.3f}"
                ),
            )
        neuron_kw = proximity.get("neuron_kruskal")
        if neuron_kw is not None:
            med = neuron_kw["medians"]
            add_row(
                level="exploratory_spot_pooled",
                folder=scope,
                file="",
                metric="Proximity — distance to neuron mask (spot edge)",
                comparison=BTX_CLASS_COMPARISON_4WAY,
                test="Kruskal-Wallis",
                test_design="EXPLORATORY: all spots pooled (non-independent within image)",
                statistic=neuron_kw["statistic_name"],
                statistic_value=neuron_kw["statistic_value"],
                p_value=neuron_kw["p_value"],
                p_value_adjusted="",
                significance=_sig_stars(neuron_kw["p_value"]),
                notes="For plot visualization only; use primary_image_level rows for inference",
            )

    roundness = _roundness_kruskal_result(master_df)
    if roundness is not None:
        add_row(
            level="exploratory_spot_pooled",
            folder=scope,
            file="",
            metric="Spot roundness (1 − eccentricity)",
            comparison=BTX_CLASS_COMPARISON_3WAY,
            test="Kruskal-Wallis",
            test_design="EXPLORATORY: all spots pooled",
            statistic=roundness["statistic_name"],
            statistic_value=roundness["statistic_value"],
            p_value=roundness["p_value"],
            p_value_adjusted="",
            significance=_sig_stars(roundness["p_value"]),
            notes=(
                f"n_spots {BTX_CLASS_EARLY_NMJ}={roundness['n_nmj']}, {BTX_CLASS_MUSCLE}={roundness['n_aneural']}, "
                f"{BTX_CLASS_NEURON}={roundness['n_neuron']}"
            ),
        )

    intensity = _intensity_mannwhitney_result(master_df)
    if intensity is not None:
        add_row(
            level="exploratory_spot_pooled",
            folder=scope,
            file="",
            metric="BTX mean spot intensity (all detected spots)",
            comparison=BTX_CLASS_COMPARISON_INTENSITY,
            test="Mann-Whitney U",
            test_design="EXPLORATORY: all spots pooled; one-sided (greater)",
            statistic=intensity["statistic_name"],
            statistic_value=intensity["statistic_value"],
            p_value=intensity["p_value"],
            p_value_adjusted="",
            significance=_sig_stars(intensity["p_value"]),
            notes=(
                f"n_spots {BTX_CLASS_EARLY_NMJ}={intensity['n_nmj']}, {BTX_CLASS_ORPHANED}={intensity['n_orphan']}; "
                f"medians {intensity['nmj_median']:.2f} vs {intensity['orphan_median']:.2f} A.U."
            ),
        )

    if otsu_th is not None and np.isfinite(otsu_th):
        filtered = master_df[master_df["MEAN_INTENSITY"] >= float(otsu_th)]
        intensity_otsu = _intensity_mannwhitney_result(filtered)
        if intensity_otsu is not None:
            add_row(
                level="exploratory_spot_pooled",
                folder=scope,
                file="",
                metric=f"BTX mean spot intensity (spots ≥ global Otsu {otsu_label})",
                comparison=BTX_CLASS_COMPARISON_INTENSITY,
                test="Mann-Whitney U",
                test_design="EXPLORATORY: all spots pooled after Otsu filter",
                statistic=intensity_otsu["statistic_name"],
                statistic_value=intensity_otsu["statistic_value"],
                p_value=intensity_otsu["p_value"],
                p_value_adjusted="",
                significance=_sig_stars(intensity_otsu["p_value"]),
                notes="For plot visualization only",
            )

    if not rows:
        stat_df = pd.DataFrame(columns=list(STAT_SUMMARY_COLUMNS))
    else:
        stat_df = pd.DataFrame(rows, columns=list(STAT_SUMMARY_COLUMNS))
    for col in ("statistic_value", "p_value", "p_value_adjusted"):
        if col in stat_df.columns:
            stat_df[col] = pd.to_numeric(stat_df[col], errors="coerce")
    return stat_df, image_medians_df, otsu_dim_noise_df


def add_global_btx_intensity_histogram_with_otsu_row(fig, outer, row_idx, master_df, *, panel_num=None):
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
        ax_combined.set_title("Global BTX Intensity Histograms (No Data)")
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
    title_prefix = f"{panel_num}. " if panel_num is not None else ""
    ax_combined.set_title(
        f"{title_prefix}Global BTX Intensity Histograms by Class (Global Otsu = {otsu_label})"
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


def _proximity_kruskal_result(df, distance_col, *, min_per_group=3):
    """Kruskal–Wallis across BTX signal classes for one proximity axis."""
    from scipy.stats import kruskal

    if df is None or len(df) == 0 or distance_col not in df.columns:
        return None
    if "BTX signal class" not in df.columns:
        return None

    groups = []
    n_by_class = {}
    medians = {}
    for cls in BTX_SIGNAL_CLASS_ORDER:
        vals = df[df["BTX signal class"] == cls][distance_col].dropna()
        n_by_class[cls] = int(len(vals))
        medians[cls] = float(vals.median()) if len(vals) else np.nan
        if len(vals) >= min_per_group:
            groups.append(vals)

    if len(groups) < 2:
        return None
    try:
        h_stat, p_val = kruskal(*groups)
    except ValueError:
        return None
    return {
        "statistic_name": "H",
        "statistic_value": float(h_stat),
        "p_value": float(p_val),
        "n_by_class": n_by_class,
        "medians": medians,
    }


def _proximity_pairwise_vs_nmj_posthoc(df, distance_col, *, min_per_group=3):
    """Pairwise Mann–Whitney (early NMJ-like vs each other class) with Holm–Bonferroni adjustment."""
    long_df = df[["BTX signal class", distance_col]].rename(columns={distance_col: "value"})
    return _pairwise_nmj_mannwhitney_posthoc(long_df, "value", min_per_group=min_per_group)


def _image_group_columns(df):
    if df is None or "SOURCE_IMAGE" not in df.columns:
        return None
    cols = []
    if "SOURCE_FOLDER" in df.columns:
        cols.append("SOURCE_FOLDER")
    cols.append("SOURCE_IMAGE")
    return cols


def build_per_image_class_medians(master_df, value_col, *, spot_filter_df=None):
    """Per-image, per-class median (and spot count) for one numeric column."""
    df = spot_filter_df if spot_filter_df is not None else master_df
    gcols = _image_group_columns(df)
    if gcols is None or value_col not in df.columns:
        return pd.DataFrame()
    work = df[gcols + ["BTX signal class", value_col]].dropna(subset=[value_col])
    if len(work) == 0:
        return pd.DataFrame()
    out = (
        work.groupby(gcols + ["BTX signal class"], observed=True)[value_col]
        .agg(median_value="median", n_spots="size")
        .reset_index()
    )
    out["metric"] = value_col
    return out


def build_per_image_all_class_medians_table(master_df):
    """Long table of class-specific medians per image for key proximity/morphology metrics."""
    if master_df is None or len(master_df) == 0:
        return pd.DataFrame()
    parts = []
    for col in ("Dist_to_Muscle_um", "Dist_to_Neuron_um", "MEAN_INTENSITY"):
        part = build_per_image_class_medians(master_df, col)
        if len(part):
            parts.append(part)
    shape_df = dataframe_for_roundness_kde_and_kruskal(master_df)
    if len(shape_df):
        part = build_per_image_class_medians(master_df, "ROUNDNESS", spot_filter_df=shape_df)
        if len(part):
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _kruskal_on_class_values(df, value_col, class_col="BTX signal class", *, min_per_group=3, class_order=None):
    """Kruskal–Wallis on one value column split by class (works for spots or image-level medians)."""
    from scipy.stats import kruskal

    if df is None or len(df) == 0 or value_col not in df.columns:
        return None
    class_order = class_order or BTX_SIGNAL_CLASS_ORDER
    groups = []
    n_by_class = {}
    medians = {}
    for cls in class_order:
        vals = df[df[class_col] == cls][value_col].dropna()
        n_by_class[cls] = int(len(vals))
        medians[cls] = float(vals.median()) if len(vals) else np.nan
        if len(vals) >= min_per_group:
            groups.append(vals)
    if len(groups) < 2:
        return None
    try:
        h_stat, p_val = kruskal(*groups)
    except ValueError:
        return None
    return {
        "statistic_name": "H",
        "statistic_value": float(h_stat),
        "p_value": float(p_val),
        "n_by_class": n_by_class,
        "medians": medians,
    }


def _pairwise_nmj_mannwhitney_posthoc(df, value_col, *, class_col="BTX signal class", min_per_group=3):
    """Pairwise Mann–Whitney (early NMJ-like vs each other class) with Holm adjustment."""
    from scipy.stats import mannwhitneyu

    if df is None or len(df) == 0:
        return None
    nmj = df[df[class_col] == BTX_CLASS_EARLY_NMJ][value_col].dropna()
    if len(nmj) < min_per_group:
        return None
    rows = []
    for other_cls in BTX_SIGNAL_CLASS_ORDER:
        if other_cls == BTX_CLASS_EARLY_NMJ:
            continue
        other = df[df[class_col] == other_cls][value_col].dropna()
        if len(other) < min_per_group:
            continue
        try:
            u_stat, p_val = mannwhitneyu(nmj, other, alternative="two-sided")
        except ValueError:
            continue
        rows.append(
            {
                "group1": BTX_CLASS_EARLY_NMJ,
                "group2": other_cls,
                "u_stat": float(u_stat),
                "p_val": float(p_val),
            }
        )
    if not rows:
        return None
    df_out = pd.DataFrame(rows)
    pvals = df_out["p_val"].to_numpy()
    m = len(pvals)
    sorted_indices = np.argsort(pvals)
    adj_p = np.zeros(m)
    running = 0.0
    for rank, orig_idx in enumerate(sorted_indices):
        adj = pvals[orig_idx] * (m - rank)
        running = max(running, min(1.0, adj))
        adj_p[orig_idx] = running
    df_out["p_val_adj"] = adj_p
    df_out["sig"] = df_out["p_val_adj"].apply(_sig_stars)
    return df_out


def image_level_proximity_analysis_results(master_df, *, min_images_per_class=3):
    """Proximity inference on per-image class medians (one value per image per class)."""
    if _image_group_columns(master_df) is None:
        return None
    muscle_med = build_per_image_class_medians(master_df, "Dist_to_Muscle_um")
    neuron_med = build_per_image_class_medians(master_df, "Dist_to_Neuron_um")
    if len(muscle_med) == 0 and len(neuron_med) == 0:
        return None
    return {
        "muscle_kruskal": _kruskal_on_class_values(
            muscle_med, "median_value", min_per_group=min_images_per_class
        ),
        "neuron_kruskal": _kruskal_on_class_values(
            neuron_med, "median_value", min_per_group=min_images_per_class
        ),
        "posthoc_muscle": _pairwise_nmj_mannwhitney_posthoc(
            muscle_med, "median_value", min_per_group=min_images_per_class
        ),
        "posthoc_neuron": _pairwise_nmj_mannwhitney_posthoc(
            neuron_med, "median_value", min_per_group=min_images_per_class
        ),
        "n_images": int(master_df[_image_group_columns(master_df)].drop_duplicates().shape[0]),
    }


def image_level_roundness_kruskal_result(master_df, *, min_images_per_class=3):
    shape_df = dataframe_for_roundness_kde_and_kruskal(master_df)
    if len(shape_df) == 0:
        return None
    med = build_per_image_class_medians(master_df, "ROUNDNESS", spot_filter_df=shape_df)
    med = med[med["BTX signal class"].isin(ROUNDNESS_KRUSKAL_CLASSES)]
    if len(med) == 0:
        return None
    result = _kruskal_on_class_values(
        med,
        "median_value",
        min_per_group=min_images_per_class,
        class_order=list(ROUNDNESS_KRUSKAL_CLASSES),
    )
    if result is None:
        return None
    result["n_images"] = int(shape_df[_image_group_columns(shape_df)].drop_duplicates().shape[0])
    return result


def _paired_image_class_series(med_df, class_a, class_b, value_col="median_value"):
    """Align per-image values for two BTX classes; returns (series_a, series_b) on shared index."""
    gcols = _image_group_columns(med_df)
    if gcols is None or value_col not in med_df.columns:
        return None, None
    a = (
        med_df[med_df["BTX signal class"] == class_a]
        .set_index(gcols)[value_col]
        .dropna()
    )
    b = (
        med_df[med_df["BTX signal class"] == class_b]
        .set_index(gcols)[value_col]
        .dropna()
    )
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return None, None
    return a.loc[shared], b.loc[shared]


def image_level_intensity_paired_nmj_vs_orphan(master_df, *, otsu_th=None, min_pairs=3):
    """Paired Wilcoxon: within each image, early NMJ-like median intensity vs Orphaned median."""
    df = master_df
    if otsu_th is not None and np.isfinite(otsu_th):
        df = df[df["MEAN_INTENSITY"] >= float(otsu_th)]
    med = build_per_image_class_medians(df, "MEAN_INTENSITY")
    if len(med) == 0:
        return None
    nmj_vals, orphan_vals = _paired_image_class_series(
        med, BTX_CLASS_EARLY_NMJ, BTX_CLASS_ORPHANED
    )
    if nmj_vals is None or len(nmj_vals) < min_pairs:
        return None
    from scipy.stats import wilcoxon

    try:
        w_stat, p_val = wilcoxon(nmj_vals, orphan_vals, alternative="greater")
    except ValueError:
        return None
    return {
        "statistic_name": "W",
        "statistic_value": float(w_stat),
        "p_value": float(p_val),
        "n_paired_images": int(len(nmj_vals)),
        "nmj_median_of_medians": float(nmj_vals.median()),
        "orphan_median_of_medians": float(orphan_vals.median()),
        "nmj_brighter_count": int((nmj_vals > orphan_vals).sum()),
    }


def image_level_intensity_nmj_vs_orphan(master_df, *, otsu_th=None, min_images_per_class=3):
    """Unpaired Mann–Whitney on per-image class medians (sensitivity analysis)."""
    df = master_df
    if otsu_th is not None and np.isfinite(otsu_th):
        df = df[df["MEAN_INTENSITY"] >= float(otsu_th)]
    med = build_per_image_class_medians(df, "MEAN_INTENSITY")
    if len(med) == 0:
        return None
    from scipy.stats import mannwhitneyu

    nmj = med[med["BTX signal class"] == BTX_CLASS_EARLY_NMJ]["median_value"].dropna()
    orphan = med[med["BTX signal class"] == BTX_CLASS_ORPHANED]["median_value"].dropna()
    if len(nmj) < min_images_per_class or len(orphan) < min_images_per_class:
        return None
    try:
        u_stat, p_val = mannwhitneyu(nmj, orphan, alternative="greater")
    except ValueError:
        return None
    return {
        "statistic_name": "U",
        "statistic_value": float(u_stat),
        "p_value": float(p_val),
        "n_nmj_images": int(len(nmj)),
        "n_orphan_images": int(len(orphan)),
        "nmj_median": float(nmj.median()),
        "orphan_median": float(orphan.median()),
    }


OTS_DIM_NOISE_REJECTION_COLUMNS = (
    "btx_signal_class",
    "n_spots",
    "n_spots_above_otsu",
    "pct_spots_above_otsu",
    "global_otsu_threshold_au",
    "interpretation",
)


def build_otsu_dim_noise_rejection_table(master_df, otsu_th=None):
    """Spot-level composition table: fraction of each class above global intensity Otsu."""
    master_df = normalize_btx_signal_classes(master_df)
    if master_df is None or len(master_df) == 0 or "MEAN_INTENSITY" not in master_df.columns:
        return pd.DataFrame(columns=list(OTS_DIM_NOISE_REJECTION_COLUMNS))
    if otsu_th is None or not np.isfinite(otsu_th):
        otsu_th = global_btx_intensity_otsu_threshold(master_df["MEAN_INTENSITY"])
    if not np.isfinite(otsu_th):
        return pd.DataFrame(columns=list(OTS_DIM_NOISE_REJECTION_COLUMNS))

    interpretations = {
        BTX_CLASS_EARLY_NMJ: (
            "Synaptic puncta near muscle and neuron; most pass Otsu — consistent with specific AChR staining."
        ),
        BTX_CLASS_MUSCLE: (
            "Muscle-proximal AChR clusters; high Otsu pass rate supports muscle-associated true signal amid dirty stain."
        ),
        BTX_CLASS_NEURON: (
            "Neuron-proximal BTX; high Otsu pass rate — may include presynaptic or developing terminals."
        ),
        BTX_CLASS_ORPHANED: (
            "Distant from both masks; low Otsu pass rate — mostly dim non-specific/background-like puncta "
            "(may include brighter immature or mislocalized signal)."
        ),
    }
    rows = []
    for cls in BTX_SIGNAL_CLASS_ORDER:
        sub = master_df[master_df["BTX signal class"] == cls]
        n = int(len(sub))
        n_above = int((sub["MEAN_INTENSITY"] >= float(otsu_th)).sum()) if n else 0
        pct = round(100.0 * n_above / n, 1) if n else np.nan
        rows.append(
            {
                "btx_signal_class": cls,
                "n_spots": n,
                "n_spots_above_otsu": n_above,
                "pct_spots_above_otsu": pct,
                "global_otsu_threshold_au": float(otsu_th),
                "interpretation": interpretations.get(cls, ""),
            }
        )
    return pd.DataFrame(rows, columns=list(OTS_DIM_NOISE_REJECTION_COLUMNS))


def build_per_image_class_fraction_above_otsu(master_df, otsu_th):
    """Per-image fraction of spots in each class with MEAN_INTENSITY >= global Otsu."""
    master_df = normalize_btx_signal_classes(master_df)
    gcols = _image_group_columns(master_df)
    if gcols is None or not np.isfinite(otsu_th):
        return pd.DataFrame()
    work = master_df[gcols + ["BTX signal class", "MEAN_INTENSITY"]].copy()
    work["above_otsu"] = work["MEAN_INTENSITY"] >= float(otsu_th)
    return (
        work.groupby(gcols + ["BTX signal class"], observed=True)["above_otsu"]
        .mean()
        .reset_index(name="frac_above_otsu")
    )


def image_level_paired_frac_above_otsu_nmj_vs_orphan(master_df, otsu_th, *, min_pairs=3):
    """Paired Wilcoxon: within image, fraction of early NMJ-like spots above Otsu vs Orphaned."""
    frac = build_per_image_class_fraction_above_otsu(master_df, otsu_th)
    if len(frac) == 0:
        return None
    nmj_vals, orphan_vals = _paired_image_class_series(
        frac, BTX_CLASS_EARLY_NMJ, BTX_CLASS_ORPHANED, value_col="frac_above_otsu"
    )
    if nmj_vals is None or len(nmj_vals) < min_pairs:
        return None
    from scipy.stats import wilcoxon

    try:
        w_stat, p_val = wilcoxon(nmj_vals, orphan_vals, alternative="greater")
    except ValueError:
        return None
    return {
        "statistic_name": "W",
        "statistic_value": float(w_stat),
        "p_value": float(p_val),
        "n_paired_images": int(len(nmj_vals)),
        "nmj_median_frac": float(nmj_vals.median()),
        "orphan_median_frac": float(orphan_vals.median()),
        "nmj_higher_frac_count": int((nmj_vals > orphan_vals).sum()),
    }


def proximity_analysis_results(df, *, min_per_group=3):
    """Global proximity statistics for muscle and neuron edge-distance axes."""
    if df is None or len(df) == 0:
        return None
    return {
        "muscle_kruskal": _proximity_kruskal_result(df, "Dist_to_Muscle_um", min_per_group=min_per_group),
        "neuron_kruskal": _proximity_kruskal_result(df, "Dist_to_Neuron_um", min_per_group=min_per_group),
        "posthoc_muscle": _proximity_pairwise_vs_nmj_posthoc(df, "Dist_to_Muscle_um", min_per_group=min_per_group),
        "posthoc_neuron": _proximity_pairwise_vs_nmj_posthoc(df, "Dist_to_Neuron_um", min_per_group=min_per_group),
    }


def get_proximity_analysis_title(
    df,
    label_base="Proximity Analysis",
    n_spots=None,
    min_per_group=3,
    *,
    primary_image_level=True,
):
    """Title for proximity scatter; uses image-level Kruskal when batch master data includes images."""
    head = label_base if n_spots is None else f"{label_base} (n={n_spots} spots)"
    if df is None or len(df) == 0:
        return f"{head}\n(No Data)"

    use_image_level = primary_image_level and _image_group_columns(df) is not None
    if use_image_level:
        res = image_level_proximity_analysis_results(df, min_images_per_class=min_per_group)
        if res is not None and res.get("n_images"):
            head = f"{label_base} ({res['n_images']} images; image-level inference)"
    else:
        res = proximity_analysis_results(df, min_per_group=min_per_group)

    if res is None:
        return head

    lines = [head]
    muscle = res.get("muscle_kruskal")
    neuron = res.get("neuron_kruskal")
    if muscle is not None:
        lines.append(
            f"Dist to muscle — Kruskal P = {muscle['p_value']:.4g} {_sig_stars(muscle['p_value'])}"
        )
    if neuron is not None:
        lines.append(
            f"Dist to neuron — Kruskal P = {neuron['p_value']:.4g} {_sig_stars(neuron['p_value'])}"
        )
    if muscle is None and neuron is None:
        need = "images" if use_image_level else "spots"
        lines.append(f"(Need ≥{min_per_group} {need} per class for Kruskal–Wallis)")
    return "\n".join(lines)


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
    full_title = get_proximity_analysis_title(df, label_base=title, n_spots=n_spots)
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
    ax_nmj_rate.set_title(f"1. {BTX_CLASS_EARLY_NMJ} Formation Rate by Folder")
    ax_nmj_rate.set_xlabel("Folder")
    ax_nmj_rate.set_ylabel(f"{BTX_CLASS_EARLY_NMJ} Rate (%)")
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
    """Batch summary figure: histogram, original panels, and Otsu-thresholded counterparts.

    Returns ``(fig, panel_specs, meta)`` where ``meta`` contains test results and messaging fields.
    """
    master_df = normalize_btx_signal_classes(master_df)
    stats_df_spec = normalize_file_stats_columns(pd.DataFrame(all_file_stats or []))

    n_rows = 6 if run_all else 5
    fig_h = 48 if run_all else 42
    fig_w = 24 if run_all else 20
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.01, wspace=0.01)
    height_ratios = [1.35] + [1.0] * (n_rows - 2) + [0.95]
    outer = fig.add_gridspec(n_rows, 2, height_ratios=height_ratios)

    hist_axes, global_otsu_th = add_global_btx_intensity_histogram_with_otsu_row(
        fig, outer, 0, master_df
    )

    ax_scatter, ax_prox_kde_x, ax_prox_kde_y, ax_prox_title = proximity_joint_axes(
        fig, outer[1, 0], title_first=True, large_main_panel=True
    )
    ax_size_kde = fig.add_subplot(outer[1, 1])
    ax_circ_kde = fig.add_subplot(outer[2, 0])
    ax_overlap_kde = fig.add_subplot(outer[2, 1])
    ax_intensity_kde = fig.add_subplot(outer[3, 0])
    ax_intensity_otsu_kde = fig.add_subplot(outer[3, 1])
    ax_abundance = fig.add_subplot(outer[4, 0])
    ax_abundance_otsu = fig.add_subplot(outer[4, 1])
    ax_control = fig.add_subplot(outer[5, :]) if run_all else None

    draw_proximity_joint(
        ax_scatter,
        ax_prox_kde_x,
        ax_prox_kde_y,
        master_df,
        distance_threshold_um,
        "Global BTX Proximity Analysis",
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
    ax_size_kde.set_title("Global BTX Size KDE (all spots)")
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
        label_base="Global early NMJ-like Roundness KDE (1 − eccentricity)",
    )
    ax_circ_kde.set_title(roundness_title_global)
    ax_circ_kde.set_xlabel("Roundness (1 = circle)")
    ax_circ_kde.set_ylabel("Probability Density")
    ax_circ_kde.set_xlim(0, 1)

    master_innervation = (
        master_df[master_df["BTX signal class"] == BTX_CLASS_EARLY_NMJ]
        if len(master_df) > 0
        else master_df
    )
    if len(master_innervation) > 0:
        sns.histplot(
            data=master_innervation,
            x="INNERVATION_OVERLAP_PCT",
            color=BTX_SIGNAL_CLASS_PALETTE[BTX_CLASS_EARLY_NMJ],
            ax=ax_overlap_kde,
        )
    ax_overlap_kde.set_title(f"Global {BTX_CLASS_EARLY_NMJ} Innervation Distribution (all spots)")
    ax_overlap_kde.set_xlabel(f"{BTX_CLASS_EARLY_NMJ} Innervation (%)")
    ax_overlap_kde.set_ylabel("Count")

    _, intensity_summary = draw_global_intensity_kde_panel(
        ax_intensity_kde,
        master_df,
        title="Global Receptor Intensity (all spots)",
        otsu_th=global_otsu_th,
        otsu_filtered=False,
    )
    otsu_int_title = (
        f"Global Receptor Intensity (spots ≥ Otsu {global_otsu_th:.1f} A.U.)"
        if np.isfinite(global_otsu_th)
        else "Global Receptor Intensity (Otsu-filtered)"
    )
    draw_global_intensity_kde_panel(
        ax_intensity_otsu_kde,
        master_df,
        title=otsu_int_title,
        otsu_th=global_otsu_th,
        otsu_filtered=True,
    )

    _, p_friedman_abundance, conover_abundance_results = draw_zone_btx_abundance_panel(
        ax_abundance,
        stats_df_spec,
        title_base="Global BTX Abundance — all detected spots",
        include_nmj_zone=True,
    )

    otsu_abundance_stats = build_otsu_thresholded_abundance_stats(
        master_df, all_file_stats, global_otsu_th
    )
    _, p_friedman_abundance_otsu, conover_abundance_otsu_results = draw_zone_btx_abundance_panel(
        ax_abundance_otsu,
        otsu_abundance_stats,
        title_base=(
            f"Global BTX Abundance — spots ≥ Otsu {global_otsu_th:.1f} A.U."
            if np.isfinite(global_otsu_th)
            else "Global BTX Abundance — Otsu-filtered"
        ),
        include_nmj_zone=True,
    )

    if ax_control is not None and "SOURCE_FOLDER" in master_df.columns and "SOURCE_IMAGE" in master_df.columns:
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
        ax_control.set_title(f"Per-Image {BTX_CLASS_EARLY_NMJ} Rate Control Chart")
        ax_control.set_xlabel("Folder")
        ax_control.set_ylabel(f"{BTX_CLASS_EARLY_NMJ} Rate (%)")
        ax_control.tick_params(axis="x", rotation=45)

    panel_specs = [
        ("panel01_global_intensity_histogram_otsu", hist_axes),
        ("panel02_global_proximity", [ax_prox_title, ax_prox_kde_x, ax_scatter, ax_prox_kde_y]),
        ("panel03_global_size_kde", [ax_size_kde]),
        ("panel04_global_roundness_kde", [ax_circ_kde]),
        ("panel05_global_innervation", [ax_overlap_kde]),
        ("panel06_global_intensity_all", [ax_intensity_kde]),
        ("panel07_global_intensity_otsu", [ax_intensity_otsu_kde]),
        ("panel08_btx_abundance_all", [ax_abundance]),
        ("panel09_btx_abundance_otsu", [ax_abundance_otsu]),
    ]
    if run_all and ax_control is not None:
        panel_specs.append(("panel10_per_image_nmj_control", [ax_control]))

    meta = {
        "intensity_summary": intensity_summary,
        "friedman_p_abundance": float(p_friedman_abundance) if p_friedman_abundance is not None else None,
        "conover_abundance_results": conover_abundance_results,
        "friedman_p_abundance_otsu": float(p_friedman_abundance_otsu) if p_friedman_abundance_otsu is not None else None,
        "conover_abundance_otsu_results": conover_abundance_otsu_results,
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
