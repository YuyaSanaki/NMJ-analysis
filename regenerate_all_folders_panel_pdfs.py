#!/usr/bin/env python3
"""Rebuild aggregate dashboard PNG + per-panel PDFs from a saved ``ALL_FOLDERS_MASTER_RESULTS*.csv``.

Does not re-read CZI files. Panel 6 (Friedman / zone densities) needs per-image columns
``File``, ``Density_early_NMJ_like``, ``Density_Muscle_associated``, ``Density_Neuron_associated``, ``Density_Orphaned``. Those are **not** in the
spot-level master CSV; new ALL-folder batch runs save them as ``ALL_FOLDERS_FILE_STATS*.csv``
next to the master file. This script loads that companion CSV automatically when present, or
use ``--file-stats-csv`` to point to one explicitly.

Examples::

    python regenerate_all_folders_panel_pdfs.py output/20260626_120000/ALL_FOLDERS_MASTER_RESULTS_thrConservative.csv
    python regenerate_all_folders_panel_pdfs.py output/20260626_120000/ALL_FOLDERS_MASTER_RESULTS.csv \\
        --distance-threshold 1.0 --output-stem data/my_rerun_dashboard
"""

from __future__ import annotations

import argparse
import os
import sys


def _companion_file_stats_csv(master_csv: str) -> str | None:
    """Look beside the master CSV for a batch-saved per-image zone-density table."""
    ab = os.path.abspath(master_csv)
    base = os.path.basename(ab)
    stem, ext = os.path.splitext(ab)
    if "ALL_FOLDERS_MASTER_RESULTS" in base:
        candidate = stem.replace("ALL_FOLDERS_MASTER_RESULTS", "ALL_FOLDERS_FILE_STATS") + ext
        if os.path.isfile(candidate):
            return candidate
    if "BATCH_MASTER_RESULTS" in base:
        candidate = stem.replace("BATCH_MASTER_RESULTS", "BATCH_FILE_STATS") + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("master_csv", help="Path to ALL_FOLDERS_MASTER_RESULTS*.csv")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=1.0,
        help="Functional NMJ boundary (μm); must match the run you want to mirror.",
    )
    parser.add_argument(
        "--file-stats-csv",
        default=None,
        help="Optional CSV with columns File, Density_early_NMJ_like, Density_Muscle_associated, "
        "Density_Neuron_associated, Density_Orphaned "
        "(one row per image) to restore panel 6.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output path stem without extension (PNG + panel PDFs). "
        "Default: <master_csv path without .csv>_dashboard_regen",
    )
    parser.add_argument(
        "--single-row-layout",
        action="store_true",
        help="Use 3×2 layout without row 7 control chart (matches single-folder batch summary).",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from nmj_master_dashboard import (
        _export_figure_panels_to_pdfs,
        build_aggregate_batch_dashboard_figure,
        ensure_roundness_column,
        normalize_btx_signal_classes,
        normalize_file_stats_columns,
        present_otsu_spot_change_comparisons,
        present_otsu_spot_count_matrix,
    )

    if not os.path.isfile(args.master_csv):
        print(f"Error: file not found: {args.master_csv}", file=sys.stderr)
        return 2

    master_df = pd.read_csv(args.master_csv)
    master_df = ensure_roundness_column(normalize_btx_signal_classes(master_df))

    file_stats_path = args.file_stats_csv
    if not file_stats_path:
        file_stats_path = _companion_file_stats_csv(args.master_csv)
        if file_stats_path:
            print(f"Using companion zone-density CSV: {file_stats_path}", file=sys.stderr)

    all_file_stats: list = []
    if file_stats_path:
        if not os.path.isfile(file_stats_path):
            print(f"Error: file-stats CSV not found: {file_stats_path}", file=sys.stderr)
            return 2
        fs = normalize_file_stats_columns(pd.read_csv(file_stats_path))
        all_file_stats = fs.to_dict("records")

    run_all = not args.single_row_layout
    stem = args.output_stem or (os.path.splitext(os.path.abspath(args.master_csv))[0] + "_dashboard_regen")

    fig, panel_specs, meta = build_aggregate_batch_dashboard_figure(
        master_df,
        args.distance_threshold,
        run_all=run_all,
        all_file_stats=all_file_stats,
    )
    out_png = f"{stem}.png"
    fig.savefig(out_png, bbox_inches="tight")
    written, errs = _export_figure_panels_to_pdfs(fig, panel_specs, stem)
    plt.close(fig)

    def _write_otsu_csv(meta_key, master_token, batch_token, fallback_name):
        df = meta.get(meta_key)
        if df is None:
            return
        if meta_key == "otsu_spot_count_matrix_df":
            df = present_otsu_spot_count_matrix(df)
        elif meta_key == "otsu_spot_change_comparisons_df":
            df = present_otsu_spot_change_comparisons(df)
        master_ab = os.path.abspath(args.master_csv)
        master_base = os.path.basename(master_ab)
        master_stem, ext = os.path.splitext(master_ab)
        if "ALL_FOLDERS_MASTER_RESULTS" in master_base:
            out_csv = master_stem.replace("ALL_FOLDERS_MASTER_RESULTS", master_token) + ext
        elif "BATCH_MASTER_RESULTS" in master_base:
            out_csv = master_stem.replace("BATCH_MASTER_RESULTS", batch_token) + ext
        else:
            out_csv = os.path.join(os.path.dirname(master_ab), fallback_name)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {fallback_name}: {out_csv}", file=sys.stderr)

    _write_otsu_csv(
        "otsu_spot_count_matrix_df",
        "ALL_FOLDERS_OTSU_SPOT_COUNT_MATRIX",
        "BATCH_OTSU_SPOT_COUNT_MATRIX",
        "OTSU_SPOT_COUNT_MATRIX.csv",
    )
    _write_otsu_csv(
        "otsu_spot_change_comparisons_df",
        "ALL_FOLDERS_OTSU_SPOT_CHANGE_COMPARISONS",
        "BATCH_OTSU_SPOT_CHANGE_COMPARISONS",
        "OTSU_SPOT_CHANGE_COMPARISONS.csv",
    )
    _write_otsu_csv(
        "paired_otsu_spot_change_df",
        "ALL_FOLDERS_PAIRED_OTSU_SPOT_CHANGE",
        "BATCH_PAIRED_OTSU_SPOT_CHANGE",
        "PAIRED_OTSU_SPOT_CHANGE.csv",
    )

    print(f"Wrote dashboard PNG: {out_png}")
    if written:
        print(f"Wrote {len(written)} panel PDF(s):")
        for p in written:
            print(f"  {p}")
    else:
        print("No panel PDFs were written.", file=sys.stderr)
    if errs:
        print("Export messages:", file=sys.stderr)
        for e in errs[:15]:
            print(f"  {e}", file=sys.stderr)
        if len(errs) > 15:
            print(f"  … and {len(errs) - 15} more", file=sys.stderr)

    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
