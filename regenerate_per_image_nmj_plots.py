#!/usr/bin/env python3
"""Regenerate per-image ``*_NMJ_Plot.png`` files for an existing batch run folder.

Re-reads source images from ``data/`` using the run snapshot ``channel_mapping_config.json``
and parameters from ``run_config.json``. Does not rewrite CSVs or aggregate summaries.

Example::

    python regenerate_per_image_nmj_plots.py output/20260629_144848
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _load_run_jobs(run_dir: str, data_root: str, *, only: str | None = None):
    run_dir = os.path.abspath(run_dir)
    data_root = os.path.abspath(data_root)
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
        run_config = json.load(f)
    channel_path = os.path.join(run_dir, "channel_mapping_config.json")
    with open(channel_path, encoding="utf-8") as f:
        channel_map = json.load(f)

    thr_tag = run_config.get("threshold_tag", "")
    jobs = []
    for folder_name, files in channel_map.items():
        dataset_dir = os.path.join(data_root, folder_name)
        for czi_file, fc in files.items():
            if fc.get("skip", False):
                continue
            if only and only not in czi_file:
                continue
            image_path = os.path.join(dataset_dir, czi_file)
            if not os.path.isfile(image_path):
                print(f"WARNING: source image missing, skipping: {image_path}", file=sys.stderr)
                continue
            file_stem = os.path.splitext(czi_file)[0]
            out_png = os.path.join(run_dir, folder_name, f"{file_stem}{thr_tag}_NMJ_Plot.png")
            jobs.append(
                {
                    "folder": folder_name,
                    "czi_file": czi_file,
                    "image_path": image_path,
                    "out_png": out_png,
                    "muscle_idx": int(fc["m"]),
                    "neuron_idx": int(fc["n"]),
                    "btx_idx": int(fc["b"]),
                    "pixel_size": float(fc.get("p", 1.0)),
                }
            )
    return run_config, jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Existing output run folder, e.g. output/20260629_144848")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Dataset root containing source images (default: data)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Optional substring filter on image filename (e.g. 0714M-HF-03)",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    from nmj_image_analysis import analyze_image_for_nmj_plot, dog_sigma_ratio_from_sensitivity, save_nmj_plot_png

    run_config, jobs = _load_run_jobs(args.run_dir, args.data_root, only=args.only)
    params = run_config.get("parameters", {})
    auto_threshold = bool(params.get("auto_threshold", True))
    dog_sigma_ratio = float(params.get("dog_sigma_ratio", 1.6))
    if auto_threshold and params.get("auto_thr_sensitivity"):
        dog_sigma_ratio = dog_sigma_ratio_from_sensitivity(str(params["auto_thr_sensitivity"]))

    if not jobs:
        print("No images matched.", file=sys.stderr)
        return 1

    ok = 0
    skipped = 0
    failed = 0
    for i, job in enumerate(jobs, start=1):
        label = f"{job['folder']}/{job['czi_file']}"
        print(f"[{i}/{len(jobs)}] {label} -> {job['out_png']}")
        try:
            ctx = analyze_image_for_nmj_plot(
                job["image_path"],
                muscle_idx=job["muscle_idx"],
                neuron_idx=job["neuron_idx"],
                btx_idx=job["btx_idx"],
                pixel_size=job["pixel_size"],
                min_diameter_um=float(params.get("min_diameter_um", 5.0)),
                max_diameter_um=float(params.get("max_diameter_um", 12.0)),
                auto_threshold=auto_threshold,
                threshold=params.get("threshold"),
                dog_sigma_ratio=dog_sigma_ratio,
                btx_bg_radius_um=float(params.get("btx_bg_radius_um", 12.0)),
                m_thresh_mult=float(params.get("m_thresh_mult", 1.0)),
                n_thresh_mult=float(params.get("n_thresh_mult", 1.0)),
                distance_threshold_um=float(params.get("distance_threshold_um", 1.0)),
            )
            if ctx is None:
                print(f"  skipped (no detectable spots or incompatible scale)")
                skipped += 1
                continue
            os.makedirs(os.path.dirname(job["out_png"]), exist_ok=True)
            save_nmj_plot_png(ctx, job["out_png"])
            ok += 1
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            failed += 1

    print(f"Done: {ok} regenerated, {skipped} skipped, {failed} failed.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
