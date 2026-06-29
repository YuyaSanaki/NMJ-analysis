#!/usr/bin/env python3
"""Consistency checks for timestamped output folders and a one-image pipeline smoke test."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime
from io import BytesIO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nmj_run_output import (  # noqa: E402
    apply_channel_mapping_to_session_keys,
    build_run_zip_bytes,
    channel_config_state_key,
    create_run_output_dir,
    export_channel_mapping_from_session,
    iter_run_files,
    mirror_dataset_output_path,
    read_channel_mapping_config,
    resolve_channel_mapping_for_folder,
    save_run_config_files,
    snapshot_channel_mappings,
)


class RunOutputHelpersTest(unittest.TestCase):
    def test_create_run_output_dir_uses_timestamp(self):
        when = datetime(2026, 6, 26, 14, 30, 52)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_output_dir(root=tmp, when=when)
            self.assertTrue(run_dir.endswith(os.path.join("20260626_143052")))
            self.assertTrue(os.path.isdir(run_dir))

    def test_mirror_dataset_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = os.path.join(tmp, "data")
            dataset = os.path.join(data_root, "SP11BTXSAA")
            run_dir = os.path.join(tmp, "output", "20260626_120000")
            os.makedirs(dataset)
            out = mirror_dataset_output_path(run_dir, data_root, dataset, "img_analysis.csv")
            self.assertEqual(
                out,
                os.path.join(run_dir, "SP11BTXSAA", "img_analysis.csv"),
            )
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("ok")
            self.assertTrue(os.path.isfile(out))

    def test_snapshot_channel_mappings_merges_ui_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = os.path.join(tmp, "data")
            folder = os.path.join(data_root, "DS1")
            os.makedirs(folder)
            with open(os.path.join(folder, "channel_mapping_config.json"), "w", encoding="utf-8") as jf:
                json.dump({"a.czi": {"m": 0, "n": 1, "b": 2, "p": 0.1, "skip": False}}, jf)
            snap = snapshot_channel_mappings(
                data_root,
                [folder],
                active_folder_path=folder,
                active_file_configs={
                    "a.czi": {"muscle": 0, "neuron": 1, "btx": 3, "pixel_size": 0.2, "skip": False},
                    "b.czi": {"muscle": 0, "neuron": 1, "btx": 3, "pixel_size": 0.2, "skip": True},
                },
            )
            self.assertIn("DS1", snap)
            self.assertEqual(snap["DS1"]["a.czi"]["b"], 3)
            self.assertIn("b.czi", snap["DS1"])

    def test_save_and_zip_run_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = create_run_output_dir(root=tmp, when=datetime(2026, 1, 1, 0, 0, 0))
            cfg = {"run_id": "20260101_000000", "mode": "test"}
            snap = {"DS1": {"x.czi": {"m": 0, "n": 1, "b": 2, "p": 0.1, "skip": False}}}
            save_run_config_files(run_dir, cfg, snap)
            nested = mirror_dataset_output_path(run_dir, os.path.join(tmp, "data"), os.path.join(tmp, "data", "DS1"), "x_analysis.csv")
            with open(nested, "w", encoding="utf-8") as fh:
                fh.write("col\n1\n")

            files = list(iter_run_files(run_dir))
            self.assertGreaterEqual(len(files), 3)
            zbytes = build_run_zip_bytes(run_dir)
            with zipfile.ZipFile(BytesIO(zbytes)) as zf:
                names = set(zf.namelist())
            self.assertIn("run_config.json", names)
            self.assertIn("channel_mapping_config.json", names)
            self.assertTrue(any(n.endswith("x_analysis.csv") for n in names))

    def test_channel_config_state_key_scopes_by_folder(self):
        self.assertNotEqual(
            channel_config_state_key("FolderA", "img.czi", "m"),
            channel_config_state_key("FolderB", "img.czi", "m"),
        )

    def test_apply_and_export_channel_mapping_roundtrip(self):
        folder_rel = "DS1"
        mapping = {"a.czi": {"m": 0, "n": 1, "b": 2, "p": 0.15, "skip": True}}
        session = apply_channel_mapping_to_session_keys(folder_rel, mapping, ["a.czi"])
        self.assertEqual(session[channel_config_state_key(folder_rel, "a.czi", "b")], 2)
        exported = export_channel_mapping_from_session(folder_rel, ["a.czi"], session)
        self.assertEqual(exported["a.czi"]["skip"], True)
        self.assertEqual(exported["a.czi"]["p"], 0.15)

    def test_read_channel_mapping_config_missing_or_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_channel_mapping_config(tmp))
            bad = os.path.join(tmp, "channel_mapping_config.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            self.assertIsNone(read_channel_mapping_config(tmp))

    def test_resolve_channel_mapping_prefers_cache(self):
        disk = {"a.czi": {"m": 0, "n": 1, "b": 2, "p": 0.1, "skip": False}}
        cache = {"a.czi": {"m": 1, "n": 2, "b": 3, "p": 0.2, "skip": True}}
        resolved = resolve_channel_mapping_for_folder(
            "DS1", disk_mapping=disk, cache_mapping=cache
        )
        self.assertEqual(resolved, cache)
        self.assertEqual(
            resolve_channel_mapping_for_folder("DS1", disk_mapping=disk, cache_mapping=None),
            disk,
        )


class OtsuAbundanceStatsTest(unittest.TestCase):
    def test_duplicate_filename_across_folders_not_merged(self):
        import pandas as pd

        from nmj_master_dashboard import (
            ABUNDANCE_COL_EARLY_NMJ,
            AREA_COL_EARLY_NMJ,
            AREA_COL_MUSCLE,
            AREA_COL_NEURON,
            AREA_COL_ORPHANED,
            BTX_CLASS_EARLY_NMJ,
            build_otsu_thresholded_abundance_stats,
        )

        otsu_th = 100.0
        master_rows = []
        for folder, n_spots in (("FolderA", 10), ("FolderB", 2)):
            for _ in range(n_spots):
                master_rows.append(
                    {
                        "SOURCE_FOLDER": folder,
                        "SOURCE_IMAGE": "dup.czi",
                        "MEAN_INTENSITY": 200.0,
                        "BTX signal class": BTX_CLASS_EARLY_NMJ,
                    }
                )
        master_df = pd.DataFrame(master_rows)

        file_stats = [
            {
                "File": "dup.czi",
                "Folder": "FolderA",
                AREA_COL_EARLY_NMJ: 1000.0,
                AREA_COL_MUSCLE: 1000.0,
                AREA_COL_NEURON: 1000.0,
                AREA_COL_ORPHANED: 1000.0,
            },
            {
                "File": "dup.czi",
                "Folder": "FolderB",
                AREA_COL_EARLY_NMJ: 500.0,
                AREA_COL_MUSCLE: 500.0,
                AREA_COL_NEURON: 500.0,
                AREA_COL_ORPHANED: 500.0,
            },
        ]

        out = build_otsu_thresholded_abundance_stats(master_df, file_stats, otsu_th)
        self.assertEqual(len(out), 2)
        by_folder = out.set_index("Folder")[ABUNDANCE_COL_EARLY_NMJ].to_dict()
        # 10 spots / 4000 µm² * 1e6 and 2 spots / 2000 µm² * 1e6 (spots per 1 mm²)
        self.assertAlmostEqual(by_folder["FolderA"], 2500.0)
        self.assertAlmostEqual(by_folder["FolderB"], 1000.0)

        # Buggy path would assign 12 spots to both rows -> 3000 and 6000 spots/mm²
        self.assertNotAlmostEqual(by_folder["FolderA"], 3000.0)
        self.assertNotAlmostEqual(by_folder["FolderB"], 6000.0)


class GlobalBtxOtsuThresholdTest(unittest.TestCase):
    def test_integer_au_histogram_stable_for_fractional_spot_means(self):
        import numpy as np
        from skimage.filters import threshold_otsu

        from nmj_master_dashboard import global_btx_intensity_otsu_threshold

        dim = np.full(500, 400.25)
        bright = np.full(500, 1800.75)
        fractional = np.concatenate([dim, bright])

        th = global_btx_intensity_otsu_threshold(fractional)
        self.assertEqual(th, float(threshold_otsu(np.rint(fractional).astype(np.int64))))
        self.assertNotEqual(th, float(threshold_otsu(fractional, nbins=256)))
        self.assertNotEqual(th, float(threshold_otsu(fractional, nbins=512)))


class DashboardFunctionalityTest(unittest.TestCase):
    def test_zone_abundance_normalized_per_mm2(self):
        import pandas as pd

        from nmj_master_dashboard import (
            ABUNDANCE_COL_EARLY_NMJ,
            AREA_COL_EARLY_NMJ,
            AREA_COL_MUSCLE,
            AREA_COL_NEURON,
            AREA_COL_ORPHANED,
            BTX_CLASS_EARLY_NMJ,
            BTX_CLASS_MUSCLE,
            BTX_CLASS_NEURON,
            BTX_CLASS_ORPHANED,
            SPOT_DENSITY_PER_MM2_LABEL,
            UM2_PER_MM2,
            _append_zone_abundance_columns,
        )

        stats = pd.DataFrame(
            [
                {
                    "File": "a.czi",
                    BTX_CLASS_EARLY_NMJ: 5,
                    BTX_CLASS_MUSCLE: 0,
                    BTX_CLASS_NEURON: 0,
                    BTX_CLASS_ORPHANED: 0,
                    AREA_COL_EARLY_NMJ: 250_000.0,
                    AREA_COL_MUSCLE: 250_000.0,
                    AREA_COL_NEURON: 250_000.0,
                    AREA_COL_ORPHANED: 250_000.0,
                }
            ]
        )
        out = _append_zone_abundance_columns(stats)
        expected = 5 / 1_000_000.0 * UM2_PER_MM2
        self.assertAlmostEqual(out[ABUNDANCE_COL_EARLY_NMJ].iloc[0], expected)
        self.assertEqual(UM2_PER_MM2, 1_000_000.0)
        self.assertEqual(SPOT_DENSITY_PER_MM2_LABEL, "Spots / mm²")

    def test_build_aggregate_dashboard_figure_smoke(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        from nmj_master_dashboard import (
            ABUNDANCE_COL_EARLY_NMJ,
            AREA_COL_EARLY_NMJ,
            AREA_COL_MUSCLE,
            AREA_COL_NEURON,
            AREA_COL_ORPHANED,
            BTX_CLASS_EARLY_NMJ,
            BTX_CLASS_MUSCLE,
            BTX_CLASS_NEURON,
            BTX_CLASS_ORPHANED,
            BTX_SIGNAL_CLASS_ORDER,
            build_aggregate_batch_dashboard_figure,
            ensure_roundness_column,
            normalize_btx_signal_classes,
        )

        rows = []
        for img_i, folder in enumerate(("FolderA", "FolderB")):
            for spot_i in range(6):
                rows.append(
                    {
                        "SOURCE_FOLDER": folder,
                        "SOURCE_IMAGE": f"img{img_i}.czi",
                        "MEAN_INTENSITY": 500.0 + spot_i * 100,
                        "RADIUS": 0.5 + 0.1 * spot_i,
                        "ROUNDNESS": 0.7,
                        "AREA_PX": 20,
                        "Dist_to_Muscle_um": float(spot_i),
                        "Dist_to_Neuron_um": float(5 - spot_i),
                        "INNERVATION_OVERLAP_PCT": 50.0,
                        "BTX signal class": BTX_SIGNAL_CLASS_ORDER[spot_i % 4],
                        "is_NMJ": spot_i % 4 == 0,
                    }
                )
        master_df = ensure_roundness_column(normalize_btx_signal_classes(pd.DataFrame(rows)))
        file_stats = [
            {
                "File": "img0.czi",
                "Folder": "FolderA",
                BTX_CLASS_EARLY_NMJ: 2,
                BTX_CLASS_MUSCLE: 2,
                BTX_CLASS_NEURON: 1,
                BTX_CLASS_ORPHANED: 1,
                "Area_early_NMJ_like_um2": 100_000.0,
                "Area_Muscle_associated_um2": 100_000.0,
                "Area_Neuron_associated_um2": 100_000.0,
                "Area_Orphaned_um2": 100_000.0,
            },
            {
                "File": "img1.czi",
                "Folder": "FolderB",
                BTX_CLASS_EARLY_NMJ: 1,
                BTX_CLASS_MUSCLE: 1,
                BTX_CLASS_NEURON: 2,
                BTX_CLASS_ORPHANED: 2,
                "Area_early_NMJ_like_um2": 200_000.0,
                "Area_Muscle_associated_um2": 200_000.0,
                "Area_Neuron_associated_um2": 200_000.0,
                "Area_Orphaned_um2": 200_000.0,
            },
        ]

        fig, panel_specs, meta = build_aggregate_batch_dashboard_figure(
            master_df,
            distance_threshold_um=1.0,
            run_all=True,
            all_file_stats=file_stats,
        )
        try:
            self.assertIsNotNone(fig)
            self.assertGreaterEqual(len(panel_specs), 9)
            panel_names = [name for name, _axes in panel_specs]
            self.assertIn("panel01a_global_intensity_histogram_all", panel_names)
            self.assertIn("panel01b_global_intensity_histogram_paired", panel_names)
            self.assertIn("panel06_global_intensity_otsu", panel_names)
            self.assertIn("panel07_total_spots_otsu", panel_names)
            self.assertNotIn("panel06_global_intensity_all", panel_names)
            self.assertIsNotNone(meta.get("global_btx_intensity_otsu"))
        finally:
            plt.close(fig)

    def test_build_paired_otsu_spot_change_table(self):
        import pandas as pd

        from nmj_master_dashboard import (
            BTX_CLASS_EARLY_NMJ,
            BTX_CLASS_MUSCLE,
            BTX_CLASS_ORPHANED,
            build_paired_otsu_spot_change_by_class_table,
        )

        master_df = pd.DataFrame(
            {
                "MEAN_INTENSITY": [1400.0, 1200.0, 1300.0, 900.0, 850.0],
                "BTX signal class": [
                    BTX_CLASS_EARLY_NMJ,
                    BTX_CLASS_EARLY_NMJ,
                    BTX_CLASS_MUSCLE,
                    BTX_CLASS_ORPHANED,
                    BTX_CLASS_ORPHANED,
                ],
            }
        )
        paired_df = pd.DataFrame(
            {
                "MEAN_INTENSITY": [1400.0, 1300.0, 900.0],
                "BTX signal class": [BTX_CLASS_EARLY_NMJ, BTX_CLASS_MUSCLE, BTX_CLASS_ORPHANED],
            }
        )
        out = build_paired_otsu_spot_change_by_class_table(master_df, paired_df, 1298.0, 1316.0)
        nmj_row = out[out["BTX signal class"] == BTX_CLASS_EARLY_NMJ].iloc[0]
        muscle_row = out[out["BTX signal class"] == BTX_CLASS_MUSCLE].iloc[0]
        orphan_row = out[out["BTX signal class"] == BTX_CLASS_ORPHANED].iloc[0]
        self.assertEqual(nmj_row["n_spots_total"], 2)
        self.assertEqual(nmj_row["spots_ge_global"], "1 / 2")
        self.assertEqual(nmj_row["spots_ge_paired"], "1 / 1")
        self.assertEqual(muscle_row["spots_ge_global"], "1 / 1")
        self.assertEqual(muscle_row["spots_ge_paired"], "0 / 1")
        self.assertEqual(orphan_row["spots_ge_global"], "0 / 2")
        self.assertEqual(orphan_row["spots_ge_paired"], "0 / 1")

    def test_build_batch_stat_summary_includes_paired_spots_table(self):
        import pandas as pd

        from nmj_master_dashboard import (
            BTX_CLASS_EARLY_NMJ,
            BTX_CLASS_ORPHANED,
            INTENSITY_PAIRED_COMPARISON_SPOTS_COLUMNS,
            build_batch_stat_summary_dataframe,
            ensure_roundness_column,
            global_btx_intensity_otsu_threshold,
            normalize_btx_signal_classes,
        )

        rows = []
        for spot_i in range(4):
            rows.append(
                {
                    "SOURCE_FOLDER": "FolderA",
                    "SOURCE_IMAGE": "paired.czi",
                    "SPOT_ID": spot_i,
                    "MEAN_INTENSITY": 800.0 if spot_i % 2 == 0 else 400.0,
                    "Dist_to_Muscle_um": float(spot_i),
                    "Dist_to_Neuron_um": float(4 - spot_i),
                    "ROUNDNESS": 0.7,
                    "AREA_PX": 20,
                    "INNERVATION_OVERLAP_PCT": 50.0,
                    "is_NMJ": spot_i % 2 == 0,
                    "GLOBAL_BTX_INTENSITY_OTSU": 100.0,
                    "BTX signal class": BTX_CLASS_EARLY_NMJ if spot_i % 2 == 0 else BTX_CLASS_ORPHANED,
                }
            )
        master_df = ensure_roundness_column(normalize_btx_signal_classes(pd.DataFrame(rows)))
        _, _, _, paired_df = build_batch_stat_summary_dataframe(
            master_df,
            distance_threshold_um=1.0,
            dash_meta={"global_btx_intensity_otsu": 100.0},
            run_all=True,
        )
        self.assertEqual(list(paired_df.columns), list(INTENSITY_PAIRED_COMPARISON_SPOTS_COLUMNS))
        self.assertEqual(len(paired_df), 4)
        self.assertEqual(paired_df.iloc[0]["SOURCE_IMAGE"], "paired.czi")
        self.assertEqual(float(paired_df.iloc[0]["GLOBAL_BTX_INTENSITY_OTSU"]), 100.0)
        self.assertEqual(
            float(paired_df.iloc[0]["PAIRED_BTX_INTENSITY_OTSU"]),
            global_btx_intensity_otsu_threshold(paired_df["MEAN_INTENSITY"]),
        )

    def test_regenerate_from_saved_master_csv_if_present(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        from nmj_master_dashboard import (
            build_aggregate_batch_dashboard_figure,
            ensure_roundness_column,
            normalize_btx_signal_classes,
            normalize_file_stats_columns,
        )

        master_csv = os.path.join(
            ROOT,
            "output",
            "20260627_072037",
            "ALL_FOLDERS_MASTER_RESULTS_thrConservative.csv",
        )
        if not os.path.isfile(master_csv):
            self.skipTest("saved master CSV not present")

        master_df = ensure_roundness_column(
            normalize_btx_signal_classes(pd.read_csv(master_csv))
        )
        file_stats_csv = master_csv.replace(
            "ALL_FOLDERS_MASTER_RESULTS", "ALL_FOLDERS_FILE_STATS"
        )
        all_file_stats = []
        if os.path.isfile(file_stats_csv):
            all_file_stats = normalize_file_stats_columns(
                pd.read_csv(file_stats_csv)
            ).to_dict("records")

        fig, panel_specs, meta = build_aggregate_batch_dashboard_figure(
            master_df,
            distance_threshold_um=1.0,
            run_all=True,
            all_file_stats=all_file_stats,
        )
        try:
            self.assertGreater(len(master_df), 0)
            self.assertGreaterEqual(len(panel_specs), 9)
            self.assertIsNotNone(meta.get("global_btx_intensity_otsu"))
        finally:
            plt.close(fig)


def smoke_test_one_image() -> None:
    """Load one CZI and verify artifacts land only under output/."""
    import numpy as np
    import pandas as pd

    data_root = os.path.join(ROOT, "data")
    dataset = os.path.join(data_root, "SP11BTXSAA")
    czi_name = "0722Mplate-01(20).czi"
    czi_path = os.path.join(dataset, czi_name)
    if not os.path.isfile(czi_path):
        print("SKIP smoke test: sample CZI not found")
        return

    try:
        from nmj_master_dashboard import load_confocal_image
    except ImportError as exc:
        print(f"SKIP smoke test: optional imaging dependency missing ({exc})")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_root = os.path.join(tmp, "output")
        run_dir = create_run_output_dir(root=out_root, when=datetime(2026, 6, 26, 15, 0, 0))
        data_root_abs = os.path.abspath(data_root)

        with open(os.path.join(dataset, "channel_mapping_config.json"), encoding="utf-8") as jf:
            fc = json.load(jf)[czi_name]

        try:
            channels = load_confocal_image(
                czi_path,
                channel_indices=[fc["m"], fc["n"], fc["b"]],
            )
        except ImportError as exc:
            print(f"SKIP smoke test: optional imaging dependency missing ({exc})")
            return
        btx = channels[fc["b"]]
        spot_count = int(np.count_nonzero(btx > np.percentile(btx, 99.5)))

        file_stem = os.path.splitext(czi_name)[0]
        out_csv = mirror_dataset_output_path(run_dir, data_root_abs, dataset, f"{file_stem}_analysis.csv")
        pd.DataFrame({"spot_proxy": [spot_count]}).to_csv(out_csv, index=False)

        snap = snapshot_channel_mappings(data_root_abs, [dataset])
        save_run_config_files(
            run_dir,
            {"run_id": os.path.basename(run_dir), "mode": "smoke_test", "spot_proxy": spot_count},
            snap,
        )

        for name in os.listdir(dataset):
            if name.endswith("_analysis.csv") or name.endswith("_NMJ_Plot.png"):
                raise AssertionError(f"Unexpected artifact in data/: {name}")

        assert os.path.isfile(out_csv), "per-image CSV missing in output run folder"
        assert os.path.isfile(os.path.join(run_dir, "run_config.json"))
        assert os.path.isfile(os.path.join(run_dir, "channel_mapping_config.json"))
        print(f"OK smoke test: spot_proxy={spot_count} -> {out_csv}")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RunOutputHelpersTest)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(OtsuAbundanceStatsTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(GlobalBtxOtsuThresholdTest))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(DashboardFunctionalityTest))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    try:
        smoke_test_one_image()
    except Exception as exc:
        print(f"FAIL smoke test: {exc}")
        raise
    print("All run-output consistency checks passed.")
