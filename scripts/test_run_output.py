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
    build_run_zip_bytes,
    create_run_output_dir,
    iter_run_files,
    mirror_dataset_output_path,
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
        # 10 spots / 4000 µm² * 1000 and 2 spots / 2000 µm² * 1000
        self.assertAlmostEqual(by_folder["FolderA"], 2.5)
        self.assertAlmostEqual(by_folder["FolderB"], 1.0)

        # Buggy path would assign 12 spots to both rows -> 3.0 and 6.0
        self.assertNotAlmostEqual(by_folder["FolderA"], 3.0)
        self.assertNotAlmostEqual(by_folder["FolderB"], 6.0)


def smoke_test_one_image() -> None:
    """Load one CZI and verify artifacts land only under output/."""
    import numpy as np
    import pandas as pd

    from nmj_master_dashboard import load_confocal_image

    data_root = os.path.join(ROOT, "data")
    dataset = os.path.join(data_root, "SP11BTXSAA")
    czi_name = "0722Mplate-01(20).czi"
    czi_path = os.path.join(dataset, czi_name)
    if not os.path.isfile(czi_path):
        print("SKIP smoke test: sample CZI not found")
        return

    with tempfile.TemporaryDirectory() as tmp:
        out_root = os.path.join(tmp, "output")
        run_dir = create_run_output_dir(root=out_root, when=datetime(2026, 6, 26, 15, 0, 0))
        data_root_abs = os.path.abspath(data_root)

        with open(os.path.join(dataset, "channel_mapping_config.json"), encoding="utf-8") as jf:
            fc = json.load(jf)[czi_name]

        channels = load_confocal_image(
            czi_path,
            channel_indices=[fc["m"], fc["n"], fc["b"]],
        )
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
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    try:
        smoke_test_one_image()
    except Exception as exc:
        print(f"FAIL smoke test: {exc}")
        raise
    print("All run-output consistency checks passed.")
