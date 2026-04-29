# 🔬 Neuromuscular Junction (NMJ) Analysis Pipeline

A fully containerized image analysis toolkit that detects, measures, and classifies Neuromuscular Junctions (NMJs) from multi-channel confocal `.czi` files—intended as a streamlined alternative to FIJI / TrackMate–heavy workflows.

## 🏗 System Architecture

The app runs in Docker with **Streamlit** UIs. **Single-image** and **batch** are separate Compose **profiles**: you start **one** service at a time so only **one** process and **one** port are active, and the configured **memory limit** applies to that container (helpful for large tiles and deep Z-stacks).

| Profile | Command | App |
|--------|---------|-----|
| `single` | `docker compose --profile single up --build` | `BTX.py` — one `.czi`, tune detection and thresholds |
| `batch` | `docker compose --profile batch up --build` | `BTX_batch.py` — whole folders, JSON configs, per-file mapping |

**URL (both modes):** `http://localhost:8501`

Plain `docker compose up` with **no** `--profile` does not start either app (by design).

## 🚀 Quick Start

1. **Start one pipeline** (examples above). Add `-d` to run in the background.

2. Open **`http://localhost:8501`** in the browser.

3. **Data:** Put `.czi` files in subfolders under this project. The compose file bind-mounts the repo to `/app` inside the container.

4. **Stop:** `Ctrl+C` in the terminal, or from another shell:

   ```bash
   docker compose --profile single down
   # or
   docker compose --profile batch down
   ```

5. **Switch modes:** bring the running stack down, then start the other profile (single vs batch).

## 🧠 Docker memory

`docker-compose.yml` sets a **12 GB** memory limit per service. If you still see exits with code **137** (OOM), raise **Docker Desktop → Settings → Resources → Memory** so the Linux VM can supply that headroom and leave margin for the host OS.

For very long batch runs, use the **“Save per-image NMJ_Plot PNGs”** option wisely (disabling it reduces peak memory).

---

## 🎨 Features & Methodologies

### 1. Robust spot detection (Difference of Gaussians)

The pipeline uses skimage’s `blob_dog` on the BTX (receptor) channel. A **morphological white top-hat** (rolling-ball–style background suppression) runs first when enabled.

Spot size in the UI is expressed as **diameter in μm** (min / max). Internally, diameters are converted to Gaussian sigmas for DoG:

`sigma_um = diameter_um / (2 * sqrt(2))`

### 2. Physical units (μm)

Pixel size **μm/pixel** is read from `.czi` metadata where possible. Distances and spot radii are reported in **micrometers**.

On large spot-diameter settings, the code may downscale for DoG stability and map spots back to full resolution; DoG sigma and background radius are capped to limit memory use.

### 3. Biological and spatial metrics

For each spot, segmentation uses a **fixed threshold tied to DoG detection** (normalized threshold × 99.9th percentile of haze-subtracted BTX), with Otsu only as a fallback on degenerate crops—this avoids splitting dim spot rims on black-dominated windows.

* **`Dist_to_Muscle_um` / `Dist_to_Neuron_um`:** Edge-corrected EDT: center-to-mask distance minus spot radius (µm), clamped at 0. Used for NMJ gating and Fisher-style proximity summaries.
* **`Dist_to_Muscle_center_um` / `Dist_to_Neuron_center_um`:** EDT at the blob center only (µm), for comparison and QC.
* **`INNERVATION_OVERLAP_PCT`:** Overlap of the spot mask with the neuron channel.
* **`MEAN_INTENSITY`:** Mean raw intensity inside the spot mask (haze-subtracted BTX).
* **`CIRCULARITY`:** `4π·area / perimeter_crofton²` on the segmented spot region.

**Muscle haze removal:** subtracts a wide Gaussian from the BTX channel. The haze σ (µm) is **max(50, 5 × max spot diameter)** from the DoG UI so large plaques are less likely to show a “donut” after subtraction (not a separate control).

Batch outputs use every DoG detection that passes diameter filtering—there is **no** muscle-vs-BTX intensity rejection step.

---

## 🔁 Using the batch system

With **`docker compose --profile batch up --build`**:

1. Select a folder that contains your `.czi` files.
2. Set the **config template** (muscle / neuron / BTX channels) at the top.
3. Use **“Paste Template to ALL Images”** or per-file expanders. Pixel sizes from metadata are not overwritten by paste where noted.
4. **Exclude** bad files with the skip checkboxes if needed; **Save Settings to Folder** writes `channel_mapping_config.json` for the next run.
5. **Run Batch Analysis (Current Folder)** or **(ALL Folders)** as required.

---

## 📊 Outputs (batch)

Artifacts are written next to the data (and optional project-wide master CSVs when using **ALL Folders**):

1. **`BATCH_MASTER_RESULTS.csv`** (or **`ALL_FOLDERS_MASTER_RESULTS.csv`**) — combined spot table with `SOURCE_FOLDER` / `SOURCE_IMAGE` where applicable.
2. **`[Filename]_analysis.csv`** — per-image spot table.
3. **`[Filename]_NMJ_Plot.png`** — 9-panel figure (optional during batch; can be turned off in the UI).
4. Summary PNGs such as **`BATCH_SUMMARY.png`** / **`ALL_FOLDERS_SUMMARY.png`** when a batch run completes with data.

Row layout of the per-image 9-panel plot: row 1 — proximity scatter, size KDE, circularity KDE; row 2 — innervation / intensity distributions and BTX view; row 3 — BTX with spots, composite with spots, composite with NMJ-only arrows.
