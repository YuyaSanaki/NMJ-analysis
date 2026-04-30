# Neuromuscular Junction (NMJ) Analysis Pipeline

A fully containerized image analysis toolkit that detects, measures, and classifies Neuromuscular Junctions (NMJs) from multi-channel confocal `.czi` files—intended as a streamlined alternative to FIJI / TrackMate-heavy workflows.

## System Architecture

The app runs in Docker with **Streamlit** UIs. **Single-image** and **batch** are separate Compose **profiles**: you start **one** service at a time so only **one** process and **one** port are active, and the configured **memory limit** applies to that container (helpful for large tiles and deep Z-stacks).

| Profile | Command | App |
|--------|---------|-----|
| `single` | `docker compose --profile single up --build` | `BTX.py` — one `.czi`, tune detection and thresholds |
| `batch` | `docker compose --profile batch up --build` | `BTX_batch.py` — whole folders, JSON configs, per-file mapping |

**URL (both modes):** `http://localhost:8501`

Plain `docker compose up` with **no** `--profile` does not start either app (by design).

## Quick Start

1. **Start one pipeline** (examples above). Add `-d` to run in the background.
2. Open **`http://localhost:8501`** in the browser.
3. **Data:** Put `.czi` files in subfolders under this project. The compose file bind-mounts the repo to `/app` inside the container.
4. **Stop:** `Ctrl+C` in the terminal, or from another shell:

   ```bash
   docker compose --profile single down
   # or
   docker compose --profile batch down
   ```

5. **Switch modes:** bring the running stack down, then start the other profile.

## Docker Memory

`docker-compose.yml` sets a **12 GB** memory limit per service. If you see exits with code **137** (OOM), raise **Docker Desktop → Settings → Resources → Memory** so the Linux VM can supply that headroom and leave margin for the host OS.

For very long batch runs, use the **"Save per-image NMJ_Plot PNGs"** option wisely (disabling it reduces peak memory).

---

## Features & Methodologies

### 1. Robust spot detection (Difference of Gaussians)

The pipeline uses skimage's `blob_dog` on the BTX (receptor) channel. A **morphological white top-hat** (rolling-ball–style background suppression) runs first when enabled.

Spot size in the UI is expressed as **diameter in μm** (min / max). Internally, diameters are converted to Gaussian sigmas for DoG:

```
sigma_um = diameter_um / (2 × sqrt(2))
```

**Auto DoG threshold:** `estimate_dog_threshold()` computes the median of positive pixel values (> 0.02) in a subsampled BTX image and applies a `× 0.4` factor, clamped between `0.03` and `0.08`. This "top-half" approach adapts to varied image brightness without an explicit per-image slider.

### 2. Physical units (μm)

Pixel size **μm/pixel** is read from `.czi` metadata where possible. Distances and spot radii are reported in **micrometers**.

On large spot-diameter settings, the code may downscale for DoG stability and maps spots back to full resolution; DoG sigma and background radius are capped to limit memory use.

### 3. Biological and spatial metrics

For each spot, segmentation uses a **fixed threshold tied to DoG detection** (normalized threshold × 99.9th percentile of haze-subtracted BTX), with Otsu only as a fallback on degenerate crops—this avoids splitting dim spot rims on black-dominated windows.

| Column | Description |
|--------|-------------|
| `Dist_to_Muscle_um` | Edge-corrected EDT: center-to-mask distance minus spot radius (µm), clamped at 0 |
| `Dist_to_Neuron_um` | Same, for neuron mask |
| `Dist_to_Muscle_center_um` | EDT at the blob center only (µm), for QC/comparison |
| `Dist_to_Neuron_center_um` | EDT at the blob center only (µm), for QC/comparison |
| `INNERVATION_OVERLAP_PCT` | Overlap fraction (%) of the spot mask with the neuron channel |
| `MEAN_INTENSITY` | Mean raw intensity inside the spot mask (haze-subtracted BTX) |
| `ROUNDNESS` | `1 − eccentricity` derived from the inertia tensor eigenvectors (see below) |
| `RADIUS` | Spot radius in µm |
| `Resolution_Class` | `"Low-Res"` if pixel size > 0.5 µm/px, otherwise `"High-Res"` |

**Muscle haze removal:** subtracts a wide Gaussian from the BTX channel. The haze σ (µm) is `max(50, 5 × max spot diameter)` so large plaques are less likely to show a "donut" after subtraction (not a separate control).

Batch outputs use every DoG detection that passes diameter filtering—there is **no** muscle-vs-BTX intensity rejection step.

### 4. Shape metric: ROUNDNESS (1 − eccentricity)

Shape is now reported as **ROUNDNESS = 1 − eccentricity**, computed from the inertia tensor eigenvectors (`skimage.measure.regionprops`, `inertia_tensor_eigvals`). A value of `1.0` is a perfect circle; values near `0` are highly elongated.

This replaces the legacy perimeter-based `CIRCULARITY` (`4πA/P²`), which was unreliable at low resolution because pixelated edges artificially inflate the perimeter even for round objects.

**Resolution gating:** any spot whose segmented mask has fewer than `MIN_PIXELS_FOR_SHAPE = 20` pixels gets `ROUNDNESS = NaN`. These rows are excluded from Roundness KDE plots via `dropna`, preventing noisy low-resolution spots from flattening the distribution.

**Backward compatibility:** CSVs written before the terminology update that contain a `CIRCULARITY` column (but no `ROUNDNESS` column) are automatically aliased — `ROUNDNESS = CIRCULARITY` — so existing result files still load correctly in the summary dashboards.

### 5. BTX signal classification (4 classes)

Each detected spot is assigned one of four classes based on its edge-corrected distances to the muscle and neuron masks:

| Class | Color | Condition |
|-------|-------|-----------|
| **NMJ** | red | near both muscle **and** neuron |
| **Aneural AChR clusters** | green | near muscle only |
| **Neuron-associated BTX signal** | blue | near neuron only |
| **Orphaned** | gray | near neither mask |

"Near" is defined by the **Functional NMJ Boundary (μm)** slider (default `1.0 µm`), applied to edge-corrected distances.

**Legacy label aliasing:** result CSVs written with older terminology (`"Muscle Only"`, `"Neuron Only"`, etc.) are silently renamed to the current terms when loaded, so older data files plot correctly alongside new ones.

### 6. Statistical tests

**Fisher's Exact Test (proximity):** For each image (and globally across a batch run), a 2×2 contingency table is built from the four spot classes and Fisher's Exact Test is computed. The p-value and significance star (`***` / `**` / `*` / `ns`) appear in the proximity scatter plot title.

**Paired Wilcoxon test (intensity):** Median `MEAN_INTENSITY` is computed per `SOURCE_IMAGE` for NMJ and Orphaned spots. A one-sided paired Wilcoxon signed-rank test (`alternative="greater"`) tests whether NMJ intensity is higher than Orphaned intensity. The result appears in the intensity KDE panel title (Panel 5 of the 9-panel figure).

---

## Using the Batch System

With **`docker compose --profile batch up --build`**:

1. Select a folder that contains your `.czi` files.
2. Set the **config template** (muscle / neuron / BTX channels) at the top.
3. Use **"Paste Template to ALL Images"** or per-file expanders. Pixel sizes from metadata are not overwritten by paste where noted.
4. **Exclude** bad files with the skip checkboxes if needed; **Save Settings to Folder** writes `channel_mapping_config.json` for the next run.
5. **Run Batch Analysis (Current Folder)** or **(ALL Folders)** as required.

**Recursive `.czi` discovery:** the batch runner uses `collect_czi_jobs()` to walk subdirectories, so `.czi` files nested inside sub-folders (e.g. `Data/Cond1/slide/image.czi`) are included automatically.

**Per-folder configs:** when running ALL Folders, each folder's `channel_mapping_config.json` is loaded automatically if present, so each dataset keeps its own channel mapping.

---

## Outputs (batch)

Artifacts are written next to the data. **ALL Folders** runs additionally write aggregate files in the **project root** (the app working directory, `/app` in Docker).

| File | Scope | Description |
|------|-------|-------------|
| `ALL_FOLDERS_MASTER_RESULTS.csv` | project root | Combined spot table for all-folder runs, with `SOURCE_FOLDER` and `SOURCE_IMAGE` columns |
| `ALL_FOLDERS_SUMMARY.png` | project root | 6-panel aggregate dashboard (see below) |
| `ALL_FOLDERS_SUMMARY_TABLE.csv` | project root | Per-folder summary statistics |
| `BATCH_MASTER_RESULTS.csv` | selected folder | Combined spot table for current-folder runs |
| `BATCH_SUMMARY.png` | selected folder | 6-panel dashboard for the current folder |
| `[Filename]_analysis.csv` | next to image | Per-image spot table |
| `[Filename]_NMJ_Plot.png` | next to image | 10-panel figure (optional; can be disabled) |

### Per-image 10-panel figure layout (4 × 3 grid)

| Row | Col 0 | Col 1 | Col 2 |
|-----|-------|-------|-------|
| 0 | **1. Proximity scatter** (Fisher P) + marginal KDEs | **2. Size KDE** (radius by class) | **3. Roundness KDE** (1 − eccentricity, valid spots only) |
| 1 | **4. Innervation Distribution** (overlap %) | **5. Receptor Intensity KDE** (Wilcoxon P) | **6. Raw BTX (L) \| Cleaned BTX (R)** |
| 2 | **7. Cleaned BTX** | **8. Cleaned BTX + Detected Spots** | **9. Composite + All Spots** |
| 3 | **10. Composite + Functional NMJs only** | *(density bar chart)* | *(unused)* |

### ALL_FOLDERS summary dashboard (3 × 2 grid)

| Panel | Content |
|-------|---------|
| 1. NMJ Formation Rate by Folder | Bar chart — NMJ rate (%) per folder |
| 2. Total BTX Spots by Folder | Bar chart — total spot count per folder |
| 3. Mean Spot Radius by Folder | Bar chart — mean radius (µm) per folder |
| 4. Mean Innervation Overlap by Folder | Bar chart — mean overlap % per folder |
| 5. Median Distance to Muscle/Neuron | Bar chart — median edge-corrected distances |
| 6. Global Proximity Scatter | Scatter + marginal KDEs across all folders (Fisher P) |
