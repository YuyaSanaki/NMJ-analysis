# Neuromuscular Junction (NMJ) Analysis Pipeline

A fully containerized image analysis toolkit that detects, measures, and classifies Neuromuscular Junctions (NMJs) from multi-channel z-stack confocal `.czi` files.

`.czi`  file should have 1. muscle staining channel, 2. neuron staining channel, and 3. alpha-bungarotoxin staining channel. Image can have extra channel but not be used in this pipeline. Strongly recommend to take high resolution image with high maginification lens (Bit: 16, Image size: ≥2000x2000, Lens: ≥40x, Z-stack image). Z-stack will convert to max projection inside the pipeline.

![Example image](readme/iamge.png)

Intended as a streamlined alternative to FIJI / TrackMate-heavy workflows.

## System Architecture

The app runs in Docker with **Streamlit** UIs. **Single-image** and **batch** are separate Compose **profiles**: you start **one** service at a time so only **one** process and **one** port are active, and the configured **memory limit** applies to that container (helpful for large tiles and deep Z-stacks).



Plain `docker compose up` with **no** `--profile` does not start either app (by design).

## Quick Start

1. **Data:** Put `.czi` files in subfolders under this project. e.g. `/Users/username/NMJanalysis/Experiment1/image.czi`. The compose file bind-mounts the repo to `/app` inside the container.
2. **Start one pipeline** Open terminal and
```bash
   cd /Users/username/NMJanalysis/Experiment1/

   docker compose --profile single up --build
   # or
   docker compose --profile batch up --build
   ```
   Keep the terminal window open during your analysis and use the terminal window for the rest of the commands.

3. Open the app in the browser (check the terminal for the exact URL):
   - **Batch:** `http://localhost:8503`
   - **Single-image:** `http://localhost:8504`

   Port 8501 is not used by this project (it may be occupied by another app on your machine).

4. Set up analysis configurations in the browser.

   - **Channel Setup:** Assign the muscle, nerve, and BTX imaging channels for each dataset folder. (Note: The pixel size will be detected automatically).
   - **Spot Detection:** Configure your spot detection parameters. The default NMJ spot size range is 5–12 µm. It is highly recommended to enable Auto Threshold per image and Auto-Optimize Background Subtraction Radius.
   - **Validation Plots:** Enabling the option to save per-image NMJ plot PNGs during batch processing will consume additional memory, but it generates a valuable validation plot for every individual image.
   - **Spot detection Threshold:** If the muscle or nerve staining appears faint, increase the DoG threshold (Detection Threshold and DoG sigma). Because this adjustment can introduce artifacts, be sure to review the per-image NMJ plots to verify your results.
   - **NMJ Logic:** This setting defines the acceptable distance from the BTX signal to assume that the BTX staining is correctly associated with the surrounding proximity tissues (i.e., the muscle and nerve).
   - **The output** is written under `output/<YYYYMMDD_HHMMSS>/` (timestamped run folder). Each run also saves `run_config.json` and a snapshot of `channel_mapping_config.json`. Only `.czi` files and the live channel config remain under `data/`.


5. **Stop:** `Ctrl+C` in the terminal, or 

   ```bash
   docker compose down
   ```


## Docker Memory

`docker-compose.yml` sets a **12 GB** memory limit per service. If you see exits with code **137** (OOM), raise **Docker Desktop → Settings → Resources → Memory** so the Linux VM can supply that headroom and leave margin for the host OS.

For very long batch runs, use the **"Save per-image NMJ_Plot PNGs"** option wisely (disabling it reduces peak memory).

---

## Features & Methodologies

### 1. Robust spot detection (Difference of Gaussians)

The pipeline subtracts broad diffuse background from the BTX channel using a large-sigma Gaussian blur. A smoothed background image is computed with σ = max(50 µm, 5 × max spot diameter), which is well above the largest expected cluster, and then subtracted from the raw image. This removes wide haze and muscle auto-fluorescence while preserving sharp puncta, and avoids the "donut" hollowing artifact that occurs when the background kernel is too close in size to the signal.

![Background subtraction](readme/backgroundsubtraction.png)

The pipeline uses skimage's `blob_dog` on the BTX (receptor) channel. A **morphological white top-hat** (rolling-ball–style background suppression) runs first when enabled.

Spot size in the UI is expressed as **diameter in μm** (min / max). Internally, diameters are converted to Gaussian sigmas for DoG:

```
sigma_um = diameter_um / (2 × sqrt(2))
```

**Auto DoG threshold:** `estimate_auto_threshold()` computes `median + 3 × (1.4826 × MAD)` on positive pixel values (> 0.005) in a subsampled, haze-subtracted BTX image, clamped to `[0.02, 0.12]`. **Auto threshold sensitivity** sets skimage `blob_dog` **`sigma_ratio`**: **Conservative (1.6)** or **High (1.3)** (shown explicitly in the UI). With a **manual** detection threshold, use **DoG sigma ratio (manual)** (default **1.6**, same as Auto Conservative).

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

**Backward compatibility:** Older result CSVs are normalized on load: legacy `CIRCULARITY` → `ROUNDNESS`; legacy BTX class labels (`NMJ`, `Aneural AChR clusters`, etc.) → current names; legacy file-stats column names (`NMJs (Both)`, `Density_NMJ`, …) → current names.

### 5. BTX signal classification (4 classes)

Each detected spot is assigned one of four classes based on its edge-corrected distances to the muscle and neuron masks:

| Class | Color | Condition |
|-------|-------|-----------|
| **early NMJ-like** | red | near both muscle **and** neuron |
| **Muscle-associated** | green | near muscle only |
| **Neuron-associated** | blue | near neuron only |
| **Orphaned** | gray | near neither mask |

"Near" is defined by the **Functional NMJ Boundary (μm)** slider (default `1.0 µm`), applied to edge-corrected distances.


### 6. Statistical tests

All p-values are annotated with significance stars: `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05 · `ns` p ≥ 0.05.

**Methods (inference unit).** Because multiple BTX puncta within a single image are not independent, primary statistical comparisons were performed on per-image summary statistics (e.g. class-specific medians); spot-level tests are reported for visualization only.

**Where results are saved**

| File | Description |
|------|-------------|
| `*_STAT_SUMMARY[_thrTag].csv` | Full table: metric, comparison, test, p-value, `level` column |
| `*_IMAGE_LEVEL_MEDIANS[_thrTag].csv` | Per-image, per-class medians used for primary tests |
| `*_OTSU_DIM_NOISE_REJECTION[_thrTag].csv` | Spot composition table: % of each class above global Otsu |
| Aggregate `*_SUMMARY[_thrTag].png` titles | Image-level Kruskal–Wallis on proximity when batch data includes `SOURCE_IMAGE` |

**`level` column in STAT_SUMMARY**

| Level | Role |
|-------|------|
| `primary_image_level` | **Primary inference** — tests on per-image class medians (unit of replication = image) |
| `primary_posthoc` | Pairwise follow-ups to primary tests (Holm–Bonferroni adjusted) |
| `primary_sensitivity` | Unpaired image-level alternatives (e.g. Mann–Whitney across different image sets) |
| `exploratory_spot_pooled` | **Exploratory only** — all spots pooled; supports plot titles, not primary claims |

**Primary tests (batch)**

| Metric | Comparison | Test |
|--------|------------|------|
| Proximity (`Dist_to_Muscle_um`, `Dist_to_Neuron_um`) | 4 BTX classes | Kruskal–Wallis on per-image class medians; Mann–Whitney posthoc early NMJ-like vs others |
| Roundness | early NMJ-like vs Muscle-associated vs Neuron-associated | Kruskal–Wallis on per-image class medians |
| BTX intensity | early NMJ-like vs Orphaned | **Wilcoxon signed-rank (paired within image)** on class medians; unpaired Mann–Whitney in `primary_sensitivity` |
| Fraction above Otsu | early NMJ-like vs Orphaned (paired within image) | Wilcoxon on per-image fraction of spots ≥ global Otsu |
| Zone abundance | 4 zones per image | Friedman (repeated measures); Conover–Iman posthoc |

Exploratory spot-pooled Kruskal / Mann–Whitney rows for proximity, roundness, and intensity are included in the same CSV for comparison with figure KDEs.

**Single-image app (`BTX.py`):** proximity panel titles use spot-level Kruskal–Wallis (no `SOURCE_IMAGE` column). Use the batch pipeline and `STAT_SUMMARY` for publication inference.

**Proximity scatter:** when many spots share the same edge-corrected distance (often after clipping to 0 µm), the scatter plot applies a **small display-only jitter** so markers do not stack invisibly; **CSV values and marginal KDEs use the true coordinates**.

---

**Kruskal–Wallis (proximity) — aggregate Panel 1**

*What it asks:* Do edge-corrected distances to muscle and neuron masks differ across the four BTX signal classes?

*Primary inference:* For each image and class, the median distance is computed across spots; Kruskal–Wallis compares those image-level medians across classes (≥ 3 images per class required). Mann–Whitney posthoc tests compare early NMJ-like image medians to each other class (Holm–Bonferroni adjusted).

*Exploratory:* The same test on all spots pooled appears in `STAT_SUMMARY` as `exploratory_spot_pooled` and supports marginal KDE visualization.

---

**Kruskal–Wallis (roundness) — Panel 3**

*What it asks:* Does receptor-cluster morphology differ across **early NMJ-like**, **Muscle-associated**, and **Neuron-associated**?

*Primary inference:* Per-image class medians of `ROUNDNESS` (spots with `AREA_PX ≥ MIN_PIXELS_FOR_SHAPE` only). Exploratory spot-pooled Kruskal is also recorded.

---

**Mann–Whitney U (intensity) — Panel 5**

*What it asks:* Are **early NMJ-like** spots brighter than **Orphaned** spots?

*Primary inference:* **Paired Wilcoxon signed-rank** on per-image class medians (`alternative="greater"`) in images that contain both classes — this tests whether synaptic/muscle-associated puncta are brighter than distant puncta **within the same field of view**. Unpaired Mann–Whitney rows are retained as `primary_sensitivity`. Otsu-filtered paired variants use only spots at/above the global batch Otsu. Exploratory spot-pooled Mann–Whitney rows are included for plot titles.

---

**Otsu dim-noise rejection — `*_OTSU_DIM_NOISE_REJECTION[_thrTag].csv`**

BTX staining can be **dirty**: many detected puncta are non-specific haze or noise. True AChR-related signal is identified by **converging evidence** — spatial co-localization (muscle / neuron masks), zone enrichment (Friedman), and intensity contrast — not by Otsu alone.

The dim-noise rejection table reports, for each BTX signal class, what fraction of detected spots exceed the **global intensity Otsu** threshold computed across all spots in the batch run. Example interpretation:

| Class | Typical pattern | How to read it |
|-------|-----------------|----------------|
| **early NMJ-like** | High % above Otsu (~80–90%) | Synaptic puncta near muscle and neuron are predominantly bright — consistent with specific AChR staining rather than dim haze. |
| **Muscle-associated** | High % above Otsu (~75–85%) | Muscle-proximal AChR clusters pass the intensity gate; supports that muscle-associated signal is real amid dirty stain. |
| **Neuron-associated** | High % above Otsu (~80–90%) | Neuron-proximal puncta are mostly bright; may include presynaptic or developing terminals, not just noise. |
| **Orphaned** | Low % above Otsu (~25–40%) | Puncta far from both masks are mostly **dim** — consistent with non-specific background. The minority above Otsu may include brighter immature or mislocalized BTX, so Orphaned is **not** a pure noise class. |

**Important caveats**

1. **Descriptive, spot-pooled:** The table counts spots, not images. It supports the biological story but does not replace image-level tests (spots within one image are correlated).
2. **Otsu is a dim-spot filter, not a synapse classifier:** A bright spot far from muscle/neuron still passes Otsu but remains Orphaned spatially. Conversely, dim true AChR could fall below Otsu. Spatial class + zone Friedman remain the strongest primary evidence.
3. **Orphaned heterogeneity:** Low Otsu pass rate does not mean Orphaned is “clean background” — it means most distant puncta are dim; some may be biological.
4. **Paired statistical follow-up:** `STAT_SUMMARY` includes a **paired Wilcoxon** on per-image fraction above Otsu (early NMJ-like vs Orphaned) to test whether synaptic puncta are more often “bright enough” than distant puncta within the same image.

**Recommended narrative for a paper:** Global Otsu separates predominantly dim distant puncta from predominantly bright muscle/synapse-associated puncta; paired image-level Wilcoxon confirms early NMJ-like > Orphaned intensity within fields that contain both; Friedman zone abundance confirms BTX is enriched at NMJ/muscle zones — together supporting identification of true AChR signal amid dirty staining.

---

**Friedman (zone abundance) — aggregate abundance panels**

*What it asks:* Does BTX spot density differ across tissue zones **within the same image**?

*How it works:* Per-image zone counts are area-normalised (spots per 1000 µm²). Friedman tests the four zone abundances as repeated measures; Conover–Iman posthoc with Holm adjustment. A second Friedman uses only spots at/above the global intensity Otsu threshold. These are tagged `primary_image_level` / `primary_posthoc` in `STAT_SUMMARY`.

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

Each run creates a timestamped folder: `output/<YYYYMMDD_HHMMSS>/`. Dataset subfolders mirror your `data/` layout; aggregate files sit at the run root. Streamlit offers ZIP and per-file downloads when the run finishes.

When **Auto Threshold per image** is enabled, filenames include a sensitivity tag so Conservative and High runs do not overwrite each other: `_thrConservative` or `_thrHigh` (e.g. `ALL_FOLDERS_MASTER_RESULTS_thrHigh.csv`). With manual threshold, that tag is omitted.

| File | Scope | Description |
|------|-------|-------------|
| `run_config.json` | run root | Run metadata (thresholds, mode, timestamps) |
| `channel_mapping_config.json` | run root | Snapshot of channel mappings used |
| `ALL_FOLDERS_MASTER_RESULTS[_thrTag].csv` | run root | Combined spot table (ALL Folders), with `SOURCE_FOLDER`, `SOURCE_IMAGE`, `TOTAL_IMAGE_AREA_um2` |
| `ALL_FOLDERS_STAT_SUMMARY[_thrTag].csv` | run root | Primary + exploratory statistical tests (`level` column) |
| `ALL_FOLDERS_IMAGE_LEVEL_MEDIANS[_thrTag].csv` | run root | Per-image class medians for primary inference |
| `ALL_FOLDERS_OTSU_DIM_NOISE_REJECTION[_thrTag].csv` | run root | % spots above global Otsu by class (dim-noise composition) |
| `ALL_FOLDERS_SUMMARY[_thrTag].png` | run root | Aggregate dashboard figure |
| `ALL_FOLDERS_SUMMARY_TABLE[_thrTag].csv` | run root | Per-folder summary statistics |
| `ALL_FOLDERS_FILE_STATS[_thrTag].csv` | run root | Per-image zone-density table |
| `BATCH_*` counterparts | run / dataset folder | Same artifacts for **Current Folder** runs |
| `[Filename][_thrTag]_analysis.csv` | under dataset in run | Per-image spot table |
| `[Filename][_thrTag]_NMJ_Plot.png` | under dataset in run | 10-panel figure (optional) |

### Per-image 10-panel figure layout (4 × 3 grid)

| Row | Col 0 | Col 1 | Col 2 |
|-----|-------|-------|-------|
| 0 | **1. Proximity scatter** (image-level Kruskal in batch aggregate) + marginal KDEs | **2. Size KDE** (radius by class) | **3. Roundness KDE** (3-way Kruskal + medians) |
| 1 | **4. Innervation Distribution** (overlap %) | **5. Receptor Intensity KDE** (Mann-Whitney P) | **6. Raw BTX (L) \| Cleaned BTX (R)** |
| 2 | **7. Cleaned BTX** | **8. Cleaned BTX + Detected Spots** | **9. Composite + All Spots** |
| 3 | **10. Composite + early NMJ-like only** | *(density bar chart)* | *(unused)* |

### Aggregate dashboard layout (`BATCH_SUMMARY*.png` / `ALL_FOLDERS_SUMMARY*.png`)

After a batch finishes, these files mirror the **Global** figure shown in Streamlit.

| Row | Left | Right |
|-----|------|-------|
| **0** | BTX intensity histogram + global Otsu line (full width) | |
| **1** | Proximity scatter + marginal KDEs (image-level Kruskal in title) | Size KDE |
| **2** | Roundness KDE | Innervation overlap |
| **3** | Intensity KDE (all spots) | Intensity KDE (Otsu-filtered) |
| **4** | Zone abundance (all spots; Friedman in title) | Zone abundance (Otsu-filtered) |
| **5** | *(ALL Folders only)* Per-image early NMJ-like rate control chart (full width) | |

**Current folder** runs use **5 rows** (no control chart). **ALL Folders** adds row 5 for the early NMJ-like rate control chart.
