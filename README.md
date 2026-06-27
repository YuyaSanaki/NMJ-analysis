# Neuromuscular Junction (NMJ) Analysis Pipeline

A containerized toolkit for detecting, measuring, and classifying BTX-labeled puncta from multi-channel confocal images. Two **Streamlit** apps share the same analysis core:

| App | Compose profile | URL | Use case |
|-----|-----------------|-----|----------|
| **Batch** (`BTX_batch.py`) | `batch` | http://localhost:8503 | Process a folder or all datasets; publication stats |
| **Single-image** (`BTX.py`) | `single` | http://localhost:8504 | Interactive QC on one file |

Plain `docker compose up` (no profile) starts **neither** app.

![Example image](readme/iamge.png)

---

## Input requirements

Each image needs three channels used by the pipeline:

1. **Muscle** staining  
2. **Neuron** staining  
3. **BTX** (α-bungarotoxin / AChR) staining  

Extra channels are ignored. Recommended acquisition: 16-bit, ≥ 2000×2000 px, ≥ 40× objective, Z-stack (max projection inside the pipeline).

**Supported formats:** `.czi`, `.nd2`, `.lif`, `.oir`, `.poir`, `.tif`, `.tiff` (see `collect_image_jobs()` in `nmj_master_dashboard.py`).

---

## Repository layout

```
NMJ-analysis/
├── data/                          # Inputs only (gitignored): images + live channel_mapping_config.json
├── output/                        # Timestamped run folders (gitignored)
├── BTX_batch.py                   # Batch Streamlit UI
├── BTX.py                         # Single-image Streamlit UI
├── nmj_master_dashboard.py        # Aggregate figures, stats, image I/O helpers
├── nmj_run_output.py              # output/<timestamp>/ helpers, ZIP downloads
├── regenerate_all_folders_panel_pdfs.py
├── scripts/test_run_output.py
├── docker-compose.yml
└── Dockerfile
```

**Data vs output:** Raw images and the editable `channel_mapping_config.json` stay under `data/<dataset>/`. Every analysis run writes artifacts to `output/<YYYYMMDD_HHMMSS>/`, including a snapshot of the channel config and `run_config.json`.

---

## Quick start

1. Place images in subfolders under `data/`, e.g. `data/Experiment1/slide01.czi`.
2. From the project root:

```bash
docker compose --profile batch up --build
# or
docker compose --profile single up --build
```

3. Open the URL printed in the terminal (8503 batch / 8504 single).
4. Stop with `Ctrl+C` or `docker compose down`.

### Docker memory

`docker-compose.yml` sets a **12 GB** limit per service. Exit code **137** usually means OOM — increase **Docker Desktop → Settings → Resources → Memory**. For long batch runs, disabling **Save per-image NMJ_Plot PNGs** lowers peak RAM.

---

## Batch workflow

With `docker compose --profile batch up --build`:

1. Select a dataset folder under `data/`.
2. Set the **channel template** (muscle / neuron / BTX) at the top.
3. **Paste Template to ALL Images** or configure per-file expanders (pixel size from metadata is preserved where noted).
4. Skip bad files with checkboxes; **Save Settings to Folder** writes `channel_mapping_config.json`.
5. Run **Current Folder** or **ALL Folders**.

**Discovery:** `collect_image_jobs()` walks subfolders recursively. **ALL Folders** loads each dataset’s own `channel_mapping_config.json` when present.

### Key UI settings

| Setting | Role |
|---------|------|
| Spot diameter (µm) | DoG blob size range (default ~5–12 µm) |
| Auto Threshold | Per-image DoG threshold; tag `_thrConservative` or `_thrHigh` in filenames |
| Background subtraction | Wide Gaussian haze removal on BTX |
| Functional NMJ Boundary (µm) | Distance cutoff for BTX class assignment (default 1.0 µm) |
| Save NMJ_Plot PNGs | Per-image 11-panel figures (memory-heavy) |

---

## Analysis pipeline

### 1. Spot detection (Difference of Gaussians)

BTX channel: optional white top-hat → muscle haze subtraction (σ = max(50 µm, 5 × max spot diameter)) → `skimage.blob_dog`.

Diameter in µm converts to DoG sigma: `sigma_um = diameter_um / (2 × sqrt(2))`.

**Auto threshold:** `median + 3 × (1.4826 × MAD)` on subsampled haze-subtracted BTX, clamped to `[0.02, 0.12]`. **Conservative** uses `sigma_ratio = 1.6`; **High** uses `1.3`.

![Background subtraction](readme/backgroundsubtraction.png)

### 2. Per-spot metrics

Segmentation threshold is tied to DoG detection (not Otsu on small crops). Batch mode keeps all diameter-filtered DoG hits — no muscle-vs-BTX intensity rejection.

| Column | Description |
|--------|-------------|
| `Dist_to_Muscle_um`, `Dist_to_Neuron_um` | Edge-corrected EDT minus spot radius (µm), clamped ≥ 0 |
| `Dist_to_Muscle_center_um`, `Dist_to_Neuron_center_um` | Center-only EDT (QC) |
| `INNERVATION_OVERLAP_PCT` | Spot mask overlap with neuron channel (%) |
| `MEAN_INTENSITY` | Mean haze-subtracted BTX inside spot mask |
| `ROUNDNESS` | `1 − eccentricity` from inertia tensor eigenvalues |
| `RADIUS` | Spot radius (µm) |
| `Resolution_Class` | `Low-Res` if pixel size > 0.5 µm/px |
| `is_NMJ` | Boolean: within NMJ boundary on both muscle and neuron axes |
| `BTX signal class` | See below |
| `TOTAL_IMAGE_AREA_um2` | Full mask area (master CSV / file stats) |

**Roundness QC:** `ROUNDNESS = NaN` when segmented mask has `< MIN_PIXELS_FOR_SHAPE` (20) pixels.

**Legacy CSVs:** On load, old class names (`NMJ`, `Aneural AChR clusters`, …) and `CIRCULARITY` are aliased to current names.

### 3. BTX signal classes (4-way)

Assigned from edge-corrected distances and the **Functional NMJ Boundary** slider:

| Class | Color | Rule |
|-------|-------|------|
| **early NMJ-like** | red | near muscle **and** neuron |
| **Muscle-associated** | green | near muscle only |
| **Neuron-associated** | blue | near neuron only |
| **Orphaned** | gray | near neither |

“Near” = distance ≤ boundary (default **1.0 µm**). **Orphaned** is a mixed distant-BTX bucket (dim noise, possible immature/mislocalized signal) — not a pure background control.

### 4. Global intensity Otsu

Computed across **all spots** in a batch run (`global_btx_intensity_otsu_threshold`). Used for:

- Vertical line on aggregate intensity histogram  
- Otsu-filtered KDE / abundance panels  
- `*_OTSU_DIM_NOISE_REJECTION*.csv` composition table  

Otsu is a **dim-spot filter**, not a synapse classifier. Spatial class + zone enrichment are the primary biological evidence.

---

## Outputs

Each run creates `output/<YYYYMMDD_HHMMSS>/`:

```
output/20260627_053347/
├── run_config.json
├── channel_mapping_config.json          # snapshot used for this run
├── ALL_FOLDERS_MASTER_RESULTS_thrConservative.csv
├── ALL_FOLDERS_STAT_SUMMARY_thrConservative.csv
├── ALL_FOLDERS_IMAGE_LEVEL_MEDIANS_thrConservative.csv
├── ALL_FOLDERS_OTSU_DIM_NOISE_REJECTION_thrConservative.csv
├── ALL_FOLDERS_FILE_STATS_thrConservative.csv
├── ALL_FOLDERS_SUMMARY_TABLE_thrConservative.csv
├── ALL_FOLDERS_SUMMARY_thrConservative.png
└── <dataset_folder>/
    ├── image_thrConservative_analysis.csv
    └── image_thrConservative_NMJ_Plot.png   # optional
```

**Filename tag:** `_thrConservative` or `_thrHigh` when Auto Threshold is on; omitted for manual threshold.

**Current Folder** runs use the `BATCH_*` prefix and write aggregate files at the run root; per-image files mirror under `output/<timestamp>/<dataset>/`.

| Artifact | Contents |
|----------|----------|
| `*_MASTER_RESULTS*.csv` | All spots; leading cols `SOURCE_FOLDER`, `SOURCE_IMAGE`, `TOTAL_IMAGE_AREA_um2` |
| `*_analysis.csv` | Per-image spot table |
| `*_FILE_STATS*.csv` | Per-image class counts, formation rate, zone areas/densities |
| `*_SUMMARY_TABLE*.csv` | Per-folder aggregates (ALL Folders only) |
| `*_STAT_SUMMARY*.csv` | All statistical tests with `level` column |
| `*_IMAGE_LEVEL_MEDIANS*.csv` | Per-image class medians (proximity, intensity, roundness) |
| `*_OTSU_DIM_NOISE_REJECTION*.csv` | % spots above global Otsu by class + interpretation |
| `*_SUMMARY*.png` | Aggregate dashboard figure |
| `run_config.json` | Run parameters, paths, image count |

Streamlit shows ZIP and per-file download buttons when a run completes.

---

## Figures

### Per-image `*_NMJ_Plot.png` (optional, 4×3 grid)

| Row | Col 0 | Col 1 | Col 2 |
|-----|-------|-------|-------|
| 0 | 1. Proximity scatter + marginal KDEs | 2. Size KDE | 3. Roundness KDE (3-class Kruskal title) |
| 1 | 4. early NMJ-like innervation hist | 5. Intensity KDE (spot-pooled MW title) | 6. Raw \| cleaned BTX |
| 2 | 7. Cleaned BTX | 8. BTX + spots | 9. Composite + all spots |
| 3 | 10. Composite + early NMJ-like only | 11. BTX density bar | — |

Per-image proximity/intensity titles use **spot-level** tests (exploratory). Use batch aggregate outputs for publication inference.

### Aggregate `*_SUMMARY*.png`

| Row | Left | Right |
|-----|------|-------|
| 0 | BTX intensity histogram + global Otsu (full width) | |
| 1 | Proximity scatter (image-level Kruskal title) | Size KDE |
| 2 | Roundness KDE | Innervation overlap |
| 3 | Intensity KDE (all spots) | Intensity KDE (Otsu-filtered) |
| 4 | Zone abundance + Friedman | Zone abundance (Otsu-filtered) |
| 5 | *(ALL Folders)* early NMJ-like rate control chart (full width) | |

Current-folder runs omit row 5.

**Display note:** Proximity scatter may apply tiny display-only jitter when many spots share clipped (0, 0) coordinates; CSV values and KDEs use true coordinates.

---

## Statistical analysis

Significance stars: `***` p < 0.001 · `**` p < 0.01 · `*` p < 0.05 · `ns` p ≥ 0.05.

### Inference unit (Methods)

> Because multiple BTX puncta within a single image are not independent, primary statistical comparisons were performed on per-image summary statistics (e.g. class-specific medians); spot-level tests are reported for visualization only.

### `level` column in `*_STAT_SUMMARY*.csv`

| Level | Role |
|-------|------|
| `primary_image_level` | Primary inference (image = unit of replication) |
| `primary_posthoc` | Pairwise follow-ups (Holm–Bonferroni) |
| `primary_sensitivity` | Unpaired alternatives (e.g. Mann–Whitney across different image sets) |
| `exploratory_spot_pooled` | All spots pooled — **plot support only** |

### Primary tests (batch)

| Question | Test |
|----------|------|
| Do classes differ in proximity to muscle/neuron? | Kruskal–Wallis on per-image class medians + Mann–Whitney posthoc (early NMJ-like vs others) |
| Does roundness differ across muscle/synapse classes? | Kruskal–Wallis on per-image medians (3 classes, shape-QC spots) |
| Is early NMJ-like brighter than Orphaned **in the same image**? | **Wilcoxon signed-rank (paired)** on class medians; Otsu-filtered variant included |
| Is the fraction above Otsu higher for early NMJ-like vs Orphaned? | Paired Wilcoxon on per-image fractions |
| Is BTX enriched in synaptic/muscle zones? | Friedman on per-image zone abundance + Conover–Iman posthoc |

Unpaired Mann–Whitney intensity rows are kept under `primary_sensitivity`.

### Identifying true AChR signal amid dirty BTX stain

No single statistic proves specificity. The pipeline uses **converging evidence**:

1. **Spatial classification** — early NMJ-like / Muscle-associated puncta lie at muscle and/or neuron masks; Orphaned lies far from both.  
2. **Zone abundance (Friedman)** — strongest primary test that BTX is not uniformly distributed; enriched at NMJ/muscle zones within each image.  
3. **Paired intensity (Wilcoxon)** — within fields that contain both early NMJ-like and Orphaned puncta, synaptic puncta are typically brighter than distant puncta.  
4. **Otsu dim-noise table** — descriptive spot composition; muscle/synapse classes are mostly above global Otsu; Orphaned is mostly below.

#### `*_OTSU_DIM_NOISE_REJECTION*.csv`

Reports `%` of spots in each class with `MEAN_INTENSITY ≥` global batch Otsu.

| Class | Typical pattern | Interpretation |
|-------|-----------------|----------------|
| **early NMJ-like** | ~80–90% above Otsu | Predominantly bright synaptic puncta — consistent with specific AChR staining. |
| **Muscle-associated** | ~75–85% | Muscle-proximal clusters pass the intensity gate amid dirty stain. |
| **Neuron-associated** | ~80–90% | Neuron-proximal puncta mostly bright; may include presynaptic or developing terminals. |
| **Orphaned** | ~25–40% | Mostly **dim** distant puncta (noise-like); minority above Otsu may be brighter immature/mislocalized BTX. |

**Caveats:** (1) Table is spot-pooled and descriptive. (2) Otsu does not use spatial context. (3) Orphaned ≠ pure background. (4) Paired Wilcoxon rows in `STAT_SUMMARY` test image-level fraction and intensity contrasts.

**Suggested paper sentence:** Global Otsu separates predominantly dim distant puncta from bright muscle/synapse-associated puncta; paired Wilcoxon supports higher early NMJ-like intensity within shared fields; Friedman confirms zone-specific enrichment — together supporting identification of AChR-related signal amid non-specific staining.

---

## Regenerating dashboards

Re-build aggregate PNG/PDFs from a saved master CSV **without** re-reading images:

```bash
python regenerate_all_folders_panel_pdfs.py \
  output/20260627_053347/ALL_FOLDERS_MASTER_RESULTS_thrConservative.csv
```

Friedman / zone-abundance panels need the companion `*_FILE_STATS*.csv` (auto-detected if beside the master file).

---

## Tests

```bash
docker compose --profile batch run --rm --no-deps multiple-image-nmj-analysis \
  python3 scripts/test_run_output.py
```

Checks timestamped `output/` routing, config snapshots, ZIP downloads, and a one-image smoke path.

---

## Single-image app notes

`BTX.py` mirrors detection and per-spot metrics for one file. It does **not** write `STAT_SUMMARY` or image-level primary tests (no `SOURCE_IMAGE` in the spot table). Use the batch pipeline for aggregate statistics and publication tables.
