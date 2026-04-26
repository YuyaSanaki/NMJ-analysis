# 🔬 Neuromuscular Junction (NMJ) Analysis Pipeline

A state-of-the-art, fully containerized image analysis toolkit designed to automatically detect, measure, and statistically categorize Neuromuscular Junctions (NMJs) directly from raw multi-channel confocal `.czi` files. Designed as a superior, fully automated replacement for FIJI/TrackMate workflows.

## 🏗 System Architecture

This project runs locally entirely inside Docker, utilizing `Streamlit` to generate powerful UI dashboards. It is structured into a dynamic **Dual-Service Architecture**:

- **🤖 Single-Image Pipeline (Port 8501):** Interactive dashboard useful for dialing in spot detection (DoG) parameters, morphological background subtraction, and Euclidean Distance thresholding on a single `.czi` file.
- **🚀 Bulk Batch Pipeline (Port 8502):** A high-throughput pipeline engineered to chew through entire folders of CZIs. It features dynamic JSON configuration saving and complex per-file channel configurations.

## 🚀 Quick Start

1. Start the Docker containers:
```bash
docker compose up --build -d
docker compose up -d
```
2. Navigate to your desired pipeline in the browser:
   * **Single Image Mode:** `http://localhost:8501`
   * **Batch Mode:** `http://localhost:8502`

3. **Data Mounting:** Place any `.czi` files into subfolders inside this project directory. The Docker container maps the local folder into `/app` so your data is immediately visible in the UI.

4. **Stop the containers** when you are done:
```bash
docker compose down
```

---

## 🎨 Features & Methodologies

### 1. Robust Spot Detection (Difference of Gaussians)
The system uses skimage's `blob_dog` algorithm to detect Acetylcholine Receptors (BTX channel) accurately. To prevent artifacts from background fluorescence, it utilizes an auto-optimized **Morphological White Top-Hat Filter** (the mathematical equivalent of FIJI's rolling-ball background subtraction) before spot detection.

### 2. Actual Physical Measurements
The pipeline fundamentally binds to reality by actively extracting the raw `Distance` scaling vectors (μm/pixel) from the intrinsic `.czi` metadata. All algorithms, boundaries, and outputs are dynamically mapped directly into **physical Micrometers (μm)**, completely isolating the math from arbitrary changes in microscope magnifications or pixel resolutions.

### 3. Biological & Spatial Exocentrism
For every detected receptor spot, the script executes highly localized Otsu thresholding across multiple spatial channels to extract deep contextual variables:

* **`Dist_to_Muscle_um` / `Dist_to_Neuron_um`**: A physical Euclidean Distance map is run against raw thresholded Muscle and Neuron channel masks. A threshold (e.g., `≤ 1.0 μm`) acts as the gating logic to classify a spot as a "Functional NMJ" vs an orphaned/background spot.
* **`INNERVATION_OVERLAP_PCT`**: Calculates what quantitative percentage of the receptor's physical 2D area (mask) is literally overlapped/covered by Neural signal, distinguishing healthy innervations from partially retracted or degenerating ones.
* **`MEAN_INTENSITY`**: Returns the average raw fluorescence directly inside the segmented receptor boundary.
* **`CIRCULARITY`**: Computes standard geometric circularity (`4*pi*Area / Perimeter^2`) to measure morphological structure (e.g., differentiating mature "pretzels" from dense oval plaques).

---

## 🔁 Using the Batch System

On `localhost:8502` you can process entire datasets seamlessly:
1. Select a dataset folder containing your CZIs.
2. Formulate a **Config Template** (Channel mapping) at the top of the screen.
3. Click **"📋 Paste Template to ALL Images"** to rapidly apply the mapping to all detected files. 
   *(Note: The UI inherently safeguards actual biological pixel sizes from being pasted over!)*
4. Expand individual files and use the **"🚫 Exclude"** toggles or manually adjust channel mappings on an individual basis if certain scans failed or differ.
5. Save your session using **"💾 Save Settings to Folder"** so you can pick up where you left off tomorrow.
6. Click **Run Batch Analysis**.

---

## 📊 Outputs

The batch system will deposit multiple artifacts directly into the source folder:

1. **`BATCH_MASTER_RESULTS.csv`**: A massive concatenated dataframe tracking every single segmented spot, tagged with the `SOURCE_IMAGE` so you can do grouping stats in R or GraphPad Prism.
2. **`[Filename]_analysis.csv`**: Granular backups of the data per-image.
3. **`[Filename]_NMJ_Plot.png`**: A beautiful aggregated massive 9-panel PNG generated for every image!
   * Row 1: Proximity Scatter Plot, Size Distribution KDE, Circularity Distribution KDE
   * Row 2: Innervation Overlap KDE, Receptor Intensity KDE, Pure Cleaned BTX Image
   * Row 3: Spots overlaid on BTX, Composite Images overlaid with ALL spots, and Composite Images overlaid exclusively with white indicator-arrows pointing out functional NMJs.
