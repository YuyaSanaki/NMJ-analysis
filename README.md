# Automated NMJ Proximity Analysis Pipeline

## Overview
This tool performs a Neuromuscular Junction (NMJ) proximity analysis directly from raw confocal image files (`.czi`). It eliminates the need for any external pre-processing with Fiji or TrackMate! 

The system reads your `.czi` files, dynamically detects receptor spots (similar to TrackMate's DoG technique), generates Distance Maps automatically via Otsu thresholding, and classifies functional NMJs with sub-pixel precision.

⚡ **Powered by a Streamlit Dashboard**, allowing you to seamlessly tune thresholds, pick channels, and view visual overlays in real-time.

## Organizing Your Data
Group your `.czi` data files into specific subfolders inside this main directory (e.g., `0714M-HF/`). The dashboard will detect them automatically.

## How to Run
Everything is containerized in Docker. No manual python setups necessary!

1. Ensure **Docker Desktop** is open and running on your Mac.
2. Open a terminal and navigate to this workspace folder.
3. Start the dashboard via Docker Compose:
```bash
docker compose up --build
```
4. Open your web browser and go to: **[http://localhost:8501](http://localhost:8501)**

## Using the Dashboard
1. Select your Dataset Folder and the `.czi` file you wish to analyze.
2. Use the **Channel Mapping** dropdowns to assign the correct channels (e.g., Channel 1 to Muscle, Channel 2 to Neuron, Channel 4 to BTX).
3. Tune the **Spot Detection (DoG)** parameters: Min/Max Spot Size and Detection Threshold (exactly as you would in TrackMate).
4. Click **Run Analysis**!

## Output Generation
When you click run, the script processes the images and saves the following directly to your dataset folder:
- **`[filename]_analysis.csv`**: A dataset containing every detected spot, its computed radius, X/Y coordinates, distance to Muscle, distance to Neuron, and the functional `is_NMJ` check flag.
- **`[filename]_NMJ_Plot.png`**: A dual-graph visualizing the raw BTX spots overlay next to the proximity calculation logic.
*(These visuals and counts will also be displayed live in your browser).*

*(To stop the dashboard later, simply press `Ctrl + C` in your terminal).*
