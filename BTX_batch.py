import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import gc
from datetime import datetime
from skimage.feature import blob_dog
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage import distance_transform_edt, zoom
from skimage.exposure import rescale_intensity

from nmj_master_dashboard import (
    BTX_CLASS_EARLY_NMJ,
    BTX_CLASS_MUSCLE,
    BTX_CLASS_NEURON,
    BTX_CLASS_ORPHANED,
    BTX_SIGNAL_CLASS_ORDER,
    BTX_SIGNAL_CLASS_PALETTE,
    BTX_SIGNAL_CLASS_LEGACY_ALIASES,
    DENSITY_COL_EARLY_NMJ,
    DENSITY_COL_MUSCLE,
    DENSITY_COL_NEURON,
    DENSITY_COL_ORPHANED,
    AREA_COL_EARLY_NMJ,
    AREA_COL_MUSCLE,
    AREA_COL_NEURON,
    AREA_COL_ORPHANED,
    UM2_PER_MM2,
    SPOT_DENSITY_PER_MM2_LABEL,
    MIN_PIXELS_FOR_SHAPE,
    RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL,
    ROUNDNESS_KRUSKAL_CLASSES,
    dataframe_for_roundness_kde_and_kruskal,
    normalize_btx_signal_classes,
    prepare_spot_table_for_master,
    finalize_master_results_dataframe,
    annotate_global_btx_intensity_otsu,
    nmj_vs_orphan_intensity_mannwhitney_title,
    roundness_3way_kruskal_title,
    proximity_joint_axes,
    draw_proximity_joint,
    save_all_folders_summary_png,
    _export_figure_panels_to_pdfs,
    build_aggregate_batch_dashboard_figure,
    build_batch_stat_summary_dataframe,
    present_otsu_spot_count_matrix,
    present_otsu_spot_change_comparisons,
    SUPPORTED_EXTENSIONS,
    collect_image_jobs,
    get_confocal_metadata,
    load_confocal_image,
    total_image_area_um2_from_metadata,
)

from nmj_run_output import (
    apply_channel_mapping_to_session_keys,
    channel_config_state_key,
    create_run_output_dir,
    default_channel_config_for_file,
    export_channel_mapping_from_session,
    mirror_dataset_output_path,
    read_channel_mapping_config,
    render_streamlit_download_section,
    resolve_channel_mapping_for_folder,
    save_run_config_files,
    snapshot_channel_mappings,
)

# Session keys for per-folder channel mapping (survive dataset folder changes).
_ACTIVE_CHANNEL_FOLDER_KEY = "_channel_config_active_folder"
_CHANNEL_MAPPING_CACHE_KEY = "_channel_mapping_cache"
_CHANNEL_FOLDER_FILES_KEY = "_channel_folder_czi_files"


def _snapshot_channel_mapping_for_folder(folder_rel, czi_files):
    if not czi_files:
        return
    exported = export_channel_mapping_from_session(folder_rel, czi_files, st.session_state)
    if exported:
        st.session_state.setdefault(_CHANNEL_MAPPING_CACHE_KEY, {})[folder_rel] = exported


def _apply_channel_mapping_for_folder(folder_rel, folder_path, czi_files):
    cache = st.session_state.get(_CHANNEL_MAPPING_CACHE_KEY, {})
    mapping = resolve_channel_mapping_for_folder(
        folder_rel,
        disk_mapping=read_channel_mapping_config(folder_path),
        cache_mapping=cache.get(folder_rel),
    )
    if not mapping:
        return False
    for key, value in apply_channel_mapping_to_session_keys(folder_rel, mapping, czi_files).items():
        st.session_state[key] = value
    st.session_state.setdefault(_CHANNEL_MAPPING_CACHE_KEY, {})[folder_rel] = mapping
    return True


def sync_selected_folder_channel_mapping(folder_rel, folder_path, czi_files):
    """Snapshot the previous folder and restore mapping for the newly selected folder."""
    prev = st.session_state.get(_ACTIVE_CHANNEL_FOLDER_KEY)
    folder_files = st.session_state.setdefault(_CHANNEL_FOLDER_FILES_KEY, {})
    if prev != folder_rel:
        if prev is not None and prev in folder_files:
            _snapshot_channel_mapping_for_folder(prev, folder_files[prev])
        st.session_state[_ACTIVE_CHANNEL_FOLDER_KEY] = folder_rel
        _apply_channel_mapping_for_folder(folder_rel, folder_path, czi_files)
    folder_files[folder_rel] = list(czi_files)

# Each subdirectory under this path is one dataset folder (contains `.czi` files).
DATA_ROOT = "data"
base_dir = DATA_ROOT  # legacy alias used in older batch paths

# DoG ``blob_dog`` sigma_ratio: **Auto threshold sensitivity** radio (auto) or manual number input (default = Conservative).
DOG_SIGMA_RATIO_CONSERVATIVE = 1.6
DOG_SIGMA_RATIO_HIGH = 1.3
# Auto DoG threshold uses median + k × (1.4826 × MAD); k is fixed (sensitivity only changes sigma_ratio).
AUTO_DOG_THRESHOLD_MAD_K = 3.0


def dog_sigma_ratio_from_sensitivity(sensitivity: str) -> float:
    return DOG_SIGMA_RATIO_HIGH if sensitivity == "High" else DOG_SIGMA_RATIO_CONSERVATIVE


def format_auto_thr_sensitivity_label(mode: str) -> str:
    if mode == "High":
        return f"High ({DOG_SIGMA_RATIO_HIGH:g})"
    return f"Conservative ({DOG_SIGMA_RATIO_CONSERVATIVE:g})"


def same_dir(a, b):
    """True if paths refer to the same directory (avoids join/rel vs abs mismatches on Docker/macOS)."""
    return os.path.normpath(os.path.abspath(a)) == os.path.normpath(os.path.abspath(b))


collect_czi_jobs = collect_image_jobs


st.set_page_config(page_title="NMJ Pipeline", layout="wide")

st.title("🔬 Multiple-Image Batch NMJ Pipeline")
st.markdown("Select a folder to batch-process supported confocal images automatically (.czi, .nd2, .lif, .oir, .poir, .tif).")

# --- 1. Folder & File Selection ---
os.makedirs(DATA_ROOT, exist_ok=True)

# Discover all image files recursively under DATA_ROOT
all_jobs_global = collect_image_jobs([DATA_ROOT])
if not all_jobs_global:
    st.warning(f"No supported confocal or TIFF image files found inside `{DATA_ROOT}/` or its subfolders.")
    st.stop()

# Get unique directories containing supported files (relative to DATA_ROOT)
folders_set = set(os.path.relpath(d, DATA_ROOT) for d, _ in all_jobs_global)
folders = sorted(list(folders_set))

selected_folder = st.selectbox("📂 Select Dataset Folder", folders)
folder_path = os.path.join(DATA_ROOT, selected_folder)

# Supported files in the selected folder (non-recursively for the UI configuration)
czi_files = [f for d, f in all_jobs_global if same_dir(d, folder_path)]

sync_selected_folder_channel_mapping(selected_folder, folder_path, czi_files)

# --- 2. Extract Metadata & Config for Batch ---
@st.cache_data(show_spinner=False)
def fast_czi_meta(path):
    num_channels, pixel_size_um, _ = get_confocal_metadata(path)
    return num_channels, pixel_size_um

st.subheader("⚙️ Batch Channel Mapping")

# --- Import / Export Logic ---
c_exp, c_imp, _ = st.columns([1.5, 1.5, 3])
config_json_path = os.path.join(folder_path, "channel_mapping_config.json")

if c_exp.button("💾 Save Settings to Folder"):
    import json
    export_data = export_channel_mapping_from_session(
        selected_folder, czi_files, st.session_state
    )
    try:
        with open(config_json_path, "w") as jf:
            json.dump(export_data, jf, indent=4)
        st.session_state.setdefault(_CHANNEL_MAPPING_CACHE_KEY, {})[selected_folder] = export_data
        st.success("Config saved successfully.")
    except Exception as e:
        st.error(f"Failed to save: {e}")

if c_imp.button("📂 Load Settings from Folder"):
    mapping = read_channel_mapping_config(folder_path)
    if mapping:
        for key, value in apply_channel_mapping_to_session_keys(
            selected_folder, mapping, czi_files
        ).items():
            st.session_state[key] = value
        st.session_state.setdefault(_CHANNEL_MAPPING_CACHE_KEY, {})[selected_folder] = mapping
        st.rerun()
    elif os.path.exists(config_json_path):
        st.error("Failed to load channel mapping config.")
    else:
        st.warning("No `channel_mapping_config.json` found in this folder!")

file_configs = {}

if len(czi_files) > 0:
    first_czi = czi_files[0]
    path_tmp = os.path.join(folder_path, first_czi)
    n_ch_global, px_size_global = fast_czi_meta(path_tmp)
    options_global = [f"Channel {i+1}" for i in range(n_ch_global)]
    
    st.markdown("### 📋 Config Template")
    st.markdown("Define a configuration here, then use the Paste buttons below to copy it into individual images.")
    
    cg1, cg2, cg3 = st.columns(3)
    g_m = cg1.selectbox("Template Muscle", range(n_ch_global), format_func=lambda x: options_global[x], index=0, key="g_m")
    g_n = cg2.selectbox("Template Neuron", range(n_ch_global), format_func=lambda x: options_global[x], index=min(1, n_ch_global-1), key="g_n")
    g_b = cg3.selectbox("Template BTX", range(n_ch_global), format_func=lambda x: options_global[x], index=min(3, n_ch_global-1), key="g_b")

    # Master Paste Button
    if st.button("🔽 Paste Template to ALL Images", type="secondary"):
        for cf in czi_files:
            n_ch_tmp, _ = fast_czi_meta(os.path.join(folder_path, cf))
            st.session_state[channel_config_state_key(selected_folder, cf, "m")] = min(g_m, n_ch_tmp - 1)
            st.session_state[channel_config_state_key(selected_folder, cf, "n")] = min(g_n, n_ch_tmp - 1)
            st.session_state[channel_config_state_key(selected_folder, cf, "b")] = min(g_b, n_ch_tmp - 1)
            # Intentionally NOT overriding pixel size to preserve true biological scaling!

    st.divider()
    st.markdown("### 📂 Individual File Settings")
    for czi_file in czi_files:
        path_tmp = os.path.join(folder_path, czi_file)
        n_ch_ind, px_size_ind = fast_czi_meta(path_tmp)
        options_ind = [f"Channel {i+1}" for i in range(n_ch_ind)]
        
        # Initialize session state so the selectboxes don't error out when natively bound
        defaults = default_channel_config_for_file(n_ch=n_ch_ind, pixel_size_um=px_size_ind)
        for field, default_val in (
            ("m", defaults["m"]),
            ("n", defaults["n"]),
            ("b", defaults["b"]),
            ("p", defaults["p"]),
            ("skip", defaults["skip"]),
        ):
            cfg_key = channel_config_state_key(selected_folder, czi_file, field)
            if cfg_key not in st.session_state:
                st.session_state[cfg_key] = default_val

        skip_key = channel_config_state_key(selected_folder, czi_file, "skip")
        m_key = channel_config_state_key(selected_folder, czi_file, "m")
        n_key = channel_config_state_key(selected_folder, czi_file, "n")
        b_key = channel_config_state_key(selected_folder, czi_file, "b")
        p_key = channel_config_state_key(selected_folder, czi_file, "p")

        with st.expander(f"▶️ Config: {czi_file}", expanded=False):

            c_skip, c_paste, _ = st.columns([1.5, 1.5, 3])
            skip_file = c_skip.checkbox("🚫 Exclude image from batch", key=skip_key)

            # The Paste Button directly artificially modifies the session state properties of the inputs below!
            if c_paste.button("📋 Paste Template Here", key=f"btn_{selected_folder}_{czi_file}"):
                st.session_state[m_key] = min(g_m, n_ch_ind - 1)
                st.session_state[n_key] = min(g_n, n_ch_ind - 1)
                st.session_state[b_key] = min(g_b, n_ch_ind - 1)
                # Intentionally NOT overriding pixel size here!
                st.rerun() # Force immediate UI refresh to show the newly pasted values

            if skip_file:
                st.warning("Image will be bypassed during batch processing.")

            c1, c2, c3, c4 = st.columns(4)
            m_id = c1.selectbox("Muscle", range(n_ch_ind), format_func=lambda x: options_ind[x], key=m_key, disabled=skip_file)
            n_id = c2.selectbox("Neuron", range(n_ch_ind), format_func=lambda x: options_ind[x], key=n_key, disabled=skip_file)
            b_id = c3.selectbox("BTX", range(n_ch_ind), format_func=lambda x: options_ind[x], key=b_key, disabled=skip_file)
            ps = c4.number_input("Pixel Size", format="%0.7f", key=p_key, help="Unique biological scale.", disabled=skip_file)
            
            file_configs[czi_file] = {"muscle": m_id, "neuron": n_id, "btx": b_id, "pixel_size": ps, "skip": skip_file}

st.divider()


# --- 2. Shared Multi-Format Loader Mapping ---
load_czi_image = load_confocal_image


def compute_sigma_bounds_px(min_sigma_um, max_sigma_um, pixel_size_um, image_shape):
    """Convert physical sigma bounds to stable pixel bounds for blob_dog."""
    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    min_px_raw = float(min_sigma_um) / pixel_size_safe
    max_px_raw = float(max_sigma_um) / pixel_size_safe
    min_px = max(0.5, min_px_raw)
    max_px = max(min_px + 0.1, max_px_raw)

    # Prevent pathological DoG scales that can OOM or disconnect Streamlit.
    h, w = image_shape[:2]
    # Keep upper sigma conservative; huge sigmas are computationally unstable in Streamlit.
    sigma_cap = max(2.0, min(64.0, min(h, w) / 12.0))
    if min_px > sigma_cap:
        return None, None, sigma_cap
    max_px = min(max_px, sigma_cap)
    return min_px, max_px, sigma_cap


def compute_bg_radius_px(bg_radius_um, pixel_size_um, image_shape):
    """Convert physical background radius to a bounded morphological kernel size."""
    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    radius_px_raw = float(bg_radius_um) / pixel_size_safe
    h, w = image_shape[:2]
    radius_cap = max(3.0, min(96.0, min(h, w) / 16.0))
    radius_px = int(max(1, min(round(radius_px_raw), radius_cap)))
    clipped = radius_px_raw > radius_cap
    return radius_px, clipped, radius_cap


def remove_muscle_haze(img, pixel_size_um, bg_sigma_um):
    """
    Subtract broad BTX background so puncta remain while wide plaques are not hollowed out.

    ``bg_sigma_um`` is the Gaussian standard deviation in micrometers for the low-frequency
    haze estimate. It comes from the Streamlit control **Manual Background Radius (μm)** or,
    when **Auto-Optimize Background Radius** is on, from the max spot diameter (µm).
    """
    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    sigma_um = max(1e-9, float(bg_sigma_um))
    bg_sigma_px = float(sigma_um) / pixel_size_safe
    background = gaussian(img, sigma=bg_sigma_px, preserve_range=True)
    result = img.astype(np.float32, copy=False) - background.astype(np.float32, copy=False)
    return np.clip(result, 0.0, None).astype(img.dtype, copy=False)


def threshold_raw_for_spot_crop(threshold_used, p_high, window_btx):
    """Map normalized DoG threshold to haze-subtracted raw units; fall back if degenerate."""
    th_raw = float(threshold_used) * float(p_high)
    if window_btx.size == 0:
        return th_raw
    wmax = float(np.max(window_btx))
    if th_raw <= 0 or (wmax > 0 and th_raw >= wmax):
        try:
            th_raw = float(threshold_otsu(window_btx))
        except ValueError:
            th_raw = wmax * 0.5 if wmax > 0 else 1e-9
    return th_raw


def detect_blobs_stable(
    img_btx_norm, min_diameter_um, max_diameter_um, pixel_size_um, threshold, sigma_ratio=None
):
    """Run DoG in a memory-safe way using diameter thresholds (µm)."""
    if sigma_ratio is None:
        sigma_ratio = DOG_SIGMA_RATIO_CONSERVATIVE
    # blob_dog scale is sigma; approximate blob radius is sigma*sqrt(2).
    # Therefore: sigma_um = (diameter_um / 2) / sqrt(2).
    min_sigma_um = float(min_diameter_um) / (2.0 * np.sqrt(2.0))
    max_sigma_um = float(max_diameter_um) / (2.0 * np.sqrt(2.0))

    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    max_sigma_px_raw = float(max_sigma_um) / pixel_size_safe
    dog_scale = 1.0
    dog_sigma_target = 48.0

    if max_sigma_px_raw > dog_sigma_target:
        dog_scale = max(0.1, dog_sigma_target / max_sigma_px_raw)
        img_for_dog = zoom(img_btx_norm, zoom=dog_scale, order=1)
    else:
        img_for_dog = img_btx_norm

    pixel_size_for_dog = pixel_size_safe / dog_scale
    min_sigma_px, max_sigma_px, sigma_cap = compute_sigma_bounds_px(
        min_sigma_um=min_sigma_um,
        max_sigma_um=max_sigma_um,
        pixel_size_um=pixel_size_for_dog,
        image_shape=img_for_dog.shape,
    )
    if min_sigma_px is None:
        return None, dog_scale, sigma_cap

    threshold_for_dog = float(threshold)
    if dog_scale < 1.0:
        # Downsampling smooths local peaks, so keep DoG sensitivity comparable by
        # reducing threshold slightly in scaled space (with a conservative floor).
        threshold_for_dog *= max(0.6, dog_scale)

    blobs = blob_dog(
        img_for_dog,
        min_sigma=min_sigma_px,
        max_sigma=max_sigma_px,
        sigma_ratio=float(sigma_ratio),
        threshold=threshold_for_dog,
    )
    if len(blobs) == 0:
        return blobs, dog_scale, sigma_cap

    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)  # radius in dog-space pixels
    if dog_scale != 1.0:
        blobs[:, :2] = blobs[:, :2] / dog_scale
        blobs[:, 2] = blobs[:, 2] / dog_scale
    # DoG `min_sigma` / `max_sigma` only approximate the UI "diameter" range; the 0.5 px
    # lower bound in compute_sigma_bounds_px and skimage's discrete scale steps can still
    # yield small effective radii. Enforce true physical min/max *diameter* (µm) here;
    # RADIUS in outputs is (column 2) * pixel_size, so diameter_um = 2 * r_px * um/pix.
    r_um = blobs[:, 2] * pixel_size_safe
    d_um = 2.0 * r_um
    ok = (d_um >= float(min_diameter_um)) & (d_um <= float(max_diameter_um))
    blobs = blobs[ok]
    return blobs, dog_scale, sigma_cap


def estimate_auto_threshold(img_btx_norm):
    """
    Compute an auto DoG threshold from a normalised BTX image.

    Uses ``median + AUTO_DOG_THRESHOLD_MAD_K × (1.4826 × MAD)`` on positives (> 0.005) in a
    subsampled image, clipped to ``[0.02, 0.12]``. DoG **sigma_ratio** (Conservative vs High) is
    separate — see :func:`dog_sigma_ratio_from_sensitivity`.
    """
    sample = np.asarray(img_btx_norm, dtype=np.float32)[::4, ::4].ravel()
    if sample.size == 0:
        return 0.05

    pos = sample[sample > 0.005]
    if pos.size < 50:
        return 0.05

    median = float(np.median(pos))
    mad = float(np.median(np.abs(pos - median)))
    std_est = 1.4826 * mad

    k = float(AUTO_DOG_THRESHOLD_MAD_K)
    return float(np.clip(median + k * std_est, 0.02, 0.12))


# --- 3. Detection Parameters ---
st.subheader("🎯 Spot Detection (DoG) & Analysis Parameters")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown("**DoG Tunning (BTX)**")
    min_diameter_um = st.number_input("Min Spot Diameter (μm)", value=5.00, step=0.10)
    max_diameter_um = st.number_input("Max Spot Diameter (μm)", value=12.00, step=0.10)
    if max_diameter_um <= min_diameter_um:
        st.error("Max Spot Diameter must be larger than Min Spot Diameter.")
        st.stop()
    auto_threshold = st.checkbox(
        "Auto Threshold per image",
        value=True,
        help="Adapts DoG threshold from each image's BTX signal/noise profile.",
    )
    auto_thr_sensitivity = st.radio(
        "Auto threshold sensitivity (DoG σ ratio)",
        options=["Conservative", "High"],
        index=0,
        format_func=format_auto_thr_sensitivity_label,
        horizontal=True,
        disabled=not auto_threshold,
        help="skimage blob_dog sigma_ratio: Conservative (1.6) vs High (1.3).",
    )
    threshold = st.number_input("Detection Threshold", value=0.12, step=0.01, disabled=auto_threshold)
    dog_sigma_ratio_manual = st.number_input(
        "DoG sigma ratio (manual)",
        value=float(DOG_SIGMA_RATIO_CONSERVATIVE),
        min_value=1.01,
        max_value=2.5,
        step=0.05,
        format="%.2f",
        disabled=auto_threshold,
        help="Used with manual Detection Threshold; default matches Auto Conservative (1.6).",
    )
    
    auto_bg = st.checkbox("Auto-Optimize Background Radius", value=True, help="Uses a physical-radius model (µm) and converts to pixels per image.")
    if not auto_bg:
        btx_bg_radius_um = st.number_input("Manual Background Radius (μm)", value=1.0, step=0.1)
    else:
        # Tie auto background radius to detected maximum diameter scale.
        btx_bg_radius_um = float(max_diameter_um)

with col_p2:
    st.markdown("**EDT Thresholds**")
    st.markdown("*Otsu's method is used automatically. Multiplier adjusts sensitivity.*")
    m_thresh_mult = st.slider("Muscle Threshold Multiplier", 0.5, 3.0, 1.0, step=0.1)
    n_thresh_mult = st.slider("Neuron Threshold Multiplier", 0.5, 3.0, 1.0, step=0.1)

with col_p3:
    st.markdown("**NMJ Logic**")
    distance_threshold_um = st.number_input("Functional NMJ Boundary (μm)", value=1.0, step=0.1)

# --- 4. Run Batch Pipeline ---
col_run1, col_run2 = st.columns(2)
run_current = col_run1.button("🚀 Run Batch Analysis (Current Folder)", type="primary")
run_all = col_run2.button("🚀 Run Batch Analysis (ALL Folders)", type="primary", help="Executes natively on every tracked folder using your active Global Template Mapping")
save_pngs = st.checkbox(
    "Save per-image NMJ_Plot PNGs during batch",
    value=True,
    help="When off, no *_NMJ_Plot.png (including Raw BTX | Cleaned BTX) is written under the run output folder—only CSVs and master summaries. "
    "Turn off to reduce memory and speed up ALL-folder runs.",
)

if run_current or run_all:
    all_file_stats = []
    master_rows_written = 0

    dog_sigma_ratio = (
        dog_sigma_ratio_from_sensitivity(auto_thr_sensitivity)
        if auto_threshold
        else float(dog_sigma_ratio_manual)
    )
    if not auto_threshold:
        st.caption(f"Manual DoG: detection threshold `{float(threshold):.4f}`, sigma_ratio `{dog_sigma_ratio}`")

    progress = st.progress(0)
    status = st.empty()
    
    target_dirs = [folder_path] if run_current else [os.path.join(DATA_ROOT, d) for d in folders]

    # Recursive discovery: .czi in subfolders (e.g. Data/Cond1/slide/1.czi) are included.
    # Per-image CSV/PNG are written under output/<run_timestamp>/<dataset_folder>/.
    all_target_czis = collect_czi_jobs(target_dirs)

    # Embed the sensitivity mode in output filenames so Conservative and High runs
    # save to separate files and neither overwrites the other.
    _thr_tag = f"_thr{auto_thr_sensitivity}" if auto_threshold else ""

    data_root_abs = os.path.abspath(DATA_ROOT)
    unique_target_dirs = sorted({os.path.abspath(d) for d, _ in all_target_czis})

    run_dir = create_run_output_dir()
    channel_snapshot = snapshot_channel_mappings(
        data_root_abs,
        unique_target_dirs,
        active_folder_path=os.path.abspath(folder_path) if run_current else None,
        active_file_configs=file_configs if run_current else None,
    )
    run_config = {
        "run_id": os.path.basename(run_dir),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
        "mode": "all_folders" if run_all else "current_folder",
        "selected_folder": selected_folder,
        "threshold_tag": _thr_tag,
        "save_pngs": bool(save_pngs),
        "parameters": {
            "min_diameter_um": float(min_diameter_um),
            "max_diameter_um": float(max_diameter_um),
            "auto_threshold": bool(auto_threshold),
            "auto_thr_sensitivity": auto_thr_sensitivity if auto_threshold else None,
            "threshold": float(threshold) if not auto_threshold else None,
            "dog_sigma_ratio_manual": float(dog_sigma_ratio_manual) if not auto_threshold else None,
            "dog_sigma_ratio": float(dog_sigma_ratio),
            "auto_bg": bool(auto_bg),
            "btx_bg_radius_um": float(btx_bg_radius_um),
            "m_thresh_mult": float(m_thresh_mult),
            "n_thresh_mult": float(n_thresh_mult),
            "distance_threshold_um": float(distance_threshold_um),
        },
    }
    save_run_config_files(run_dir, run_config, channel_snapshot)
    st.info(f"Writing outputs to `{run_dir}` (channel mapping JSON stays in `data/`).")

    if run_all:
        master_csv = os.path.join(run_dir, f"ALL_FOLDERS_MASTER_RESULTS{_thr_tag}.csv")
        master_png = os.path.join(run_dir, f"ALL_FOLDERS_SUMMARY{_thr_tag}.png")
        summary_table_csv = os.path.join(run_dir, f"ALL_FOLDERS_SUMMARY_TABLE{_thr_tag}.csv")
    else:
        master_csv = os.path.join(run_dir, f"BATCH_MASTER_RESULTS{_thr_tag}.csv")
        master_png = os.path.join(run_dir, f"BATCH_SUMMARY{_thr_tag}.png")
        summary_table_csv = None

    for i, (current_d, czi_file) in enumerate(all_target_czis):
        czi_path = os.path.join(current_d, czi_file)
        
        # Determine config logic based on whether we exist natively in the actively mapped UI window
        if same_dir(current_d, folder_path) and czi_file in file_configs:
            conf = file_configs[czi_file]
        else:
            # For folders outside the active UI, try loading saved per-folder configs first
            folder_configs = read_channel_mapping_config(current_d)
            loaded_conf = None
            if folder_configs and czi_file in folder_configs:
                fc = folder_configs[czi_file]
                loaded_conf = {
                    "muscle": fc["m"],
                    "neuron": fc["n"],
                    "btx": fc["b"],
                    "pixel_size": fc.get("p", 1.0),
                    "skip": fc.get("skip", False),
                }

            if loaded_conf is not None:
                conf = loaded_conf
            else:
                # Fall back to global template mapping
                n_ch_temp, px_size_temp = fast_czi_meta(czi_path)
                conf = {
                    "muscle": min(st.session_state.get("g_m", 0), n_ch_temp-1),
                    "neuron": min(st.session_state.get("g_n", 1), n_ch_temp-1),
                    "btx": min(st.session_state.get("g_b", 3), n_ch_temp-1),
                    "pixel_size": float(px_size_temp),
                    "skip": False
                }
        
        if conf.get("skip", False):
            status.write(f"⏭️ **Skipping:** `{current_d}/{czi_file}` (Marked as Excluded)")
            progress.progress((i + 1) / len(all_target_czis))
            continue
            
        pixel_size = conf['pixel_size']
        
        status.write(f"🔄 **Processing ({i+1}/{len(all_target_czis)}):** `{current_d}/{czi_file}` ...")
        
        try:
            fig = None  # Per-iteration figure; closed in ``finally`` (never ``plt.close('all')``).
            total_image_area_um2 = total_image_area_um2_from_metadata(czi_path)
            # Extract channels using selective channel reads. Only the muscle/neuron/BTX
            # channels are pulled off disk + Z-projected, instead of loading the full
            # multi-channel volume and keeping it pinned alive via numpy views.
            channels = load_czi_image(
                czi_path,
                channel_indices=[conf['muscle'], conf['neuron'], conf['btx']],
            )
            if isinstance(channels, dict):
                img_muscle = channels[conf['muscle']]
                img_neuron = channels[conf['neuron']]
                img_btx = channels[conf['btx']]
            else:
                # Defensive fallback: legacy 3D ndarray path.
                img_muscle = channels[conf['muscle']].copy()
                img_neuron = channels[conf['neuron']].copy()
                img_btx = channels[conf['btx']].copy()
            channels = None
            image_data = None
            gc.collect()
            # Only retain the pre-tophat BTX copy when we are going to render the PNG;
            # for very large multi-folder runs this single skip can save hundreds of MB
            # per iteration on top of the locals() bug fix.
            img_btx_raw = img_btx.copy() if save_pngs else None

            # --- Background subtraction + BTX normalization (single shared p_high for DoG and spot crops) ---
            img_btx = remove_muscle_haze(img_btx, pixel_size, btx_bg_radius_um)
            p_high = float(np.percentile(img_btx, 99.9))
            if p_high <= 0:
                p_high = 1e-5
            img_btx_norm = np.clip(img_btx.astype(np.float32, copy=False) / p_high, 0.0, 1.0)
            threshold_used = estimate_auto_threshold(img_btx_norm) if auto_threshold else float(threshold)
            if auto_threshold:
                st.caption(
                    f"{czi_file}: Detection threshold `{threshold_used:.4f}` — DoG sigma_ratio `{dog_sigma_ratio}`"
                )
            blobs, dog_scale, sigma_cap = detect_blobs_stable(
                img_btx_norm=img_btx_norm,
                min_diameter_um=min_diameter_um,
                max_diameter_um=max_diameter_um,
                pixel_size_um=pixel_size,
                threshold=threshold_used,
                sigma_ratio=dog_sigma_ratio,
            )
            if auto_threshold and blobs is not None and len(blobs) == 0:
                # One rescue pass for strict auto thresholds on low-contrast images.
                threshold_retry = max(0.02, float(threshold_used) * 0.8)
                blobs_retry, dog_scale_retry, sigma_cap_retry = detect_blobs_stable(
                    img_btx_norm=img_btx_norm,
                    min_diameter_um=min_diameter_um,
                    max_diameter_um=max_diameter_um,
                    pixel_size_um=pixel_size,
                    threshold=threshold_retry,
                    sigma_ratio=dog_sigma_ratio,
                )
                if blobs_retry is not None and len(blobs_retry) > 0:
                    blobs = blobs_retry
                    dog_scale = dog_scale_retry
                    sigma_cap = sigma_cap_retry
                    threshold_used = float(threshold_retry)
                    st.caption(f"{czi_file}: Auto-threshold rescue retry used `{threshold_used:.4f}`")
            if blobs is None:
                st.warning(
                    f"Skipped {czi_file}: Min Spot Diameter too large for image scale. "
                    f"Use below approximately {2.0 * np.sqrt(2.0) * sigma_cap * float(pixel_size):.3g} μm."
                )
                progress.progress((i + 1) / len(all_target_czis))
                continue
            if dog_scale < 1.0:
                st.warning(
                    f"{czi_file}: Large spot-size mode enabled (DoG scale {dog_scale:.2f}x) for stability."
                )
            
            # --- DoG Spot Detection ---
            if len(blobs) == 0:
                continue # Skip file if absolutely no spots found

            # Compute Otsu directly on the raw intensity images
            m_thresh = threshold_otsu(img_muscle) * m_thresh_mult
            n_thresh = threshold_otsu(img_neuron) * n_thresh_mult
            
            muscle_mask = img_muscle > m_thresh
            neuron_mask = img_neuron > n_thresh
            
            # Distance Transform (outputs in raw pixels). scipy returns float64 by
            # default; downcasting to float32 halves the resident memory of the four
            # large EDT arrays per image, which is one of the dominant allocations in
            # the per-iteration footprint.
            edt_muscle_px = distance_transform_edt(muscle_mask == 0).astype(np.float32, copy=False)
            edt_neuron_px = distance_transform_edt(neuron_mask == 0).astype(np.float32, copy=False)

            # Convert direct arrays into physical Micrometers using CZI metadata
            edt_muscle_um = (edt_muscle_px * np.float32(pixel_size))
            edt_neuron_um = (edt_neuron_px * np.float32(pixel_size))
            # The pixel-domain EDTs are no longer needed; drop them now to keep the
            # working set small for the per-spot loop below.
            edt_muscle_px = None
            edt_neuron_px = None

            from skimage.measure import regionprops, label
            
            # --- Extract Spot Distances & Morphological Shape ---
            spots_data = []
            distances_m = []
            distances_n = []
            
            for index, blob in enumerate(blobs):
                y, x, r = blob
                y_idx, x_idx = int(round(y)), int(round(x))
                
                # Boundary check
                y_idx = np.clip(y_idx, 0, edt_muscle_um.shape[0] - 1)
                x_idx = np.clip(x_idx, 0, edt_muscle_um.shape[1] - 1)

                d_m_center = float(edt_muscle_um[y_idx, x_idx])
                d_n_center = float(edt_neuron_um[y_idx, x_idx])
                r_um = float(r * pixel_size)
                d_m_um = max(0.0, d_m_center - r_um)
                d_n_um = max(0.0, d_n_center - r_um)

                distances_m.append(d_m_um)
                distances_n.append(d_n_um)
                
                # --- Morphological & Biological Metrics ---
                roundness = np.nan
                area_px_spot = np.nan
                mean_intensity = 0.0
                overlap_ratio = 0.0
                
                r_int = max(3, int(r * 2)) # crop window buffer
                box_y1 = max(0, y_idx - r_int)
                box_y2 = min(img_btx.shape[0], y_idx + r_int)
                box_x1 = max(0, x_idx - r_int)
                box_x2 = min(img_btx.shape[1], x_idx + r_int)
                
                window_btx = img_btx[box_y1:box_y2, box_x1:box_x2]
                window_neuron = neuron_mask[box_y1:box_y2, box_x1:box_x2]
                
                if window_btx.size >= 4:
                    try:
                        th = threshold_raw_for_spot_crop(threshold_used, p_high, window_btx)
                        labeled = label(window_btx > th)
                        center_y, center_x = y_idx - box_y1, x_idx - box_x1
                        spot_label = labeled[center_y, center_x]
                        
                        # If geometric center fell on a black noise pixel, snap to max bright pixel in crop
                        if spot_label == 0:
                            spot_label = labeled[np.unravel_index(np.argmax(window_btx), window_btx.shape)]
                            
                        if spot_label > 0:
                            props_list = regionprops(labeled)
                            props_dict = {p.label: p for p in props_list}
                            if spot_label in props_dict:
                                prop = props_dict[spot_label]
                            elif props_list:
                                prop = max(props_list, key=lambda x: x.area)
                                spot_label = prop.label
                            else:
                                prop = None
                            if prop is not None:
                                spot_mask = (labeled == spot_label)
                                area_px_spot = float(prop.area)
                                # Roundness from eccentricity (moment-based; stable vs pixelated perimeters).
                                if prop.area > MIN_PIXELS_FOR_SHAPE:
                                    roundness = float(
                                        np.clip(1.0 - float(prop.eccentricity), 0.0, 1.0)
                                    )

                                # 2. Mean Fluorescence Intensity
                                mean_intensity = float(np.mean(window_btx[spot_mask]))

                                # 3. Innervation/Colocalization (Overlap %)
                                overlap_pixels = np.sum(spot_mask & window_neuron)
                                if prop.area > 0:
                                    overlap_ratio = float(overlap_pixels / prop.area) * 100.0

                    except Exception:
                        pass # if crop is fully uniform or algo fails

                spots_data.append({
                    "SPOT_ID": index,
                    "POSITION_X": x * pixel_size, # physical um
                    "POSITION_Y": y * pixel_size, # physical um
                    "RADIUS": r * pixel_size, # Spot radius in physical um
                    # Legacy column name: now moment-based roundness (1 − eccentricity), not 4πA/P².
                    "CIRCULARITY": roundness,
                    "ROUNDNESS": roundness,
                    "AREA_PX": area_px_spot,
                    "MEAN_INTENSITY": mean_intensity,
                    "INNERVATION_OVERLAP_PCT": overlap_ratio,
                    "Dist_to_Muscle_um": d_m_um,
                    "Dist_to_Neuron_um": d_n_um,
                    "Dist_to_Muscle_center_um": d_m_center,
                    "Dist_to_Neuron_center_um": d_n_center,
                    "DETECTION_THRESHOLD_USED": threshold_used,
                    "is_NMJ": (d_m_um <= distance_threshold_um) and (d_n_um <= distance_threshold_um),
                })

            df_spots = pd.DataFrame(spots_data)
            if df_spots.empty:
                continue

            def classify_quadrant(row):
                if row["Dist_to_Muscle_um"] <= distance_threshold_um and row["Dist_to_Neuron_um"] <= distance_threshold_um:
                    return BTX_CLASS_EARLY_NMJ
                elif row["Dist_to_Muscle_um"] <= distance_threshold_um:
                    return BTX_CLASS_MUSCLE
                elif row["Dist_to_Neuron_um"] <= distance_threshold_um:
                    return BTX_CLASS_NEURON
                else:
                    return BTX_CLASS_ORPHANED

            df_spots["BTX signal class"] = df_spots.apply(classify_quadrant, axis=1)
            df_spots = normalize_btx_signal_classes(df_spots)
            df_spots["TOTAL_IMAGE_AREA_um2"] = total_image_area_um2
            df_spots["Resolution_Class"] = np.where(
                float(pixel_size) > RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL,
                "Low-Res",
                "High-Res",
            )

            total_spots = len(df_spots)

            # Outputs
            nmj_count = df_spots["is_NMJ"].sum()
            formation_rate = (nmj_count / total_spots * 100) if total_spots > 0 else 0.0

            near_m_only = len(
                df_spots[
                    (df_spots["Dist_to_Muscle_um"] <= distance_threshold_um)
                    & (df_spots["Dist_to_Neuron_um"] > distance_threshold_um)
                ]
            )
            near_n_only = len(
                df_spots[
                    (df_spots["Dist_to_Neuron_um"] <= distance_threshold_um)
                    & (df_spots["Dist_to_Muscle_um"] > distance_threshold_um)
                ]
            )
            orphaned = len(
                df_spots[
                    (df_spots["Dist_to_Muscle_um"] > distance_threshold_um)
                    & (df_spots["Dist_to_Neuron_um"] > distance_threshold_um)
                ]
            )

            # --- AREA-NORMALIZED DENSITY (NMJ vs muscle-only vs neuron-only vs orphan) ---
            mask_m_zone = edt_muscle_um <= distance_threshold_um
            mask_n_zone = edt_neuron_um <= distance_threshold_um
            um2_per_px = float(pixel_size**2)
            area_nmj_um2 = float(np.sum(mask_m_zone & mask_n_zone)) * um2_per_px
            area_m_um2 = float(np.sum(mask_m_zone & ~mask_n_zone)) * um2_per_px
            area_n_um2 = float(np.sum(mask_n_zone & ~mask_m_zone)) * um2_per_px
            area_o_um2 = float(np.sum(~mask_m_zone & ~mask_n_zone)) * um2_per_px

            c_nmj = int(nmj_count)
            c_m = int(near_m_only)
            c_n_only = int(near_n_only)
            c_orphan = int(orphaned)

            dens_nmj = (c_nmj / area_nmj_um2 * UM2_PER_MM2) if area_nmj_um2 > 0 else 0.0
            dens_m = (c_m / area_m_um2 * UM2_PER_MM2) if area_m_um2 > 0 else 0.0
            dens_n = (c_n_only / area_n_um2 * UM2_PER_MM2) if area_n_um2 > 0 else 0.0
            dens_o = (c_orphan / area_o_um2 * UM2_PER_MM2) if area_o_um2 > 0 else 0.0

            file_stem = os.path.splitext(czi_file)[0]
            out_csv = mirror_dataset_output_path(
                run_dir, data_root_abs, current_d, f"{file_stem}{_thr_tag}_analysis.csv"
            )
            df_spots.to_csv(out_csv, index=False)

            # Tag source file and stream to master CSV to avoid RAM blow-up on large all-folder runs
            df_spots_master = prepare_spot_table_for_master(
                df_spots,
                source_folder=os.path.basename(os.path.normpath(current_d)),
                source_image=czi_file,
                total_image_area_um2=total_image_area_um2,
            )
            df_spots_master.to_csv(master_csv, mode="a", header=(master_rows_written == 0), index=False)
            master_rows_written += len(df_spots_master)

            all_file_stats.append(
                {
                    "File": czi_file,
                    "Folder": os.path.basename(os.path.normpath(current_d)),
                    "TOTAL_IMAGE_AREA_um2": float(total_image_area_um2),
                    "Total Spots": total_spots,
                    BTX_CLASS_EARLY_NMJ: nmj_count,
                    BTX_CLASS_MUSCLE: near_m_only,
                    BTX_CLASS_NEURON: near_n_only,
                    BTX_CLASS_ORPHANED: orphaned,
                    "Formation Rate (%)": formation_rate,
                    DENSITY_COL_EARLY_NMJ: dens_nmj,
                    DENSITY_COL_MUSCLE: dens_m,
                    DENSITY_COL_NEURON: dens_n,
                    DENSITY_COL_ORPHANED: dens_o,
                    AREA_COL_EARLY_NMJ: area_nmj_um2,
                    AREA_COL_MUSCLE: area_m_um2,
                    AREA_COL_NEURON: area_n_um2,
                    AREA_COL_ORPHANED: area_o_um2,
                }
            )

            # Memory-safe fast path: skip all figure/composite creation unless PNG export is requested.
            if not save_pngs:
                progress.progress((i + 1) / len(all_target_czis))
                continue

            # Normalize images for composite display using percentiles (Auto Contrast)
            def auto_contrast(img):
                p_low, p_high = np.percentile(img, (5, 99.5))
                return rescale_intensity(img, in_range=(p_low, p_high), out_range=(0.0, 1.0))

            img_m_norm = auto_contrast(img_muscle)
            img_n_norm = auto_contrast(img_neuron)
            img_b_norm = auto_contrast(img_btx)

            def robust_minmax(img):
                p_low, p_high = np.percentile(img, (1, 99.8))
                if p_high <= p_low:
                    p_high = p_low + 1e-6
                return float(p_low), float(p_high)

            btx_clean_vis = img_btx.astype(np.float32)

            # Use a shared display range so raw/clean are visually comparable.
            disp_low, disp_high = robust_minmax(img_btx_raw.astype(np.float32))
            img_btx_raw_vis = rescale_intensity(img_btx_raw.astype(np.float32), in_range=(disp_low, disp_high), out_range=(0.0, 1.0))
            img_btx_clean_vis = rescale_intensity(btx_clean_vis, in_range=(disp_low, disp_high), out_range=(0.0, 1.0))
            raw_clean_side_by_side = np.concatenate([img_btx_raw_vis, img_btx_clean_vis], axis=1)
            
            # Composite RGB:
            # Red = Neuron + BTX (Yellow needs Red)
            # Green = Muscle + BTX (Yellow needs Green)
            # Blue = 0
            # We multiply BTX by a factor to make it pop even more as bright yellow
            comp_r = np.clip(img_n_norm + (img_b_norm * 1.2), 0, 1)
            comp_g = np.clip(img_m_norm + (img_b_norm * 1.2), 0, 1)
            comp_b = np.zeros_like(img_m_norm)
            composite_rgb = np.stack([comp_r, comp_g, comp_b], axis=-1)

            # Plot proximity graphs and images in a 4x3 grid.
            # Added a dedicated "Cleaned BTX only" panel next to the marked BTX view.
            fig = plt.figure(figsize=(24, 30))
            outer = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
            ax_scatter, ax_prox_kde_x, ax_prox_kde_y, ax_prox_title = proximity_joint_axes(
                fig, outer[0, 0], title_first=True
            )
            ax_size_kde = fig.add_subplot(outer[0, 1])
            ax_circ_kde = fig.add_subplot(outer[0, 2])
            ax_overlap_kde = fig.add_subplot(outer[1, 0])
            ax_intensity_kde = fig.add_subplot(outer[1, 1])
            ax_btx_clean = fig.add_subplot(outer[1, 2])
            ax_btx_only = fig.add_subplot(outer[2, 0])
            ax_btx_marked = fig.add_subplot(outer[2, 1])
            ax_comp_marked = fig.add_subplot(outer[2, 2])
            ax_comp_arrows = fig.add_subplot(outer[3, 0])
            ax_unused_1 = fig.add_subplot(outer[3, 1])
            ax_unused_2 = fig.add_subplot(outer[3, 2])
            
            
            # Graph 6: Roundness KDE (same rows as Kruskal title: ≥MIN pixels, three classes, no Orphaned)
            _roundness_order = list(ROUNDNESS_KRUSKAL_CLASSES)
            df_shape = dataframe_for_roundness_kde_and_kruskal(df_spots)
            if len(df_shape) > 0:
                sns.kdeplot(
                    data=df_shape,
                    x="ROUNDNESS",
                    hue="BTX signal class",
                    hue_order=_roundness_order,
                    palette=BTX_SIGNAL_CLASS_PALETTE,
                    ax=ax_circ_kde,
                    common_norm=False,
                    fill=True,
                    clip=(0, 1),
                    warn_singular=False,
                )
            roundness_title_img = roundness_3way_kruskal_title(
                df_spots,
                label_base=f"3. {BTX_CLASS_EARLY_NMJ} Roundness KDE (1 − eccentricity)",
            )
            ax_circ_kde.set_title(roundness_title_img)
            ax_circ_kde.set_xlabel("Roundness (1 = circle)")
            ax_circ_kde.set_ylabel('Probability Density')
            ax_circ_kde.set_xlim(0, 1)
            
            # Graph 7: Size KDE
            if len(df_spots) > 0:
                sns.kdeplot(
                    data=df_spots, x='RADIUS', hue='BTX signal class',
                    hue_order=BTX_SIGNAL_CLASS_ORDER,
                    palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_size_kde,
                    common_norm=False, fill=True,
                    warn_singular=False,
                )
            ax_size_kde.set_title('2. BTX Size KDE')
            ax_size_kde.set_xlabel('Radius (μm)')
            ax_size_kde.set_ylabel('Probability Density')
            
            # Graph 8: NMJ Innervation Histogram (NMJ class only)
            df_innervation_img = (
                df_spots[df_spots["BTX signal class"] == BTX_CLASS_EARLY_NMJ]
                if len(df_spots) > 0
                else df_spots
            )
            if len(df_innervation_img) > 0:
                sns.histplot(
                    data=df_innervation_img, x='INNERVATION_OVERLAP_PCT',
                    color=BTX_SIGNAL_CLASS_PALETTE[BTX_CLASS_EARLY_NMJ], ax=ax_overlap_kde,
                )
            ax_overlap_kde.set_title(f'4. {BTX_CLASS_EARLY_NMJ} Innervation Distribution')
            ax_overlap_kde.set_xlabel(f'{BTX_CLASS_EARLY_NMJ} Innervation (%)')
            ax_overlap_kde.set_ylabel('Count')
            
            # Graph 9: Mean Intensity KDE
            if len(df_spots) > 0:
                _int_vals_img = df_spots['MEAN_INTENSITY'].dropna() if 'MEAN_INTENSITY' in df_spots.columns else pd.Series(dtype=float)
                _int_max_img = float(_int_vals_img.quantile(0.999)) if len(_int_vals_img) > 0 else None
                sns.kdeplot(
                    data=df_spots, x='MEAN_INTENSITY', hue='BTX signal class',
                    hue_order=BTX_SIGNAL_CLASS_ORDER,
                    palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_intensity_kde,
                    common_norm=False, fill=True,
                    warn_singular=False,
                    clip=(0, _int_max_img) if _int_max_img is not None else None,
                )
                if _int_max_img is not None:
                    ax_intensity_kde.set_xlim(0, _int_max_img * 1.05)
            intensity_title_img, _intensity_summary_img = nmj_vs_orphan_intensity_mannwhitney_title(
                df_spots.assign(SOURCE_IMAGE=czi_file) if len(df_spots) else df_spots,
                label_base="5. Receptor Intensity KDE",
            )
            ax_intensity_kde.set_title(intensity_title_img)
            ax_intensity_kde.set_xlabel('Mean Fluorescence Intensity')
            ax_intensity_kde.set_ylabel('Probability Density')
            
            draw_proximity_joint(
                ax_scatter,
                ax_prox_kde_x,
                ax_prox_kde_y,
                df_spots,
                distance_threshold_um,
                "1. BTX Proximity Analysis",
                marginal_combined_black=True,
                title_ax=ax_prox_title,
            )

            # Graph 2: Raw and cleaned BTX shown side-by-side for subtraction verification
            # ``aspect='auto'``: default ``equal`` can collapse wide 2×-width panels in ``GridSpec`` + ``tight`` bbox.
            ax_btx_clean.imshow(
                raw_clean_side_by_side, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto"
            )
            pane_w = img_btx_raw_vis.shape[1]
            ax_btx_clean.axvline(x=pane_w - 0.5, color='yellow', linewidth=2.5)
            ax_btx_clean.set_title("6. Raw BTX (L) | Cleaned BTX (R)")
            ax_btx_clean.axis('off')

            # Graph 3: Strictly scaled cleaned BTX only
            ax_btx_only.imshow(img_btx_clean_vis, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
            ax_btx_only.set_title("7. Cleaned BTX")
            ax_btx_only.axis('off')

            # Graph 4: Strictly scaled cleaned BTX overlaid with spots
            ax_btx_marked.imshow(img_btx_clean_vis, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
            ax_btx_marked.set_title("8. Cleaned BTX + Detected Spots")
            ax_btx_marked.axis('off')

            # Graph 5: Composite Image + All Spots
            ax_comp_marked.imshow(composite_rgb, aspect="auto")
            ax_comp_marked.set_title("9. Composite + All Detected Spots")
            ax_comp_marked.axis('off')
            
            # Graph 6: Composite Image + NMJ Arrows
            ax_comp_arrows.imshow(composite_rgb, aspect="auto")
            ax_comp_arrows.set_title(f"10. Composite + {BTX_CLASS_EARLY_NMJ} Only")
            ax_comp_arrows.axis('off')
            dens_data = pd.DataFrame({
                "Zone": [BTX_CLASS_EARLY_NMJ, BTX_CLASS_MUSCLE, BTX_CLASS_NEURON, BTX_CLASS_ORPHANED],
                "Density": [dens_nmj, dens_m, dens_n, dens_o],
            })
            sns.barplot(
                data=dens_data,
                x="Zone",
                y="Density",
                hue="Zone",
                palette=["red", "green", "blue", "gray"],
                legend=False,
                ax=ax_unused_1,
            )
            ax_unused_1.set_title(f"11. BTX Density ({SPOT_DENSITY_PER_MM2_LABEL})")
            ax_unused_1.set_ylabel(SPOT_DENSITY_PER_MM2_LABEL)
            ax_unused_1.set_xlabel("")
            ax_unused_2.axis('off')
            
            by_spot_id = df_spots.set_index("SPOT_ID")
            # Plot the overlays
            for index, blob in enumerate(blobs):
                y, x, r = blob
                c1 = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
                c2 = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
                ax_btx_marked.add_patch(c1)
                ax_comp_marked.add_patch(c2)
                
                # If this spot is functionally classified as an NMJ, point an arrow at it on the final layout
                if index in by_spot_id.index and bool(by_spot_id.at[index, "is_NMJ"]):
                    # The arrow points to the very edge of the radius (x+r, y-r) so it doesn't cover the spot itself.
                    target_x = x + r + 2
                    target_y = y - r - 2
                    # Extended the starting point heavily (80 pixels) so the arrow has a clearly visible long tail on large images
                    start_x = target_x + 80
                    start_y = target_y - 80
                    
                    # "-|>" creates a solid, closed triangle arrowhead instead of an open "v" shape
                    ax_comp_arrows.annotate('', xy=(target_x, target_y), xytext=(start_x, start_y),
                                      arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))

            file_stem = os.path.splitext(czi_file)[0]
            out_img = mirror_dataset_output_path(
                run_dir, data_root_abs, current_d, f"{file_stem}{_thr_tag}_NMJ_Plot.png"
            )
            fig.savefig(out_img, bbox_inches="tight")
            # Figure teardown runs in ``finally`` (single ``plt.close(fig)``, thread-safe vs ``close('all')``).
            
        except Exception as e:
            st.warning(f"Analysis failed organically on {czi_file}: {e}")
        finally:
            # CRITICAL: `del locals()[var]` is a no-op in CPython — when Streamlit runs
            # the script via exec(code, globals, locals) the dict returned by locals() is
            # a snapshot, so mutating it does not free the real bindings. The previous
            # implementation therefore leaked every iteration's images and matplotlib
            # state, causing the container to OOM-kill (exit 137) on multi-folder runs.
            # Explicitly rebind the heavy per-iteration buffers to None so reference
            # counts drop before gc.collect() runs.
            image_data = None
            channels = None
            img_muscle = None
            img_neuron = None
            img_btx = None
            img_btx_raw = None
            img_btx_norm = None
            blobs = None
            muscle_mask = None
            neuron_mask = None
            edt_muscle_px = None
            edt_neuron_px = None
            edt_muscle_um = None
            edt_neuron_um = None
            spots_data = None
            df_spots = None
            df_spots_master = None
            raw_clean_side_by_side = None
            composite_rgb = None
            img_btx_raw_vis = None
            img_btx_clean_vis = None
            btx_clean_vis = None
            img_m_norm = None
            img_n_norm = None
            img_b_norm = None
            comp_r = None
            comp_g = None
            comp_b = None
            window_btx = None
            window_neuron = None
            labeled = None
            spot_mask = None
            _fig_cleanup = fig
            fig = None
            axes = None
            if _fig_cleanup is not None:
                try:
                    _fig_cleanup.clf()
                except Exception:
                    pass
                plt.close(_fig_cleanup)
            gc.collect()
            
        progress.progress((i + 1) / len(all_target_czis))
        
    # --- AFTER BATCH COMPLETES ---
    status.write("✅ **Batch Processing Complete!**")
    
    if master_rows_written > 0:
        master_df = finalize_master_results_dataframe(
            pd.read_csv(master_csv),
            file_stats=all_file_stats,
        )
        master_df.to_csv(master_csv, index=False)
        st.success(f"Aggregate Master dataset uniquely saved: `{master_csv}`")
        
        st.subheader("📈 Batch Statistical Summary")

        if run_all:
            folder_stats_df = (
                master_df.groupby("SOURCE_FOLDER")
                .agg(
                    total_spots=("is_NMJ", "size"),
                    nmj_spots=("is_NMJ", "sum"),
                    mean_radius_um=("RADIUS", "mean"),
                    mean_overlap_pct=("INNERVATION_OVERLAP_PCT", "mean"),
                    median_dist_muscle_um=("Dist_to_Muscle_um", "median"),
                    median_dist_neuron_um=("Dist_to_Neuron_um", "median"),
                )
                .reset_index()
                .sort_values("SOURCE_FOLDER")
            )
            folder_stats_df["nmj_rate_pct"] = np.where(
                folder_stats_df["total_spots"] > 0,
                folder_stats_df["nmj_spots"] / folder_stats_df["total_spots"] * 100.0,
                0.0,
            )
            folder_stats_df.to_csv(summary_table_csv, index=False)
            st.success(f"All-folders summary table saved: `{summary_table_csv}`")
            # Panel 6 (Friedman) uses per-image zone densities — not derivable from the spot-level master CSV alone.
            file_stats_csv = os.path.join(run_dir, f"ALL_FOLDERS_FILE_STATS{_thr_tag}.csv")
            pd.DataFrame(all_file_stats).to_csv(file_stats_csv, index=False)
            st.success(
                f"Per-image zone-density table saved (for panel 6 / dashboard regeneration): `{file_stats_csv}`"
            )
        elif not run_all and all_file_stats:
            file_stats_csv = os.path.join(run_dir, f"BATCH_FILE_STATS{_thr_tag}.csv")
            pd.DataFrame(all_file_stats).to_csv(file_stats_csv, index=False)
            st.success(f"Per-image zone-density table saved: `{file_stats_csv}`")

        # Create summary dashboard (shared with ``regenerate_all_folders_panel_pdfs.py``).
        fig, panel_specs, dash_meta = build_aggregate_batch_dashboard_figure(
            master_df,
            distance_threshold_um,
            run_all=run_all,
            all_file_stats=all_file_stats,
        )

        stat_summary_df, image_medians_df, otsu_dim_noise_df, paired_intensity_images_df = build_batch_stat_summary_dataframe(
            master_df,
            distance_threshold_um=distance_threshold_um,
            dash_meta=dash_meta,
            run_all=run_all,
        )
        if run_all:
            stat_summary_csv = os.path.join(run_dir, f"ALL_FOLDERS_STAT_SUMMARY{_thr_tag}.csv")
            image_medians_csv = os.path.join(run_dir, f"ALL_FOLDERS_IMAGE_LEVEL_MEDIANS{_thr_tag}.csv")
            otsu_rejection_csv = os.path.join(run_dir, f"ALL_FOLDERS_OTSU_DIM_NOISE_REJECTION{_thr_tag}.csv")
            paired_intensity_images_csv = os.path.join(
                run_dir, f"ALL_FOLDERS_INTENSITY_PAIRED_IMAGES{_thr_tag}.csv"
            )
            paired_otsu_spot_change_csv = os.path.join(
                run_dir, f"ALL_FOLDERS_PAIRED_OTSU_SPOT_CHANGE{_thr_tag}.csv"
            )
            otsu_spot_count_matrix_csv = os.path.join(
                run_dir, f"ALL_FOLDERS_OTSU_SPOT_COUNT_MATRIX{_thr_tag}.csv"
            )
            otsu_spot_change_comparisons_csv = os.path.join(
                run_dir, f"ALL_FOLDERS_OTSU_SPOT_CHANGE_COMPARISONS{_thr_tag}.csv"
            )
        else:
            stat_summary_csv = os.path.join(run_dir, f"BATCH_STAT_SUMMARY{_thr_tag}.csv")
            image_medians_csv = os.path.join(run_dir, f"BATCH_IMAGE_LEVEL_MEDIANS{_thr_tag}.csv")
            otsu_rejection_csv = os.path.join(run_dir, f"BATCH_OTSU_DIM_NOISE_REJECTION{_thr_tag}.csv")
            paired_intensity_images_csv = os.path.join(
                run_dir, f"BATCH_INTENSITY_PAIRED_IMAGES{_thr_tag}.csv"
            )
            paired_otsu_spot_change_csv = os.path.join(
                run_dir, f"BATCH_PAIRED_OTSU_SPOT_CHANGE{_thr_tag}.csv"
            )
            otsu_spot_count_matrix_csv = os.path.join(
                run_dir, f"BATCH_OTSU_SPOT_COUNT_MATRIX{_thr_tag}.csv"
            )
            otsu_spot_change_comparisons_csv = os.path.join(
                run_dir, f"BATCH_OTSU_SPOT_CHANGE_COMPARISONS{_thr_tag}.csv"
            )
        stat_summary_df.to_csv(stat_summary_csv, index=False)
        if len(image_medians_df) > 0:
            image_medians_df.to_csv(image_medians_csv, index=False)
            st.success(f"Per-image class medians saved: `{image_medians_csv}`")
        if len(otsu_dim_noise_df) > 0:
            otsu_dim_noise_df.to_csv(otsu_rejection_csv, index=False)
            st.success(f"Otsu dim-noise rejection table saved: `{otsu_rejection_csv}`")
        paired_intensity_images_df.to_csv(paired_intensity_images_csv, index=False)
        n_paired_spots = len(paired_intensity_images_df)
        if n_paired_spots > 0 and {"SOURCE_FOLDER", "SOURCE_IMAGE"} <= set(paired_intensity_images_df.columns):
            n_paired_images = int(
                paired_intensity_images_df[["SOURCE_FOLDER", "SOURCE_IMAGE"]].drop_duplicates().shape[0]
            )
        else:
            n_paired_images = 0
        st.success(
            f"Intensity paired-comparison spot table saved: `{paired_intensity_images_csv}` "
            f"({n_paired_spots} spot{'s' if n_paired_spots != 1 else ''} "
            f"from {n_paired_images} image{'s' if n_paired_images != 1 else ''})"
        )
        paired_otsu_spot_change_df = dash_meta.get("paired_otsu_spot_change_df")
        otsu_spot_count_matrix_df = dash_meta.get("otsu_spot_count_matrix_df")
        otsu_spot_change_comparisons_df = dash_meta.get("otsu_spot_change_comparisons_df")
        if otsu_spot_count_matrix_df is not None and len(otsu_spot_count_matrix_df) > 0:
            present_otsu_spot_count_matrix(otsu_spot_count_matrix_df).to_csv(
                otsu_spot_count_matrix_csv,
                index=False,
            )
            st.success(f"Otsu spot-count matrix saved: `{otsu_spot_count_matrix_csv}`")
        if otsu_spot_change_comparisons_df is not None and len(otsu_spot_change_comparisons_df) > 0:
            present_otsu_spot_change_comparisons(otsu_spot_change_comparisons_df).to_csv(
                otsu_spot_change_comparisons_csv,
                index=False,
            )
            st.success(f"Otsu spot-change comparisons saved: `{otsu_spot_change_comparisons_csv}`")
        if paired_otsu_spot_change_df is not None:
            paired_otsu_spot_change_df.to_csv(paired_otsu_spot_change_csv, index=False)
            st.success(
                f"Legacy mixed Otsu spot-change table saved: `{paired_otsu_spot_change_csv}` "
                "(global@all vs paired@paired-cohort)"
            )
        if otsu_spot_count_matrix_df is not None and len(otsu_spot_count_matrix_df) > 0:
            st.markdown("#### Otsu spot-count sensitivity (2×2 factorial)")
            st.caption(
                "Each column counts spots at or above the stated Otsu threshold within the "
                "stated image set (all images vs paired-cohort images with both early NMJ-like "
                "and Orphaned spots)."
            )
            st.dataframe(present_otsu_spot_count_matrix(otsu_spot_count_matrix_df))
        if otsu_spot_change_comparisons_df is not None and len(otsu_spot_change_comparisons_df) > 0:
            st.markdown("#### Otsu spot-count comparisons")
            st.dataframe(present_otsu_spot_change_comparisons(otsu_spot_change_comparisons_df))
        st.success(f"Statistical test summary saved: `{stat_summary_csv}`")

        global_otsu = dash_meta.get("global_btx_intensity_otsu")
        p_friedman_ab = dash_meta.get("friedman_p_abundance")
        p_friedman_ab_otsu = dash_meta.get("friedman_p_abundance_otsu")
        
        if p_friedman_ab is not None or p_friedman_ab_otsu is not None:
            st.markdown("### 🧪 Statistical Analysis Summary")
            st.caption(
                "Primary inference rows (`primary_image_level`, `primary_posthoc`) use per-image class "
                "medians; `primary_sensitivity` rows are unpaired alternatives; exploratory rows "
                "(`exploratory_spot_pooled`) pool all spots for visualization only."
            )
            st.dataframe(stat_summary_df)
            if len(otsu_dim_noise_df) > 0:
                st.markdown("#### Otsu dim-noise rejection (spot composition by class)")
                st.caption(
                    "Descriptive spot-pooled table: fraction of each class above the global intensity "
                    "Otsu threshold. Supports — but does not replace — image-level paired Wilcoxon tests."
                )
                st.dataframe(
                    otsu_dim_noise_df[
                        [
                            "btx_signal_class",
                            "n_spots",
                            "n_spots_above_otsu",
                            "pct_spots_above_otsu",
                            "global_otsu_threshold_au",
                        ]
                    ]
                )
            otsu_note = (
                f"{global_otsu:.1f} A.U." if global_otsu is not None and np.isfinite(global_otsu) else "n/a"
            )
            if p_friedman_ab is not None:
                st.markdown("#### Global BTX Abundance (all detected spots)")
                if p_friedman_ab < 0.05:
                    st.success(
                        "Zone abundance differs significantly across images "
                        f"(Friedman p={p_friedman_ab:.4e})."
                    )
                else:
                    st.warning(
                        "Zone abundance differences did not reach significance "
                        f"(Friedman p={p_friedman_ab:.4g})."
                    )
            if p_friedman_ab_otsu is not None:
                st.markdown(f"#### Global BTX Abundance (spots ≥ Otsu {otsu_note})")
                if p_friedman_ab_otsu < 0.05:
                    st.success(
                        "Otsu-filtered zone abundance differs significantly "
                        f"(Friedman p={p_friedman_ab_otsu:.4e})."
                    )
                else:
                    st.warning(
                        "Otsu-filtered zone abundance did not reach significance "
                        f"(Friedman p={p_friedman_ab_otsu:.4g})."
                    )

            conover_df_ab = dash_meta.get("conover_abundance_results")
            if conover_df_ab is not None and not conover_df_ab.empty:
                display_df_ab = conover_df_ab.copy().rename(columns={
                    "group1": "Zone 1",
                    "group2": "Zone 2",
                    "t_stat": "t-statistic",
                    "p_val": "p-value (raw)",
                    "p_val_adj": "p-value (adjusted)",
                    "sig": "Significance",
                })
                cols_order = ["Zone 1", "Zone 2", "t-statistic", "p-value (raw)", "p-value (adjusted)", "Significance"]
                st.dataframe(display_df_ab[cols_order].style.format({
                    "t-statistic": "{:.3f}",
                    "p-value (raw)": "{:.4e}",
                    "p-value (adjusted)": "{:.4e}"
                }))

        # Persist PNG + per-panel PDFs before ``st.pyplot`` — Streamlit may clear the figure
        # after display unless ``clear_figure=False``, which would otherwise yield empty PDFs.
        fig.savefig(master_png, bbox_inches="tight")

        stem, _ext = os.path.splitext(master_png)

        written_pdfs, pdf_export_errors = _export_figure_panels_to_pdfs(fig, panel_specs, stem)
        if len(written_pdfs) < len(panel_specs):
            err_tail = ""
            if pdf_export_errors:
                err_tail = " Details: " + " | ".join(pdf_export_errors[:5])
                if len(pdf_export_errors) > 5:
                    err_tail += f" … (+{len(pdf_export_errors) - 5} more)"
            st.warning(
                f"Per-panel PDF export: wrote {len(written_pdfs)}/{len(panel_specs)} files under "
                f"`{os.path.abspath(os.path.dirname(stem) or '.')}`."
                + err_tail
            )
        pdf_note = ""
        if written_pdfs:
            pdf_note = (
                f" Also saved {len(written_pdfs)} per-panel PDF(s) next to the dashboard "
                f"(e.g. `{os.path.basename(written_pdfs[0])}`)."
            )

        st.pyplot(fig, clear_figure=False)

        fig.clf()
        plt.close(fig)

        st.success(f"Aggregate Dashboard generated: `{master_png}`.{pdf_note}")

        # Drop the (potentially huge) aggregate DataFrame and any derived helpers so
        # the script's resident memory shrinks back down before Streamlit re-runs.
        master_df = None
        try:
            del folder_stats_df
        except NameError:
            pass
        gc.collect()

    if all_file_stats:
        st.subheader("📊 Batch Summary Metrics")
        st.dataframe(pd.DataFrame(all_file_stats))

    run_config["completed_at"] = datetime.now().isoformat(timespec="seconds")
    run_config["images_analyzed"] = len(all_file_stats)
    if master_rows_written > 0:
        run_config["master_csv"] = os.path.relpath(master_csv, run_dir)
        if os.path.isfile(master_png):
            run_config["summary_png"] = os.path.relpath(master_png, run_dir)
        try:
            if stat_summary_csv and os.path.isfile(stat_summary_csv):
                run_config["stat_summary_csv"] = os.path.relpath(stat_summary_csv, run_dir)
            if paired_intensity_images_csv and os.path.isfile(paired_intensity_images_csv):
                run_config["paired_intensity_images_csv"] = os.path.relpath(
                    paired_intensity_images_csv, run_dir
                )
            if paired_otsu_spot_change_csv and os.path.isfile(paired_otsu_spot_change_csv):
                run_config["paired_otsu_spot_change_csv"] = os.path.relpath(
                    paired_otsu_spot_change_csv, run_dir
                )
            if otsu_spot_count_matrix_csv and os.path.isfile(otsu_spot_count_matrix_csv):
                run_config["otsu_spot_count_matrix_csv"] = os.path.relpath(
                    otsu_spot_count_matrix_csv, run_dir
                )
            if otsu_spot_change_comparisons_csv and os.path.isfile(otsu_spot_change_comparisons_csv):
                run_config["otsu_spot_change_comparisons_csv"] = os.path.relpath(
                    otsu_spot_change_comparisons_csv, run_dir
                )
        except NameError:
            pass
    save_run_config_files(run_dir, run_config, channel_snapshot)
    st.session_state["last_run_dir"] = run_dir
    render_streamlit_download_section(st, run_dir)
