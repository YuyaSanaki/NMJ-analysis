import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import aicspylibczi
import gc
from skimage.feature import blob_dog
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt, zoom
from skimage.exposure import rescale_intensity

BTX_SIGNAL_CLASS_ORDER = ("NMJ", "Aneural AChR clusters", "Neuron-associated BTX signal", "Orphaned")
BTX_SIGNAL_CLASS_PALETTE = {
    "NMJ": "red",
    "Aneural AChR clusters": "green",
    "Neuron-associated BTX signal": "blue",
    "Orphaned": "gray",
}

# Legacy strings written by older runs / per-image CSVs before terminology update
BTX_SIGNAL_CLASS_LEGACY_ALIASES = {
    "Muscle Only": "Aneural AChR clusters",
    "Muscle only": "Aneural AChR clusters",
    "Neuron Only": "Neuron-associated BTX signal",
    "Neuron only": "Neuron-associated BTX signal",
}


def normalize_btx_signal_classes(df):
    """Map legacy BTX signal class labels so plots/counts match BTX_SIGNAL_CLASS_*."""
    if df is None or len(df) == 0 or "BTX signal class" not in df.columns:
        return df
    out = df.copy()
    out["BTX signal class"] = (
        out["BTX signal class"].astype(str).str.strip().replace(BTX_SIGNAL_CLASS_LEGACY_ALIASES)
    )
    return out

st.set_page_config(page_title="NMJ Pipeline", layout="wide")

st.title("🔬 Multiple-Image Batch NMJ Pipeline")
st.markdown("Select a folder to batch-process all `.czi` files automatically.")

# --- 1. Folder & File Selection ---
base_dir = "."
folders = [d for d in os.listdir(base_dir) if os.path.isdir(d) and not d.startswith(".") and d != "__pycache__"]

if not folders:
    st.warning("No data folders found.")
    st.stop()

selected_folder = st.selectbox("📂 Select Dataset Folder", sorted(folders))
folder_path = os.path.join(base_dir, selected_folder)
files_in_folder = os.listdir(folder_path)

czi_files = [f for f in files_in_folder if f.endswith(".czi")]

if not czi_files:
    st.error(f"No `.czi` files found in '{selected_folder}'. Please ensure your raw confocal data is there.")
    st.stop()

# --- 2. Extract Metadata & Config for Batch ---
@st.cache_data(show_spinner=False)
def fast_czi_meta(path):
    czi = None
    try:
        czi = aicspylibczi.CziFile(path)
        dims = czi.get_dims_shape()[0]
        cc = dims.get('C', [0, 4])
        num_channels = cc[1] - cc[0]

        pixel_size_um = 1.0  # Default fallback
        try:
            for dist in czi.meta.findall('.//Distance'):
                if dist.attrib.get('Id') == 'X':
                    val = dist.find('Value')
                    if val is not None:
                        pixel_size_um = float(val.text) * 1e6
                        break
        except Exception:
            pass
        return num_channels, pixel_size_um
    finally:
        if czi is not None and hasattr(czi, "close"):
            try:
                czi.close()
            except Exception:
                pass

st.subheader("⚙️ Batch Channel Mapping")

# --- Import / Export Logic ---
c_exp, c_imp, _ = st.columns([1.5, 1.5, 3])
config_json_path = os.path.join(folder_path, "channel_mapping_config.json")

if c_exp.button("💾 Save Settings to Folder"):
    import json
    export_data = {}
    for f in czi_files:
        if f"m_{f}" in st.session_state:
            export_data[f] = {
                "m": st.session_state.get(f"m_{f}", 0),
                "n": st.session_state.get(f"n_{f}", 0),
                "b": st.session_state.get(f"b_{f}", 0),
                "p": st.session_state.get(f"p_{f}", 1.0),
                "skip": st.session_state.get(f"skip_{f}", False)
            }
    try:
        with open(config_json_path, "w") as jf:
            json.dump(export_data, jf, indent=4)
        st.success("Config saved successfully.")
    except Exception as e:
        st.error(f"Failed to save: {e}")

if c_imp.button("📂 Load Settings from Folder"):
    import json
    if os.path.exists(config_json_path):
        try:
            with open(config_json_path, "r") as jf:
                imported_data = json.load(jf)
            for cf in czi_files:
                if cf in imported_data:
                    st.session_state[f"m_{cf}"] = imported_data[cf]["m"]
                    st.session_state[f"n_{cf}"] = imported_data[cf]["n"]
                    st.session_state[f"b_{cf}"] = imported_data[cf]["b"]
                    st.session_state[f"p_{cf}"] = imported_data[cf]["p"]
                    st.session_state[f"skip_{cf}"] = imported_data[cf].get("skip", False)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load: {e}")
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
            st.session_state[f"m_{cf}"] = min(g_m, n_ch_tmp-1)
            st.session_state[f"n_{cf}"] = min(g_n, n_ch_tmp-1)
            st.session_state[f"b_{cf}"] = min(g_b, n_ch_tmp-1)
            # Intentionally NOT overriding pixel size to preserve true biological scaling!

    st.divider()
    st.markdown("### 📂 Individual File Settings")
    for czi_file in czi_files:
        path_tmp = os.path.join(folder_path, czi_file)
        n_ch_ind, px_size_ind = fast_czi_meta(path_tmp)
        options_ind = [f"Channel {i+1}" for i in range(n_ch_ind)]
        
        # Initialize session state so the selectboxes don't error out when natively bound
        if f"m_{czi_file}" not in st.session_state: st.session_state[f"m_{czi_file}"] = 0
        if f"n_{czi_file}" not in st.session_state: st.session_state[f"n_{czi_file}"] = min(1, n_ch_ind-1)
        if f"b_{czi_file}" not in st.session_state: st.session_state[f"b_{czi_file}"] = min(3, n_ch_ind-1)
        if f"p_{czi_file}" not in st.session_state: st.session_state[f"p_{czi_file}"] = float(px_size_ind)
        if f"skip_{czi_file}" not in st.session_state: st.session_state[f"skip_{czi_file}"] = False
        
        with st.expander(f"▶️ Config: {czi_file}", expanded=False):
            
            c_skip, c_paste, _ = st.columns([1.5, 1.5, 3])
            skip_file = c_skip.checkbox("🚫 Exclude image from batch", key=f"skip_{czi_file}")
            
            # The Paste Button directly artificially modifies the session state properties of the inputs below!
            if c_paste.button("📋 Paste Template Here", key=f"btn_{czi_file}"):
                st.session_state[f"m_{czi_file}"] = min(g_m, n_ch_ind-1)
                st.session_state[f"n_{czi_file}"] = min(g_n, n_ch_ind-1)
                st.session_state[f"b_{czi_file}"] = min(g_b, n_ch_ind-1)
                # Intentionally NOT overriding pixel size here!
                st.rerun() # Force immediate UI refresh to show the newly pasted values

            if skip_file:
                st.warning("Image will be bypassed during batch processing.")
            
            c1, c2, c3, c4 = st.columns(4)
            m_id = c1.selectbox("Muscle", range(n_ch_ind), format_func=lambda x: options_ind[x], key=f"m_{czi_file}", disabled=skip_file)
            n_id = c2.selectbox("Neuron", range(n_ch_ind), format_func=lambda x: options_ind[x], key=f"n_{czi_file}", disabled=skip_file)
            b_id = c3.selectbox("BTX", range(n_ch_ind), format_func=lambda x: options_ind[x], key=f"b_{czi_file}", disabled=skip_file)
            ps = c4.number_input("Pixel Size", format="%0.7f", key=f"p_{czi_file}", help="Unique biological scale.", disabled=skip_file)
            
            file_configs[czi_file] = {"muscle": m_id, "neuron": n_id, "btx": b_id, "pixel_size": ps, "skip": skip_file}

st.divider()


def _czi_channel_zmax_2d(czi, c_idx, dims0):
    """Return a single (Y, X) plane as Z-max projection with *one Z plane in RAM at a time*.

    ``read_image(C=...)`` loads the full Z stack for that channel (e.g. 25×2576×2576 ≈ 316 MB
    for `0714M-HF-03.czi`), while ``DesBTXNEFM`` tiles use Z=13 (~164 MB). Deep Z stacks alone
    explain why the former can OOM Docker/Streamlit even when XY matches. Streaming Z reduces
    peak read memory to ~one plane (~13 MB for 2576² uint16) plus the accumulator.
    """
    z_rng = dims0.get("Z", (0, 1))
    z0, z1 = int(z_rng[0]), int(z_rng[1])
    if z1 - z0 <= 1:
        img, _ = czi.read_image(C=c_idx)
        arr = np.squeeze(np.asarray(img))
        while arr.ndim > 2:
            arr = np.max(arr, axis=0)
        return arr.copy() if not arr.flags.owndata else arr

    acc = None
    for zi in range(z0, z1):
        img, _ = czi.read_image(C=c_idx, Z=zi)
        plane = np.squeeze(np.asarray(img))
        while plane.ndim > 2:
            plane = np.max(plane, axis=0)
        if acc is None:
            acc = plane.copy()
        else:
            np.maximum(acc, plane, out=acc)
        del img
    return acc


# Removed st.cache_data decorator to prevent catastrophic Out-Of-Memory (OOM) accumulation during massive multi-GB dataset runs
def load_czi_image(path, channel_indices=None):
    """Load a CZI image as 2D channel arrays.

    When ``channel_indices`` is provided, only the requested channels are read
    from disk and projected (Z-max). This dramatically reduces peak memory on
    multi-GB / multi-channel CZIs, which is the dominant driver of the
    container OOM-kill (exit 137) during ``Run Batch (ALL Folders)``. The
    legacy 3D ``(C, Y, X)`` array path is kept as a safe fallback for
    pathological CZI layouts where per-channel reads fail.
    """
    czi = None
    try:
        czi = aicspylibczi.CziFile(path)

        if channel_indices is not None:
            wanted = list(dict.fromkeys(int(c) for c in channel_indices))
            try:
                dims0 = czi.get_dims_shape()[0]
                channels = {}
                for c_idx in wanted:
                    channels[c_idx] = _czi_channel_zmax_2d(czi, c_idx, dims0)
                return channels
            except Exception:
                # Fall through to the legacy single-shot read on incompatible CZIs.
                pass

        img, _ = czi.read_image()
        img_sq = np.squeeze(img)
        if img_sq.ndim == 4:
            img_sq = np.max(img_sq, axis=1)
        if img_sq.ndim < 3:
            img_sq = np.expand_dims(img_sq, axis=0)
        # Detect (Y, X, C) layout: last axis small (≤10, typical channel count) and smaller than first
        if img_sq.ndim == 3 and img_sq.shape[-1] <= 10 and img_sq.shape[-1] < img_sq.shape[0]:
            img_sq = np.moveaxis(img_sq, -1, 0)

        if channel_indices is not None:
            wanted = list(dict.fromkeys(int(c) for c in channel_indices))
            # .copy() detaches each plane so the giant parent volume can be freed.
            return {c_idx: img_sq[c_idx].copy() for c_idx in wanted}
        return img_sq
    finally:
        if czi is not None and hasattr(czi, "close"):
            try:
                czi.close()
            except Exception:
                pass


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


def detect_blobs_stable(img_btx_norm, min_diameter_um, max_diameter_um, pixel_size_um, threshold):
    """Run DoG in a memory-safe way using diameter thresholds (µm)."""
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
    """Estimate a robust per-image DoG threshold from upper-tail intensity."""
    values = np.asarray(img_btx_norm, dtype=np.float32).ravel()
    if values.size == 0:
        return 0.05

    # Bound memory/CPU for very large images by sampling a fixed-size subset.
    sample_cap = 300_000
    if values.size > sample_cap:
        step = int(np.ceil(values.size / sample_cap))
        values = values[::step]

    p99 = float(np.percentile(values, 99.0))
    # Use a percentile-scaled threshold to stay in a practical DoG range across tiles.
    auto_thr = 0.12 * p99
    return float(np.clip(auto_thr, 0.01, 0.10))


def save_all_folders_summary_png(master_df, out_png, distance_threshold_um):
    """Create a single mother-directory PNG summarizing all folders."""
    from scipy.stats import fisher_exact

    master_df = normalize_btx_signal_classes(master_df)

    folder_stats = (
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
    folder_stats["nmj_rate_pct"] = np.where(
        folder_stats["total_spots"] > 0,
        folder_stats["nmj_spots"] / folder_stats["total_spots"] * 100.0,
        0.0,
    )

    total_nmj = int((master_df["BTX signal class"] == "NMJ").sum())
    total_m_only = int((master_df["BTX signal class"] == "Aneural AChR clusters").sum())
    total_n_only = int((master_df["BTX signal class"] == "Neuron-associated BTX signal").sum())
    total_orph = int((master_df["BTX signal class"] == "Orphaned").sum())
    _, global_fisher_p = fisher_exact([[total_nmj, total_m_only], [total_n_only, total_orph]])

    fig, axes = plt.subplots(3, 2, figsize=(22, 24))
    ax_nmj_rate = axes[0, 0]
    ax_total_spots = axes[0, 1]
    ax_radius = axes[1, 0]
    ax_overlap = axes[1, 1]
    ax_distance = axes[2, 0]
    ax_proximity = axes[2, 1]

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="nmj_rate_pct", ax=ax_nmj_rate, color="#d62728")
    ax_nmj_rate.set_title("1. NMJ Formation Rate by Folder")
    ax_nmj_rate.set_xlabel("Folder")
    ax_nmj_rate.set_ylabel("NMJ Rate (%)")
    ax_nmj_rate.tick_params(axis="x", rotation=45)

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="total_spots", ax=ax_total_spots, color="#1f77b4")
    ax_total_spots.set_title("2. Total BTX Spots by Folder")
    ax_total_spots.set_xlabel("Folder")
    ax_total_spots.set_ylabel("Count")
    ax_total_spots.tick_params(axis="x", rotation=45)

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="mean_radius_um", ax=ax_radius, color="#2ca02c")
    ax_radius.set_title("3. Mean Spot Radius by Folder")
    ax_radius.set_xlabel("Folder")
    ax_radius.set_ylabel("Radius (um)")
    ax_radius.tick_params(axis="x", rotation=45)

    sns.barplot(data=folder_stats, x="SOURCE_FOLDER", y="mean_overlap_pct", ax=ax_overlap, color="#9467bd")
    ax_overlap.set_title("4. Mean Innervation Overlap by Folder")
    ax_overlap.set_xlabel("Folder")
    ax_overlap.set_ylabel("Overlap (%)")
    ax_overlap.tick_params(axis="x", rotation=45)

    sns.scatterplot(
        data=folder_stats,
        x="median_dist_muscle_um",
        y="median_dist_neuron_um",
        hue="SOURCE_FOLDER",
        s=120,
        ax=ax_distance,
    )
    ax_distance.axvline(x=distance_threshold_um, color="black", linestyle="--")
    ax_distance.axhline(y=distance_threshold_um, color="black", linestyle="--")
    ax_distance.set_title("5. Folder Medians in Distance Space")
    ax_distance.set_xlabel("Median Dist to Muscle (um)")
    ax_distance.set_ylabel("Median Dist to Neuron (um)")

    sns.scatterplot(
        data=master_df,
        x="Dist_to_Muscle_um",
        y="Dist_to_Neuron_um",
        hue="BTX signal class",
        hue_order=BTX_SIGNAL_CLASS_ORDER,
        palette=BTX_SIGNAL_CLASS_PALETTE,
        alpha=0.35,
        s=18,
        ax=ax_proximity,
    )
    ax_proximity.axvline(x=distance_threshold_um, color="black", linestyle="--")
    ax_proximity.axhline(y=distance_threshold_um, color="black", linestyle="--")
    sig_star = "***" if global_fisher_p < 0.001 else "**" if global_fisher_p < 0.01 else "*" if global_fisher_p < 0.05 else "ns"
    ax_proximity.set_title(f"6. All-Folders Proximity (Fisher P = {global_fisher_p:.4g} {sig_star})")
    ax_proximity.set_xlabel("Distance to Muscle (um)")
    ax_proximity.set_ylabel("Distance to Neuron (um)")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return folder_stats

# --- 3. Detection Parameters ---
st.subheader("🎯 Spot Detection (DoG) & Analysis Parameters")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown("**DoG Tunning (BTX)**")
    min_diameter_um = st.number_input("Min Spot Diameter (μm)", value=3.00, step=0.10)
    max_diameter_um = st.number_input("Max Spot Diameter (μm)", value=10.00, step=0.10)
    if max_diameter_um <= min_diameter_um:
        st.error("Max Spot Diameter must be larger than Min Spot Diameter.")
        st.stop()
    auto_threshold = st.checkbox(
        "Auto Threshold per image",
        value=False,
        help="Adapts DoG threshold from each image's BTX signal/noise profile.",
    )
    threshold = st.number_input("Detection Threshold", value=0.05, step=0.01, disabled=auto_threshold)
    
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
    help="Turn off to reduce memory and speed up ALL-folder runs. CSV outputs and master summaries are still generated.",
)

if run_current or run_all:
    all_file_stats = []
    master_rows_written = 0
    
    progress = st.progress(0)
    status = st.empty()
    
    target_dirs = [folder_path] if run_current else [os.path.join(base_dir, d) for d in folders]
    
    all_target_czis = []
    for target_d in target_dirs:
        for f in os.listdir(target_d):
            if f.endswith('.czi'):
                all_target_czis.append((target_d, f))

    if run_all:
        master_csv = os.path.join(".", "ALL_FOLDERS_MASTER_RESULTS.csv")
        master_png = os.path.join(".", "ALL_FOLDERS_SUMMARY.png")
        summary_table_csv = os.path.join(".", "ALL_FOLDERS_SUMMARY_TABLE.csv")
    else:
        master_csv = os.path.join(folder_path, "BATCH_MASTER_RESULTS.csv")
        master_png = os.path.join(folder_path, "BATCH_SUMMARY.png")
        summary_table_csv = None

    # Start fresh for this run so append-mode streaming does not duplicate previous runs.
    if os.path.exists(master_csv):
        os.remove(master_csv)

    for i, (current_d, czi_file) in enumerate(all_target_czis):
        czi_path = os.path.join(current_d, czi_file)
        
        # Determine config logic based on whether we exist natively in the actively mapped UI window
        if current_d == folder_path and czi_file in file_configs:
            conf = file_configs[czi_file]
        else:
            # For folders outside the active UI, try loading saved per-folder configs first
            import json
            other_config_path = os.path.join(current_d, "channel_mapping_config.json")
            loaded_conf = None
            if os.path.exists(other_config_path):
                try:
                    with open(other_config_path, "r") as jf:
                        folder_configs = json.load(jf)
                    if czi_file in folder_configs:
                        fc = folder_configs[czi_file]
                        loaded_conf = {
                            "muscle": fc["m"],
                            "neuron": fc["n"],
                            "btx": fc["b"],
                            "pixel_size": fc.get("p", 1.0),
                            "skip": fc.get("skip", False)
                        }
                except Exception:
                    pass
            
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

            # --- Background Subtraction ---
            # Radius is controlled in physical units (um), then converted per file to pixels.
            btx_bg_um_eff = float(btx_bg_radius_um)
            if btx_bg_um_eff > 0:
                from skimage.morphology import white_tophat, disk
                radius_px, bg_clipped, bg_cap = compute_bg_radius_px(
                    bg_radius_um=btx_bg_um_eff,
                    pixel_size_um=pixel_size,
                    image_shape=img_btx.shape,
                )
                if bg_clipped:
                    st.warning(
                        f"{czi_file}: Background radius clipped to safe limit "
                        f"(~{bg_cap * float(pixel_size):.3g} μm)."
                    )
                # White top-hat is the direct morphological equivalent of FIJI's rolling ball background subtraction
                img_btx = white_tophat(img_btx, disk(radius_px))

            # Normalize BTX for spot detection
            img_btx_norm = rescale_intensity(img_btx, out_range=(0.0, 1.0))
            threshold_used = estimate_auto_threshold(img_btx_norm) if auto_threshold else float(threshold)
            if auto_threshold:
                st.caption(f"{czi_file}: Detection threshold used `{threshold_used:.4f}`")
            blobs, dog_scale, sigma_cap = detect_blobs_stable(
                img_btx_norm=img_btx_norm,
                min_diameter_um=min_diameter_um,
                max_diameter_um=max_diameter_um,
                pixel_size_um=pixel_size,
                threshold=threshold_used,
            )
            if auto_threshold and blobs is not None and len(blobs) == 0:
                # One rescue pass for strict auto thresholds on low-contrast images.
                threshold_retry = max(0.01, float(threshold_used) * 0.8)
                blobs_retry, dog_scale_retry, sigma_cap_retry = detect_blobs_stable(
                    img_btx_norm=img_btx_norm,
                    min_diameter_um=min_diameter_um,
                    max_diameter_um=max_diameter_um,
                    pixel_size_um=pixel_size,
                    threshold=threshold_retry,
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

            total_spots = len(blobs)
            
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
                
                d_m_um = edt_muscle_um[y_idx, x_idx]
                d_n_um = edt_neuron_um[y_idx, x_idx]
                
                distances_m.append(d_m_um)
                distances_n.append(d_n_um)
                
                # --- Morphological & Biological Metrics ---
                circ = 0.0
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
                        th = threshold_otsu(window_btx)
                        labeled = label(window_btx > th)
                        center_y, center_x = y_idx - box_y1, x_idx - box_x1
                        spot_label = labeled[center_y, center_x]
                        
                        # If geometric center fell on a black noise pixel, snap to max bright pixel in crop
                        if spot_label == 0:
                            spot_label = labeled[np.unravel_index(np.argmax(window_btx), window_btx.shape)]
                            
                        if spot_label > 0:
                            spot_mask = (labeled == spot_label)
                            
                            # 1. Circularity
                            props = {p.label: p for p in regionprops(labeled)}
                            prop = props[spot_label]
                            if prop.perimeter > 0:
                                circ = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
                            else:
                                circ = 1.0 # 1 or 2 pixels is basically circular
                                
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
                    "CIRCULARITY": np.clip(circ, 0.0, 1.0),
                    "MEAN_INTENSITY": mean_intensity,
                    "INNERVATION_OVERLAP_PCT": overlap_ratio,
                    "Dist_to_Muscle_um": d_m_um,
                    "Dist_to_Neuron_um": d_n_um,
                    "DETECTION_THRESHOLD_USED": threshold_used,
                    "is_NMJ": (d_m_um <= distance_threshold_um) and (d_n_um <= distance_threshold_um)
                })

            df_spots = pd.DataFrame(spots_data)

            # Outputs
            nmj_count = df_spots['is_NMJ'].sum()
            formation_rate = nmj_count / total_spots * 100
            
            near_m_only = len(df_spots[(df_spots['Dist_to_Muscle_um'] <= distance_threshold_um) & (df_spots['Dist_to_Neuron_um'] > distance_threshold_um)])
            near_n_only = len(df_spots[(df_spots['Dist_to_Neuron_um'] <= distance_threshold_um) & (df_spots['Dist_to_Muscle_um'] > distance_threshold_um)])
            orphaned = len(df_spots[(df_spots['Dist_to_Muscle_um'] > distance_threshold_um) & (df_spots['Dist_to_Neuron_um'] > distance_threshold_um)])
            
            # Fisher's Exact Test to determine if proximity to Neuron is associated with proximity to Muscle
            from scipy.stats import fisher_exact
            _, fisher_p = fisher_exact([[nmj_count, near_m_only], [near_n_only, orphaned]])
            
            # Pre-calculate spatial classifications before saving downstream
            def classify_quadrant(row):
                if row['Dist_to_Muscle_um'] <= distance_threshold_um and row['Dist_to_Neuron_um'] <= distance_threshold_um:
                    return 'NMJ'
                elif row['Dist_to_Muscle_um'] <= distance_threshold_um:
                    return 'Aneural AChR clusters'
                elif row['Dist_to_Neuron_um'] <= distance_threshold_um:
                    return 'Neuron-associated BTX signal'
                else:
                    return 'Orphaned'
            
            df_spots['BTX signal class'] = df_spots.apply(classify_quadrant, axis=1)
            df_spots = normalize_btx_signal_classes(df_spots)

            out_csv = os.path.join(current_d, f"{czi_file.replace('.czi', '')}_analysis.csv")
            df_spots.to_csv(out_csv, index=False)
            
            # Tag source file and stream to master CSV to avoid RAM blow-up on large all-folder runs
            df_spots['SOURCE_IMAGE'] = czi_file
            df_spots['SOURCE_FOLDER'] = os.path.basename(os.path.normpath(current_d))
            cols = df_spots.columns.tolist()
            cols.insert(0, cols.pop(cols.index('SOURCE_IMAGE')))
            cols.insert(0, cols.pop(cols.index('SOURCE_FOLDER')))
            df_spots_master = df_spots.reindex(columns=cols)
            df_spots_master.to_csv(master_csv, mode='a', header=(master_rows_written == 0), index=False)
            master_rows_written += len(df_spots_master)
            
            all_file_stats.append({
                "File": czi_file,
                "Total Spots": total_spots,
                "NMJs (Both)": nmj_count,
                "Near Aneural AChR clusters": near_m_only,
                "Near Neuron-associated BTX signal": near_n_only,
                "Orphaned": orphaned,
                "Formation Rate (%)": formation_rate,
                "Fisher P-Value": fisher_p
            })

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

            # Plot Proximity Graph & Images in a perfectly balanced 3x3 grid (9 spots!)
            fig, axes = plt.subplots(3, 3, figsize=(24, 24))
            
            # Row 1: Graphs (Scatter, Size, Circ)
            ax_scatter = axes[0, 0]
            ax_size_kde = axes[0, 1]
            ax_circ_kde = axes[0, 2]
            
            # Row 2: Graphs (Overlap, Intensity) + 1st Image (Clean BTX)
            ax_overlap_kde = axes[1, 0]
            ax_intensity_kde = axes[1, 1]
            ax_btx_clean = axes[1, 2]
            
            # Row 3: Remaining 3 Images
            ax_btx_marked = axes[2, 0]
            ax_comp_marked = axes[2, 1]
            ax_comp_arrows = axes[2, 2]
            
            
            # Graph 6: Circularity KDE
            if len(df_spots) > 0:
                sns.kdeplot(
                    data=df_spots, x='CIRCULARITY', hue='BTX signal class',
                    hue_order=BTX_SIGNAL_CLASS_ORDER,
                    palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_circ_kde,
                    common_norm=False, fill=True, clip=(0, 1)
                )
            ax_circ_kde.set_title('3. NMJ Circularity KDE')
            ax_circ_kde.set_xlabel('Circularity (1 = Perfect Circle)')
            ax_circ_kde.set_ylabel('Probability Density')
            ax_circ_kde.set_xlim(0, 1)
            
            # Graph 7: Size KDE
            if len(df_spots) > 0:
                sns.kdeplot(
                    data=df_spots, x='RADIUS', hue='BTX signal class',
                    hue_order=BTX_SIGNAL_CLASS_ORDER,
                    palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_size_kde,
                    common_norm=False, fill=True
                )
            ax_size_kde.set_title('2. NMJ Size KDE')
            ax_size_kde.set_xlabel('Radius (μm)')
            ax_size_kde.set_ylabel('Probability Density')
            
            # Graph 8: NMJ Innervation Histogram (Bar Graph)
            if len(df_spots) > 0:
                sns.histplot(
                    data=df_spots, x='INNERVATION_OVERLAP_PCT', hue='BTX signal class',
                    hue_order=BTX_SIGNAL_CLASS_ORDER,
                    palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_overlap_kde,
                    common_norm=False, multiple="layer"
                )
            ax_overlap_kde.set_title('4. NMJ Innervation Distribution')
            ax_overlap_kde.set_xlabel('NMJ Innervation (%)')
            ax_overlap_kde.set_ylabel('Count')
            
            # Graph 9: Mean Intensity KDE
            if len(df_spots) > 0:
                sns.kdeplot(
                    data=df_spots, x='MEAN_INTENSITY', hue='BTX signal class',
                    hue_order=BTX_SIGNAL_CLASS_ORDER,
                    palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_intensity_kde,
                    common_norm=False, fill=True
                )
            ax_intensity_kde.set_title('5. Receptor Intensity KDE')
            ax_intensity_kde.set_xlabel('Mean Fluorescence Intensity')
            ax_intensity_kde.set_ylabel('Probability Density')
            
            # Graph 1: Scatter NMJ
            sns.scatterplot(
                data=df_spots, x='Dist_to_Muscle_um', y='Dist_to_Neuron_um',
                hue='BTX signal class', hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_scatter
            )
            ax_scatter.axvline(x=distance_threshold_um, color='black', linestyle='--')
            ax_scatter.axhline(y=distance_threshold_um, color='black', linestyle='--')
            
            sig_star = "***" if fisher_p < 0.001 else "**" if fisher_p < 0.01 else "*" if fisher_p < 0.05 else "ns"
            ax_scatter.set_title(f'1. NMJ Proximity Analysis (Fisher P = {fisher_p:.4f} {sig_star})')
            ax_scatter.set_xlabel('Distance to Muscle (μm)')
            ax_scatter.set_ylabel('Distance to Neuron (μm)')

            # Graph 2: Raw and cleaned BTX shown side-by-side for subtraction verification
            ax_btx_clean.imshow(raw_clean_side_by_side, cmap='gray', vmin=0.0, vmax=1.0)
            pane_w = img_btx_raw_vis.shape[1]
            ax_btx_clean.axvline(x=pane_w - 0.5, color='yellow', linewidth=2.5)
            ax_btx_clean.set_title("6. Raw BTX (L) | Cleaned BTX (R)")
            ax_btx_clean.axis('off')

            # Graph 3: Strictly scaled cleaned BTX overlaid with spots
            ax_btx_marked.imshow(img_btx_clean_vis, cmap='gray', vmin=0.0, vmax=1.0)
            ax_btx_marked.set_title("7. Cleaned BTX + Detected Spots")
            ax_btx_marked.axis('off')

            # Graph 4: Composite Image + All Spots
            ax_comp_marked.imshow(composite_rgb)
            ax_comp_marked.set_title("8. Composite + All Detected Spots")
            ax_comp_marked.axis('off')
            
            # Graph 5: Composite Image + NMJ Arrows
            ax_comp_arrows.imshow(composite_rgb)
            ax_comp_arrows.set_title("9. Composite + Functional NMJs Only")
            ax_comp_arrows.axis('off')
            
            # Plot the overlays
            for index, blob in enumerate(blobs):
                y, x, r = blob
                c1 = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
                c2 = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
                ax_btx_marked.add_patch(c1)
                ax_comp_marked.add_patch(c2)
                
                # If this spot is functionally classified as an NMJ, point an arrow at it on the final layout
                if df_spots.loc[index, 'is_NMJ']:
                    # The arrow points to the very edge of the radius (x+r, y-r) so it doesn't cover the spot itself.
                    target_x = x + r + 2
                    target_y = y - r - 2
                    # Extended the starting point heavily (80 pixels) so the arrow has a clearly visible long tail on large images
                    start_x = target_x + 80
                    start_y = target_y - 80
                    
                    # "-|>" creates a solid, closed triangle arrowhead instead of an open "v" shape
                    ax_comp_arrows.annotate('', xy=(target_x, target_y), xytext=(start_x, start_y),
                                      arrowprops=dict(arrowstyle="-|>", color='white', lw=1.5))

            # Save visual directly to disk, completely bypassing the Streamlit frontend DOM to prevent DOM payload crash
            out_img = os.path.join(current_d, f"{czi_file.replace('.czi', '')}_NMJ_Plot.png")
            fig.savefig(out_img, bbox_inches='tight')
            plt.close(fig) # Prevent Matplotlib from leaking memory during large batches!
            
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
            fig = None
            axes = None
            # Defensively close any matplotlib figures still tracked by pyplot; this
            # plugs leaks if an exception fired mid-figure construction.
            plt.close('all')
            gc.collect()
            
        progress.progress((i + 1) / len(all_target_czis))
        
    # --- AFTER BATCH COMPLETES ---
    status.write("✅ **Batch Processing Complete!**")
    
    if master_rows_written > 0:
        master_df = normalize_btx_signal_classes(pd.read_csv(master_csv))
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

        # Create summary dashboard.
        # For ALL-folder mode, extend to a 10-panel figure with extra comparative analytics.
        if run_all:
            fig, axes = plt.subplots(5, 2, figsize=(24, 42))
        else:
            fig, axes = plt.subplots(3, 2, figsize=(20, 24))

        ax_scatter = axes[0, 0]
        ax_size_kde = axes[0, 1]
        ax_circ_kde = axes[1, 0]
        ax_overlap_kde = axes[1, 1]
        ax_intensity_kde = axes[2, 0]
        ax_extra = axes[2, 1]

        # 1. NMJ Proximity Scatterplot
        from scipy.stats import fisher_exact
        total_nmj = len(master_df[master_df['BTX signal class'] == 'NMJ'])
        total_m_only = len(master_df[master_df['BTX signal class'] == 'Aneural AChR clusters'])
        total_n_only = len(master_df[master_df['BTX signal class'] == 'Neuron-associated BTX signal'])
        total_orph = len(master_df[master_df['BTX signal class'] == 'Orphaned'])
        _, global_fisher_p = fisher_exact([[total_nmj, total_m_only], [total_n_only, total_orph]])

        sns.scatterplot(
            data=master_df, x='Dist_to_Muscle_um', y='Dist_to_Neuron_um',
            hue='BTX signal class', hue_order=BTX_SIGNAL_CLASS_ORDER,
            palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_scatter
        )
        ax_scatter.axvline(x=distance_threshold_um, color='black', linestyle='--')
        ax_scatter.axhline(y=distance_threshold_um, color='black', linestyle='--')

        sig_star = "***" if global_fisher_p < 0.001 else "**" if global_fisher_p < 0.01 else "*" if global_fisher_p < 0.05 else "ns"
        ax_scatter.set_title(f'1. Global NMJ Proximity Analysis (Fisher P = {global_fisher_p:.4g} {sig_star})')
        ax_scatter.set_xlabel('Distance to Muscle (μm)')
        ax_scatter.set_ylabel('Distance to Neuron (μm)')

        # 2. NMJ Size KDE
        if len(master_df) > 0:
            sns.kdeplot(
                data=master_df, x='RADIUS', hue='BTX signal class',
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_size_kde,
                common_norm=False, fill=True
            )
        ax_size_kde.set_title('2. Global NMJ Size KDE')
        ax_size_kde.set_xlabel('Radius (μm)')
        ax_size_kde.set_ylabel('Probability Density')

        # 3. Circularity KDE
        if len(master_df) > 0:
            sns.kdeplot(
                data=master_df, x='CIRCULARITY', hue='BTX signal class',
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_circ_kde,
                common_norm=False, fill=True, clip=(0, 1)
            )
        ax_circ_kde.set_title('3. Global NMJ Circularity KDE')
        ax_circ_kde.set_xlabel('Circularity (1 = Perfect Circle)')
        ax_circ_kde.set_ylabel('Probability Density')
        ax_circ_kde.set_xlim(0, 1)

        # 4. Innervation Histogram
        if len(master_df) > 0:
            sns.histplot(
                data=master_df, x='INNERVATION_OVERLAP_PCT', hue='BTX signal class',
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_overlap_kde,
                common_norm=False, multiple="layer"
            )
        ax_overlap_kde.set_title('4. Global NMJ Innervation Distribution')
        ax_overlap_kde.set_xlabel('NMJ Innervation (%)')
        ax_overlap_kde.set_ylabel('Count')

        # 5. Mean Intensity KDE
        if len(master_df) > 0:
            sns.kdeplot(
                data=master_df, x='MEAN_INTENSITY', hue='BTX signal class',
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE, ax=ax_intensity_kde,
                common_norm=False, fill=True
            )
        ax_intensity_kde.set_title('5. Global Receptor Intensity KDE')
        ax_intensity_kde.set_xlabel('Mean Fluorescence Intensity')
        ax_intensity_kde.set_ylabel('Probability Density')

        # 6. Additional panel only for ALL-folder runs
        if run_all:
            sns.barplot(
                data=folder_stats_df,
                x='SOURCE_FOLDER',
                y='nmj_rate_pct',
                color='red',
                ax=ax_extra
            )
            ax_extra.set_title('6. NMJ Formation Rate by Folder')
            ax_extra.set_xlabel('Folder')
            ax_extra.set_ylabel('NMJ Rate (%)')
            ax_extra.tick_params(axis='x', rotation=45)
        else:
            ax_extra.axis('off') # Keep same single-folder layout

        if run_all:
            ax_size_overlap = axes[3, 0]
            ax_forest = axes[3, 1]
            ax_corr = axes[4, 0]
            ax_control = axes[4, 1]

            # 7. Spot size vs innervation overlap scatter
            sns.scatterplot(
                data=master_df,
                x='RADIUS',
                y='INNERVATION_OVERLAP_PCT',
                hue='BTX signal class',
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE,
                alpha=0.35,
                s=18,
                ax=ax_size_overlap
            )
            ax_size_overlap.set_title('7. Spot Size vs Innervation Overlap')
            ax_size_overlap.set_xlabel('Radius (μm)')
            ax_size_overlap.set_ylabel('Innervation Overlap (%)')

            # 8. Folder-wise NMJ odds ratio (forest style)
            odds_rows = []
            for _, fs_row in folder_stats_df.iterrows():
                folder_name = fs_row['SOURCE_FOLDER']
                df_f = master_df[master_df['SOURCE_FOLDER'] == folder_name]
                nmj = int((df_f['BTX signal class'] == 'NMJ').sum())
                m_only = int((df_f['BTX signal class'] == 'Aneural AChR clusters').sum())
                n_only = int((df_f['BTX signal class'] == 'Neuron-associated BTX signal').sum())
                orphan = int((df_f['BTX signal class'] == 'Orphaned').sum())

                # Haldane-Anscombe correction for stability when any cell is zero
                a = nmj + 0.5
                b = m_only + 0.5
                c = n_only + 0.5
                d = orphan + 0.5
                odds_ratio = (a * d) / (b * c)
                se_log_or = np.sqrt((1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d))
                log_or = np.log(odds_ratio)
                ci_low = np.exp(log_or - 1.96 * se_log_or)
                ci_high = np.exp(log_or + 1.96 * se_log_or)
                odds_rows.append({
                    'SOURCE_FOLDER': folder_name,
                    'OR': odds_ratio,
                    'CI_LOW': ci_low,
                    'CI_HIGH': ci_high
                })

            odds_df = pd.DataFrame(odds_rows).sort_values('OR')
            y_pos = np.arange(len(odds_df))
            ax_forest.errorbar(
                odds_df['OR'],
                y_pos,
                xerr=[odds_df['OR'] - odds_df['CI_LOW'], odds_df['CI_HIGH'] - odds_df['OR']],
                fmt='o',
                color='black',
                ecolor='black',
                capsize=3
            )
            ax_forest.axvline(1.0, linestyle='--', color='red')
            ax_forest.set_yticks(y_pos)
            ax_forest.set_yticklabels(odds_df['SOURCE_FOLDER'])
            ax_forest.set_xscale('log')
            ax_forest.set_title('8. Folder-wise NMJ Odds Ratio (95% CI)')
            ax_forest.set_xlabel('Odds Ratio (log scale)')
            ax_forest.set_ylabel('Folder')

            # 9. Per-folder correlation heatmap
            corr_features = ['RADIUS', 'CIRCULARITY', 'MEAN_INTENSITY', 'INNERVATION_OVERLAP_PCT', 'Dist_to_Muscle_um', 'Dist_to_Neuron_um']
            corr_records = []
            for folder_name in folder_stats_df['SOURCE_FOLDER']:
                df_f = master_df[master_df['SOURCE_FOLDER'] == folder_name].copy()
                if len(df_f) < 3:
                    continue
                df_f['NMJ_NUM'] = df_f['is_NMJ'].astype(float)
                row = {'SOURCE_FOLDER': folder_name}
                for feat in corr_features:
                    row[feat] = df_f[feat].corr(df_f['NMJ_NUM'])
                corr_records.append(row)
            corr_df = pd.DataFrame(corr_records).set_index('SOURCE_FOLDER') if len(corr_records) > 0 else pd.DataFrame()
            if not corr_df.empty:
                sns.heatmap(
                    corr_df,
                    cmap='coolwarm',
                    center=0.0,
                    vmin=-1.0,
                    vmax=1.0,
                    annot=True,
                    fmt='.2f',
                    linewidths=0.5,
                    ax=ax_corr
                )
            else:
                ax_corr.text(0.5, 0.5, 'Not enough data for\nper-folder correlations', ha='center', va='center')
                ax_corr.set_axis_off()
            ax_corr.set_title('9. Per-Folder Correlation with NMJ State')
            ax_corr.set_xlabel('Feature')
            ax_corr.set_ylabel('Folder')

            # 10. Per-image NMJ rate control chart
            per_image = (
                master_df.groupby(['SOURCE_FOLDER', 'SOURCE_IMAGE'])
                .agg(total_spots=('is_NMJ', 'size'), nmj_spots=('is_NMJ', 'sum'))
                .reset_index()
            )
            per_image['nmj_rate_pct'] = np.where(
                per_image['total_spots'] > 0,
                per_image['nmj_spots'] / per_image['total_spots'] * 100.0,
                0.0
            )
            sns.stripplot(
                data=per_image,
                x='SOURCE_FOLDER',
                y='nmj_rate_pct',
                color='black',
                alpha=0.65,
                jitter=0.25,
                ax=ax_control
            )
            sns.pointplot(
                data=per_image,
                x='SOURCE_FOLDER',
                y='nmj_rate_pct',
                estimator=np.mean,
                errorbar='sd',
                join=False,
                color='red',
                markers='D',
                scale=0.8,
                ax=ax_control
            )
            ax_control.set_title('10. Per-Image NMJ Rate Control Chart')
            ax_control.set_xlabel('Folder')
            ax_control.set_ylabel('NMJ Rate (%)')
            ax_control.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)
        fig.savefig(master_png, bbox_inches='tight')
        plt.close(fig)
        plt.close('all')

        st.success(f"Aggregate Dashboard generated: `{master_png}`")

        # Drop the (potentially huge) aggregate DataFrame and any derived helpers so
        # the script's resident memory shrinks back down before Streamlit re-runs.
        master_df = None
        try:
            del folder_stats_df
        except NameError:
            pass
        try:
            del per_image
        except NameError:
            pass
        try:
            del corr_df
        except NameError:
            pass
        try:
            del odds_df
        except NameError:
            pass
        gc.collect()

    if all_file_stats:
        st.subheader("📊 Batch Summary Metrics")
        st.dataframe(pd.DataFrame(all_file_stats))

