import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import aicspylibczi
from skimage.feature import blob_dog
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage import distance_transform_edt, zoom
from skimage.exposure import rescale_intensity

BTX_SIGNAL_CLASS_ORDER = ("NMJ", "Aneural AChR clusters", "Neuron-associated BTX signal", "Orphaned")
BTX_SIGNAL_CLASS_PALETTE = {
    "NMJ": "red",
    "Aneural AChR clusters": "green",
    "Neuron-associated BTX signal": "blue",
    "Orphaned": "gray",
}

BTX_SIGNAL_CLASS_LEGACY_ALIASES = {
    "Muscle Only": "Aneural AChR clusters",
    "Muscle only": "Aneural AChR clusters",
    "Neuron Only": "Neuron-associated BTX signal",
    "Neuron only": "Neuron-associated BTX signal",
}


def normalize_btx_signal_classes(df):
    if df is None or len(df) == 0 or "BTX signal class" not in df.columns:
        return df
    out = df.copy()
    out["BTX signal class"] = (
        out["BTX signal class"].astype(str).str.strip().replace(BTX_SIGNAL_CLASS_LEGACY_ALIASES)
    )
    return out


st.set_page_config(page_title="NMJ Pipeline", layout="wide")

st.title("🔬 Single-Image NMJ Pipeline (CZI)")
st.markdown("Select a single `.czi` file to automatically detect spots (DoG) and compute distance maps from raw fluorescence data.")

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

selected_czi = st.selectbox("🔬 Select CZI File", czi_files)
czi_path = os.path.join(folder_path, selected_czi)

# --- 2. Load CZI and Configure Channels ---


def _czi_channel_zmax_2d(czi, c_idx, dims0):
    """Z-max to (Y, X) with at most one Z-plane in memory (see BTX_batch.load_czi_image)."""
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


@st.cache_data(show_spinner=False)
def load_czi_image(path):
    czi = None
    try:
        czi = aicspylibczi.CziFile(path)
        dims0 = czi.get_dims_shape()[0]
        c_rng = dims0.get("C", (0, 1))
        c0, c1 = int(c_rng[0]), int(c_rng[1])
        planes = [_czi_channel_zmax_2d(czi, c_idx, dims0) for c_idx in range(c0, c1)]
        img_sq = np.stack(planes, axis=0)
        if img_sq.ndim < 3:
            img_sq = np.expand_dims(img_sq, axis=0)
        if img_sq.ndim == 3 and img_sq.shape[-1] <= 10 and img_sq.shape[-1] < img_sq.shape[0]:
            img_sq = np.moveaxis(img_sq, -1, 0)

        # --- Extract Pixel Size (Microns) ---
        pixel_size_um = 1.0  # Default fallback
        try:
            # Zeiss stores scaling in elements like <Distance Id="X"><Value>1.03e-07</Value>
            for dist in czi.meta.findall('.//Distance'):
                if dist.attrib.get('Id') == 'X':
                    val = dist.find('Value')
                    if val is not None:
                        # Values are saved natively in pure meters. Convert to micrometers.
                        pixel_size_um = float(val.text) * 1e6
                        break
        except Exception:
            pass

        return img_sq, pixel_size_um
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


def remove_muscle_haze(img, pixel_size_um):
    """
    Subtract broad BTX background so 3-10 um puncta remain.
    Uses a fixed 30 um Gaussian sigma as a wide haze model.
    """
    pixel_size_safe = max(float(pixel_size_um), 1e-9)
    bg_sigma_um = 30.0
    bg_sigma_px = bg_sigma_um / pixel_size_safe
    background = gaussian(img, sigma=bg_sigma_px, preserve_range=True)
    result = img.astype(np.float32, copy=False) - background.astype(np.float32, copy=False)
    return np.clip(result, 0.0, None).astype(img.dtype, copy=False)


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
    """
    Robustly estimate a DoG threshold for NMJ BTX signals.
    Uses median + K*mad_k7_clip014 so sparse dim spots are less affected by bright clusters.
    """
    sample = np.asarray(img_btx_norm, dtype=np.float32)[::4, ::4].ravel()
    if sample.size == 0:
        return 0.05

    # After haze subtraction + clipping, many zeros are background floor. Compute
    # robust stats on positive pixels so mad_k7_clip014 does not collapse to near-zero.
    pos = sample[sample > 0]
    if pos.size == 0:
        return 0.05

    median = float(np.median(pos))
    mad = float(np.median(np.abs(pos - median)))
    std_est = 1.4826 * mad
    auto_thr = median + (7.0 * std_est)
    return float(np.clip(auto_thr, 0.01, 0.14))

with st.spinner("Loading CZI..."):
    try:
        image_data, pixel_size_um = load_czi_image(czi_path)
        num_channels = image_data.shape[0]
        st.success(f"Loaded successfully! Detected {num_channels} channels. Shape: {image_data.shape[1:]}")
    except Exception as e:
        st.error(f"Error loading CZI: {e}")
        st.stop()


# Channel mapping UI
st.subheader("⚙️ Channel Mapping")
channel_options = [f"Channel {i+1}" for i in range(num_channels)]

c1, c2, c3, c4 = st.columns(4)
with c1:
     muscle_idx = st.selectbox("Muscle Channel", range(num_channels), format_func=lambda i: channel_options[i], index=0 if num_channels > 0 else 0)
with c2:
     neuron_idx = st.selectbox("Neuron Channel", range(num_channels), format_func=lambda i: channel_options[i], index=1 if num_channels > 1 else 0)
with c3:
     btx_idx = st.selectbox("BTX (Receptors) Channel", range(num_channels), format_func=lambda i: channel_options[i], index=3 if num_channels > 3 else min(num_channels-1, 2))
with c4:
     pixel_size = st.number_input("Pixel Size (um/pixel)", value=float(pixel_size_um), format="%0.7f", help="Automatically pulled from CZI metadata. Change manually if needed.")

st.divider()

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

# --- 4. Run Pipeline ---
if st.button("🚀 Process Pipeline", type="primary"):
    with st.spinner("Processing Images & Computing Distances..."):
        try:
            # Extract channels
            img_muscle = image_data[muscle_idx]
            img_neuron = image_data[neuron_idx]
            img_btx = image_data[btx_idx]
            img_btx_raw = img_btx.copy()

            # --- Background Subtraction + Robust Normalization ---
            img_btx = remove_muscle_haze(img_btx, pixel_size)
            p_high = float(np.percentile(img_btx, 99.9))
            if p_high <= 0:
                p_high = 1e-5
            img_btx_norm = np.clip(img_btx.astype(np.float32, copy=False) / p_high, 0.0, 1.0)
            threshold_used = estimate_auto_threshold(img_btx_norm) if auto_threshold else float(threshold)
            st.caption(f"Detection threshold used: `{threshold_used:.4f}`")

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
                    st.caption(f"Auto-threshold rescue retry used: `{threshold_used:.4f}`")
            if blobs is None:
                st.error(
                    f"Min Spot Diameter is too large for this image scale. "
                    f"Please reduce it below approximately {2.0 * np.sqrt(2.0) * sigma_cap * float(pixel_size):.3g} μm."
                )
                st.stop()
            if dog_scale < 1.0:
                st.warning(
                    f"Large spot-size mode enabled: detection ran on a {dog_scale:.2f}x scaled image "
                    f"for stability, then mapped back to original coordinates."
                )
            
            # --- DoG Spot Detection ---
            st.info("Detecting BTX Spots...")
            
            total_spots = len(blobs)
            if total_spots == 0:
                st.error("No spots found! Try lowering the Detection Threshold.")
                st.stop()
            
            st.success(f"TrackMate Replacement: Detected {total_spots} Spots!")

            # --- Auto EDT Generation ---
            st.info("Generating Auto Distance Maps (EDT)...")
            # Compute Otsu directly on the raw intensity images
            m_thresh = threshold_otsu(img_muscle) * m_thresh_mult
            n_thresh = threshold_otsu(img_neuron) * n_thresh_mult
            
            muscle_mask = img_muscle > m_thresh
            neuron_mask = img_neuron > n_thresh
            
            # Distance Transform (outputs in raw pixels)
            edt_muscle_px = distance_transform_edt(muscle_mask == 0)
            edt_neuron_px = distance_transform_edt(neuron_mask == 0)
            
            # Convert direct arrays into physical Micrometers using CZI metadata
            edt_muscle_um = edt_muscle_px * pixel_size
            edt_neuron_um = edt_neuron_px * pixel_size

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
                            perimeter = getattr(prop, "perimeter_crofton", 0.0)
                            if perimeter > 0:
                                circ = (4 * np.pi * prop.area) / (perimeter ** 2)
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
            
            from scipy.stats import fisher_exact
            _, fisher_p = fisher_exact([[nmj_count, near_m_only], [near_n_only, orphaned]])

            # --- Visualisation ---
            st.divider()
            st.subheader("📊 Results")
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Total BTX Spots", total_spots)
            sm2.metric(f"NMJs (≤ {distance_threshold_um} µm)", nmj_count)
            sm3.metric("NMJ Formation Rate", f"{formation_rate:.2f}%")
            
            st.markdown("### Proximity Statistics")
            st.markdown(f"- **Near Aneural AChR clusters:** {near_m_only}")
            st.markdown(f"- **Near Neuron-associated BTX signal:** {near_n_only}")
            st.markdown(f"- **Orphaned (Far from both):** {orphaned}")
            sig_star = "***" if fisher_p < 0.001 else "**" if fisher_p < 0.01 else "*" if fisher_p < 0.05 else "ns"
            st.markdown(f"- **Fisher's Exact P-Value:** `{fisher_p:.4g}` {sig_star} *(Measures if spot recruitment to Muscle is statistically associated with recruitment to Neuron)*")

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

            # Save CSV
            out_csv = os.path.join(folder_path, f"{selected_czi.replace('.czi', '')}_analysis.csv")
            df_spots.to_csv(out_csv, index=False)

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

            st.pyplot(fig)
            
            # Save visual
            out_img = os.path.join(folder_path, f"{selected_czi.replace('.czi', '')}_NMJ_Plot.png")
            fig.savefig(out_img, bbox_inches='tight')
            plt.close(fig)

            st.success(f"Files saved: `{out_csv}` and `{out_img}`")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            import traceback
            st.code(traceback.format_exc())

