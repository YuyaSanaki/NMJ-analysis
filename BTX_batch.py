import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import aicspylibczi
from skimage.feature import blob_dog
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
from skimage.exposure import rescale_intensity

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
    czi = aicspylibczi.CziFile(path)
    dims = czi.get_dims_shape()[0]
    cc = dims.get('C', [0, 4])
    num_channels = cc[1] - cc[0]
    
    pixel_size_um = 1.0 # Default fallback
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

@st.cache_data(show_spinner=False)
def load_czi_image(path):
    czi = aicspylibczi.CziFile(path)
    img, _ = czi.read_image()
    img_sq = np.squeeze(img)
    if img_sq.ndim == 4:
        img_sq = np.max(img_sq, axis=1)
    if img_sq.ndim < 3:
        img_sq = np.expand_dims(img_sq, axis=0)
    if img_sq.ndim == 3 and img_sq.shape[0] > img_sq.shape[-1]:
        img_sq = np.moveaxis(img_sq, -1, 0)
    return img_sq

# --- 3. Detection Parameters ---
st.subheader("🎯 Spot Detection (DoG) & Analysis Parameters")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown("**DoG Tunning (BTX)**")
    min_sigma = st.number_input("Min Spot Size (Sigma)", value=3.0, step=0.5)
    max_sigma = st.number_input("Max Spot Size (Sigma)", value=5.0, step=0.5)
    threshold = st.number_input("Detection Threshold", value=0.05, step=0.01)
    
    auto_bg = st.checkbox("Auto-Optimize Background Radius", value=True, help="Automatically sets the background filter size to safely cover your largest spots without erasing them.")
    if not auto_bg:
        btx_bg_radius = st.number_input("Manual Background Radius", value=1.0, step=0.5)
    else:
        # Golden rule for morphological subtraction: Radius should be slightly larger than the largest actual objects.
        btx_bg_radius = max_sigma * 2.0

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
run_all = col_run2.button("🚀 Run Batch Analysis (ALL Folders)", type="secondary", help="Executes natively on every tracked folder using your active Global Template Mapping")

if run_current or run_all:
    all_spots_data = []
    all_file_stats = []
    
    progress = st.progress(0)
    status = st.empty()
    
    target_dirs = [folder_path] if run_current else [os.path.join(base_dir, d) for d in folders]
    
    all_target_czis = []
    for target_d in target_dirs:
        for f in os.listdir(target_d):
            if f.endswith('.czi'):
                all_target_czis.append((target_d, f))

    for i, (current_d, czi_file) in enumerate(all_target_czis):
        czi_path = os.path.join(current_d, czi_file)
        
        # Determine config logic based on whether we exist natively in the actively mapped UI window
        if current_d == folder_path and czi_file in file_configs:
            conf = file_configs[czi_file]
        else:
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
            # Extract channels
            image_data = load_czi_image(czi_path)
            img_muscle = image_data[conf['muscle']]
            img_neuron = image_data[conf['neuron']]
            img_btx = image_data[conf['btx']]

            # --- Background Subtraction ---
            if btx_bg_radius > 0:
                from skimage.morphology import white_tophat, disk
                radius_px = max(1, int(round(btx_bg_radius)))
                # White top-hat is the direct morphological equivalent of FIJI's rolling ball background subtraction
                img_btx = white_tophat(img_btx, disk(radius_px))

            # Normalize BTX for spot detection
            img_btx_norm = rescale_intensity(img_btx, out_range=(0.0, 1.0))
            
            # --- DoG Spot Detection ---
            blobs = blob_dog(img_btx_norm, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
            
            if len(blobs) == 0:
                continue # Skip file if absolutely no spots found
                
            blobs[:, 2] = blobs[:, 2] * np.sqrt(2) # compute radius
            total_spots = len(blobs)
            
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
                            props = regionprops(labeled)
                            prop = props[spot_label - 1]
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
            _, fisher_p = fisher_exact([[nmj_count, near_n_only], [near_m_only, orphaned]])
            
            out_csv = os.path.join(current_d, f"{czi_file.replace('.czi', '')}_analysis.csv")
            df_spots.to_csv(out_csv, index=False)
            
            # Tag source file and push to master payload
            df_spots['SOURCE_IMAGE'] = czi_file
            df_spots['SOURCE_FOLDER'] = os.path.basename(os.path.normpath(current_d))
            all_spots_data.extend(df_spots.to_dict('records'))
            
            all_file_stats.append({
                "File": czi_file,
                "Total Spots": total_spots,
                "NMJs (Both)": nmj_count,
                "Near Muscle Only": near_m_only,
                "Near Neuron Only": near_n_only,
                "Orphaned": orphaned,
                "Formation Rate (%)": formation_rate,
                "Fisher P-Value": fisher_p
            })

            # Normalize images for composite display using percentiles (Auto Contrast)
            def auto_contrast(img):
                p_low, p_high = np.percentile(img, (5, 99.5))
                return rescale_intensity(img, in_range=(p_low, p_high), out_range=(0.0, 1.0))

            img_m_norm = auto_contrast(img_muscle)
            img_n_norm = auto_contrast(img_neuron)
            img_b_norm = auto_contrast(img_btx)
            
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
                    data=df_spots, x='CIRCULARITY', hue='is_NMJ',
                    palette={True: 'red', False: 'gray'}, ax=ax_circ_kde,
                    common_norm=False, fill=True, clip=(0, 1)
                )
            ax_circ_kde.set_title('5. NMJ Circularity KDE')
            ax_circ_kde.set_xlabel('Circularity (1 = Perfect Circle)')
            ax_circ_kde.set_ylabel('Probability Density')
            ax_circ_kde.set_xlim(0, 1)
            
            # Graph 7: Size KDE
            if len(df_spots) > 0:
                sns.kdeplot(
                    data=df_spots, x='RADIUS', hue='is_NMJ',
                    palette={True: 'red', False: 'gray'}, ax=ax_size_kde,
                    common_norm=False, fill=True
                )
            ax_size_kde.set_title('6. NMJ Size KDE')
            ax_size_kde.set_xlabel('Radius (μm)')
            ax_size_kde.set_ylabel('Probability Density')
            
            # Graph 8: NMJ Innervation Histogram (Bar Graph)
            if len(df_spots) > 0:
                sns.histplot(
                    data=df_spots, x='INNERVATION_OVERLAP_PCT', hue='is_NMJ',
                    palette={True: 'red', False: 'gray'}, ax=ax_overlap_kde,
                    common_norm=False, multiple="layer"
                )
            ax_overlap_kde.set_title('4. NMJ Innervation Distribution')
            ax_overlap_kde.set_xlabel('NMJ Innervation (%)')
            ax_overlap_kde.set_ylabel('Count')
            
            # Graph 9: Mean Intensity KDE
            if len(df_spots) > 0:
                sns.kdeplot(
                    data=df_spots, x='MEAN_INTENSITY', hue='is_NMJ',
                    palette={True: 'red', False: 'gray'}, ax=ax_intensity_kde,
                    common_norm=False, fill=True
                )
            ax_intensity_kde.set_title('5. Receptor Intensity KDE')
            ax_intensity_kde.set_xlabel('Mean Fluorescence Intensity')
            ax_intensity_kde.set_ylabel('Probability Density')
            
            # Graph 1: Scatter NMJ
            def classify_quadrant(row):
                if row['Dist_to_Muscle_um'] <= distance_threshold_um and row['Dist_to_Neuron_um'] <= distance_threshold_um:
                    return 'NMJ'
                elif row['Dist_to_Muscle_um'] <= distance_threshold_um:
                    return 'Muscle Only'
                elif row['Dist_to_Neuron_um'] <= distance_threshold_um:
                    return 'Neuron Only'
                else:
                    return 'Orphaned'
            
            df_spots['BTX signal class'] = df_spots.apply(classify_quadrant, axis=1)
            
            sns.scatterplot(
                data=df_spots, x='Dist_to_Muscle_um', y='Dist_to_Neuron_um',
                hue='BTX signal class', palette={'NMJ': 'red', 'Muscle Only': 'green', 'Neuron Only': 'blue', 'Orphaned': 'gray'}, ax=ax_scatter
            )
            ax_scatter.axvline(x=distance_threshold_um, color='black', linestyle='--')
            ax_scatter.axhline(y=distance_threshold_um, color='black', linestyle='--')
            
            sig_star = "***" if fisher_p < 0.001 else "**" if fisher_p < 0.01 else "*" if fisher_p < 0.05 else "ns"
            ax_scatter.set_title(f'NMJ Proximity Analysis (Fisher P = {fisher_p:.4f} {sig_star})')
            ax_scatter.set_xlabel('Distance to Muscle (μm)')
            ax_scatter.set_ylabel('Distance to Neuron (μm)')

            # Graph 2: PURE Cleaned BTX (No Marks)
            ax_btx_clean.imshow(img_btx, cmap='gray', vmax=np.percentile(img_btx, 99.5))
            ax_btx_clean.set_title("1. BTX Channel (Background Subtracted)")
            ax_btx_clean.axis('off')

            # Graph 3: Original BTX overlaid with Spots
            ax_btx_marked.imshow(img_btx, cmap='gray', vmax=np.percentile(img_btx, 99.5))
            ax_btx_marked.set_title("2. BTX Channel + Detected Spots")
            ax_btx_marked.axis('off')

            # Graph 4: Composite Image + All Spots
            ax_comp_marked.imshow(composite_rgb)
            ax_comp_marked.set_title("3. Composite + All Detected Spots")
            ax_comp_marked.axis('off')
            
            # Graph 5: Composite Image + NMJ Arrows
            ax_comp_arrows.imshow(composite_rgb)
            ax_comp_arrows.set_title("4. Composite + Functional NMJs Only")
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
            out_img = os.path.join(current_d, f"{czi_file.replace('.czi', '')}_NMJ_Plot.png")
            fig.savefig(out_img, bbox_inches='tight')
            plt.close(fig) # Prevent Matplotlib from leaking memory during large batches!
            
        except Exception as e:
            st.warning(f"Analysis failed organically on {czi_file}: {e}")
            
        progress.progress((i + 1) / len(all_target_czis))
        
    # --- AFTER BATCH COMPLETES ---
    status.write("✅ **Batch Processing Complete!**")
    
    if all_spots_data:
        master_df = pd.DataFrame(all_spots_data)
        # Move SOURCE_IMAGE and SOURCE_FOLDER columns to the front
        cols = master_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('SOURCE_IMAGE')))
        cols.insert(0, cols.pop(cols.index('SOURCE_FOLDER')))
        master_df = master_df.reindex(columns=cols)
        
        if run_all:
            master_csv = os.path.join(".", "ALL_FOLDERS_MASTER_RESULTS.csv")
            master_png = os.path.join(".", "ALL_FOLDERS_SUMMARY.png")
        else:
            master_csv = os.path.join(folder_path, "BATCH_MASTER_RESULTS.csv")
            master_png = os.path.join(folder_path, "BATCH_SUMMARY.png")
            
        master_df.to_csv(master_csv, index=False)
        st.success(f"Aggregate Master dataset uniquely saved: `{master_csv}`")
        
        st.subheader("📈 Batch Statistical Summary")
        
        # Create a massive 5-panel master figure
        fig, axes = plt.subplots(3, 2, figsize=(24, 28))
        
        # 1. NMJ Proximity (Bar plot of BTX Signal Class Proportions)
        ax_class = axes[0, 0]
        class_counts = master_df.groupby(['SOURCE_IMAGE', 'BTX signal class']).size().unstack(fill_value=0)
        class_props = class_counts.div(class_counts.sum(axis=1), axis=0) * 100
        target_cols = [c for c in ['NMJ', 'Muscle Only', 'Neuron Only', 'Orphaned'] if c in class_props.columns]
        class_props[target_cols].plot(kind='bar', stacked=True, ax=ax_class, 
                            color={'NMJ':'red', 'Muscle Only':'green', 'Neuron Only':'blue', 'Orphaned':'gray'})
        ax_class.set_title("1. BTX Signal Class Proportional Distribution")
        ax_class.set_ylabel("Percentage (%)")
        ax_class.tick_params(axis='x', rotation=90)
        
        # We exclusively analyze functional NMJs for specific biological metrics to prevent noisy background/orphaned spots from skewing true junction data.
        df_nmjs = master_df[master_df['is_NMJ'] == True]
        
        # 2. NMJ Size
        ax_size = axes[0, 1]
        if len(df_nmjs) > 0:
            sns.violinplot(data=df_nmjs, x='SOURCE_IMAGE', y='RADIUS', ax=ax_size, hue='SOURCE_FOLDER', inner='quartile')
        ax_size.set_title("2. Functional NMJ Size (Radius μm)")
        ax_size.tick_params(axis='x', rotation=90)
        
        # 3. Circularity
        ax_circ = axes[1, 0]
        if len(df_nmjs) > 0:
            sns.violinplot(data=df_nmjs, x='SOURCE_IMAGE', y='CIRCULARITY', ax=ax_circ, hue='SOURCE_FOLDER', inner='quartile')
        ax_circ.set_title("3. Functional NMJ Circularity")
        ax_circ.set_ylim(0, 1)
        ax_circ.tick_params(axis='x', rotation=90)
        
        # 4. Innervation Overlap
        ax_innerv = axes[1, 1]
        if len(df_nmjs) > 0:
            sns.violinplot(data=df_nmjs, x='SOURCE_IMAGE', y='INNERVATION_OVERLAP_PCT', ax=ax_innerv, hue='SOURCE_FOLDER', inner='quartile')
        ax_innerv.set_title("4. Functional NMJ Innervation (%)")
        ax_innerv.set_ylim(-10, 110)
        ax_innerv.tick_params(axis='x', rotation=90)
        
        # 5. Intensity
        ax_int = axes[2, 0]
        if len(df_nmjs) > 0:
            sns.violinplot(data=df_nmjs, x='SOURCE_IMAGE', y='MEAN_INTENSITY', ax=ax_int, hue='SOURCE_FOLDER', inner='quartile')
        ax_int.set_title("5. Functional NMJ Receptor Intensity")
        ax_int.tick_params(axis='x', rotation=90)
        
        # Format axes
        axes[2, 1].axis('off') # Hide empty 6th panel
        
        plt.tight_layout()
        st.pyplot(fig)
        fig.savefig(master_png, bbox_inches='tight')
        plt.close(fig)
        
        st.success(f"Aggregate Dashboard generated: `{master_png}`")
        
    if all_file_stats:
        st.subheader("📊 Batch Summary Metrics")
        st.dataframe(pd.DataFrame(all_file_stats))

