import os
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

from nmj_master_dashboard import (
    BTX_CLASS_EARLY_NMJ,
    BTX_CLASS_MUSCLE,
    BTX_CLASS_NEURON,
    BTX_CLASS_ORPHANED,
    annotate_global_btx_intensity_otsu,
    collect_image_jobs,
    get_confocal_metadata,
    proximity_analysis_results,
)
from nmj_image_analysis import (
    DOG_SIGMA_RATIO_CONSERVATIVE,
    DOG_SIGMA_RATIO_HIGH,
    analyze_image_for_nmj_plot,
    build_nmj_plot_figure,
    dog_sigma_ratio_from_sensitivity,
)
from nmj_run_output import (
    create_run_output_dir,
    mirror_dataset_output_path,
    render_streamlit_download_section,
    save_run_config_files,
    snapshot_channel_mappings,
)


# Each subdirectory under this path is one dataset folder (contains `.czi` files).
DATA_ROOT = "data"


def format_auto_thr_sensitivity_label(mode: str) -> str:
    if mode == "High":
        return f"High ({DOG_SIGMA_RATIO_HIGH:g})"
    return f"Conservative ({DOG_SIGMA_RATIO_CONSERVATIVE:g})"


st.set_page_config(page_title="NMJ Pipeline", layout="wide")

st.title("🔬 Single-Image NMJ Pipeline (Multi-Format)")
st.markdown("Select a single image file (.czi, .lif, .nd2, .oir, .poir, .tif, .tiff) to automatically detect spots (DoG) and compute distance maps from raw fluorescence data.")

# --- 1. Recursive File Selection ---
os.makedirs(DATA_ROOT, exist_ok=True)
all_jobs = collect_image_jobs([DATA_ROOT])

if not all_jobs:
    st.warning(f"No supported confocal or TIFF images found inside `{DATA_ROOT}/` or its subfolders.")
    st.stop()

# Build relative path options
display_paths = [os.path.relpath(os.path.join(d, f), DATA_ROOT) for d, f in all_jobs]
selected_display_path = st.selectbox("🔬 Select Image File", display_paths)

# Extract directories and paths
selected_idx = display_paths.index(selected_display_path)
folder_path, selected_czi = all_jobs[selected_idx]
czi_path = os.path.join(folder_path, selected_czi)

# --- 2. Shared Multi-Format Loader Mapping ---
fast_czi_meta = get_confocal_metadata

with st.spinner("Reading CZI metadata..."):
    try:
        num_channels, pixel_size_um, shape_yx = fast_czi_meta(czi_path)
        shape_msg = (
            f" Nominal frame (Y×X): {shape_yx[0]}×{shape_yx[1]} pixels."
            if shape_yx is not None
            else ""
        )
        st.success(
            f"Ready — {num_channels} channel(s).{shape_msg} "
            "(Image planes load when you run Process Pipeline; only the three mapped channels are read from disk.)"
        )
    except Exception as e:
        st.error(f"Error reading CZI metadata: {e}")
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
    st.markdown("**DoG Tuning (BTX)**")
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

# --- 4. Run Pipeline ---
if st.button("🚀 Process Pipeline", type="primary"):
    with st.spinner("Processing Images & Computing Distances..."):
        try:
            data_root_abs = os.path.abspath(DATA_ROOT)
            run_dir = create_run_output_dir()
            rel_folder = os.path.relpath(folder_path, data_root_abs)
            channel_snapshot = snapshot_channel_mappings(
                data_root_abs,
                [folder_path],
                active_folder_path=folder_path,
                active_file_configs={
                    selected_czi: {
                        "muscle": muscle_idx,
                        "neuron": neuron_idx,
                        "btx": btx_idx,
                        "pixel_size": float(pixel_size),
                        "skip": False,
                    }
                },
            )
            dog_sigma_ratio = (
                dog_sigma_ratio_from_sensitivity(auto_thr_sensitivity)
                if auto_threshold
                else float(dog_sigma_ratio_manual)
            )
            run_config = {
                "run_id": os.path.basename(run_dir),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "completed_at": None,
                "mode": "single_image",
                "source_folder": rel_folder,
                "source_image": selected_czi,
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

            # Shared detection + analysis core (same code path as batch + PNG regeneration).
            ctx = analyze_image_for_nmj_plot(
                czi_path,
                muscle_idx=muscle_idx,
                neuron_idx=neuron_idx,
                btx_idx=btx_idx,
                pixel_size=float(pixel_size),
                min_diameter_um=float(min_diameter_um),
                max_diameter_um=float(max_diameter_um),
                auto_threshold=bool(auto_threshold),
                threshold=float(threshold) if not auto_threshold else None,
                dog_sigma_ratio=dog_sigma_ratio,
                btx_bg_radius_um=float(btx_bg_radius_um),
                m_thresh_mult=float(m_thresh_mult),
                n_thresh_mult=float(n_thresh_mult),
                distance_threshold_um=float(distance_threshold_um),
            )
            if ctx is None:
                st.error(
                    "No spots detected (or Min Spot Diameter too large for this image scale). "
                    "Try lowering the Detection Threshold or reducing the Min Spot Diameter."
                )
                st.stop()

            df_spots = ctx.df_spots
            total_spots = len(df_spots)
            st.caption(
                f"Detection threshold used: `{ctx.threshold_used:.4f}` — DoG sigma_ratio: `{dog_sigma_ratio}`"
            )
            st.success(f"Detected {total_spots} BTX spots.")

            nmj_count = int(df_spots["is_NMJ"].sum())
            formation_rate = (nmj_count / total_spots * 100) if total_spots else 0.0
            near_m_only = int(len(df_spots[(df_spots["Dist_to_Muscle_um"] <= distance_threshold_um) & (df_spots["Dist_to_Neuron_um"] > distance_threshold_um)]))
            near_n_only = int(len(df_spots[(df_spots["Dist_to_Neuron_um"] <= distance_threshold_um) & (df_spots["Dist_to_Muscle_um"] > distance_threshold_um)]))
            orphaned = int(len(df_spots[(df_spots["Dist_to_Muscle_um"] > distance_threshold_um) & (df_spots["Dist_to_Neuron_um"] > distance_threshold_um)]))

            st.divider()
            st.subheader("📊 Results")
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Total BTX Spots", total_spots)
            sm2.metric(f"{BTX_CLASS_EARLY_NMJ} (≤ {distance_threshold_um} µm)", nmj_count)
            sm3.metric(f"{BTX_CLASS_EARLY_NMJ} Formation Rate", f"{formation_rate:.2f}%")

            st.markdown("### Proximity Statistics")
            st.markdown(f"- **{BTX_CLASS_MUSCLE}:** {near_m_only}")
            st.markdown(f"- **{BTX_CLASS_NEURON}:** {near_n_only}")
            st.markdown(f"- **{BTX_CLASS_ORPHANED} (Far from both):** {orphaned}")

            prox = proximity_analysis_results(df_spots)
            if prox is None:
                st.markdown("- **Proximity tests:** not enough data.")
            else:
                muscle = prox.get("muscle_kruskal")
                neuron = prox.get("neuron_kruskal")
                if muscle is not None:
                    st.markdown(
                        f"- **Distance to muscle (Kruskal–Wallis across classes):** "
                        f"`p={muscle['p_value']:.4g}`"
                    )
                if neuron is not None:
                    st.markdown(
                        f"- **Distance to neuron (Kruskal–Wallis across classes):** "
                        f"`p={neuron['p_value']:.4g}`"
                    )
                if muscle is None and neuron is None:
                    st.markdown(
                        "- **Proximity tests:** need ≥3 spots per class for Kruskal–Wallis."
                    )

            file_stem = os.path.splitext(selected_czi)[0]
            out_csv = mirror_dataset_output_path(
                run_dir, data_root_abs, folder_path, f"{file_stem}_analysis.csv"
            )
            annotate_global_btx_intensity_otsu(df_spots).to_csv(out_csv, index=False)

            fig = build_nmj_plot_figure(ctx)
            st.pyplot(fig)
            out_img = mirror_dataset_output_path(
                run_dir, data_root_abs, folder_path, f"{file_stem}_NMJ_Plot.png"
            )
            fig.savefig(out_img, bbox_inches="tight")
            fig.clf()
            plt.close(fig)

            run_config["completed_at"] = datetime.now().isoformat(timespec="seconds")
            run_config["analysis_csv"] = os.path.relpath(out_csv, run_dir)
            run_config["nmj_plot_png"] = os.path.relpath(out_img, run_dir)
            save_run_config_files(run_dir, run_config, channel_snapshot)
            st.session_state["last_run_dir"] = run_dir

            st.success(f"Files saved: `{out_csv}` and `{out_img}`")
            render_streamlit_download_section(st, run_dir)

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            import traceback
            st.code(traceback.format_exc())
