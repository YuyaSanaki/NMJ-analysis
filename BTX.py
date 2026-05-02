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

MIN_PIXELS_FOR_SHAPE = 20
RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL = 0.5

ROUNDNESS_KRUSKAL_CLASSES = ("NMJ", "Aneural AChR clusters", "Neuron-associated BTX signal")


def dataframe_for_roundness_kde_and_kruskal(df):
    """Roundness KDE input: three tissue classes, AREA_PX ≥ MIN_PIXELS_FOR_SHAPE, finite ROUNDNESS."""
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
        return df.iloc[0:0]
    required = {"AREA_PX", "BTX signal class", "ROUNDNESS"}
    if not required <= set(df.columns):
        return df.iloc[0:0]
    return df[
        (df["AREA_PX"] >= MIN_PIXELS_FOR_SHAPE)
        & (df["BTX signal class"].isin(ROUNDNESS_KRUSKAL_CLASSES))
    ].dropna(subset=["ROUNDNESS"])


def normalize_btx_signal_classes(df):
    if df is None or len(df) == 0 or "BTX signal class" not in df.columns:
        return df
    out = df.copy()
    out["BTX signal class"] = (
        out["BTX signal class"].astype(str).str.strip().replace(BTX_SIGNAL_CLASS_LEGACY_ALIASES)
    )
    return out


def proximity_joint_axes(fig, outer_cell, hspace=0.08, wspace=0.08, title_first=False):
    """Proximity scatter + marginal KDEs. If title_first, top row is for the panel title (axes off), then x-KDE, then main+y-KDE."""
    if title_first:
        inner = outer_cell.subgridspec(
            3, 2, height_ratios=[0.28, 1, 4], width_ratios=[4, 1], hspace=hspace, wspace=wspace
        )
        ax_title = fig.add_subplot(inner[0, :])
        ax_title.axis("off")
        ax_kde_x = fig.add_subplot(inner[1, 0])
        ax_corner = fig.add_subplot(inner[1, 1])
        ax_corner.axis("off")
        ax_main = fig.add_subplot(inner[2, 0], sharex=ax_kde_x)
        ax_kde_y = fig.add_subplot(inner[2, 1], sharey=ax_main)
        ax_kde_x.tick_params(labelbottom=False)
        ax_kde_y.tick_params(labelleft=False)
        return ax_main, ax_kde_x, ax_kde_y, ax_title
    inner = outer_cell.subgridspec(
        2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=hspace, wspace=wspace
    )
    ax_kde_x = fig.add_subplot(inner[0, 0])
    ax_corner = fig.add_subplot(inner[0, 1])
    ax_corner.axis("off")
    ax_main = fig.add_subplot(inner[1, 0], sharex=ax_kde_x)
    ax_kde_y = fig.add_subplot(inner[1, 1], sharey=ax_main)
    ax_kde_x.tick_params(labelbottom=False)
    ax_kde_y.tick_params(labelleft=False)
    return ax_main, ax_kde_x, ax_kde_y


def _scatter_dataframe_with_clip_jitter(df, sigma_um=0.02, seed=42):
    """Return a copy of df with tiny deterministic jitter applied ONLY to rows where
    ``Dist_to_Muscle_um`` or ``Dist_to_Neuron_um`` were clipped to 0.

    The clipping in :func:`max(0.0, d_center - r_um)` collapses every spot whose
    center lies inside (or near) an EDT mask onto the same scatter coordinate, so a
    naive ``sns.scatterplot`` renders many overlapping NMJs / clusters as a single
    marker. This jitter only nudges the displayed coordinate (the source ``df`` is
    untouched and the marginal KDEs continue to use the true distances).
    """
    if df is None or len(df) == 0:
        return df
    if "Dist_to_Muscle_um" not in df.columns or "Dist_to_Neuron_um" not in df.columns:
        return df
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    jx = np.abs(rng.normal(0.0, sigma_um, size=n))
    jy = np.abs(rng.normal(0.0, sigma_um, size=n))
    mx = out["Dist_to_Muscle_um"].to_numpy(copy=True).astype(float)
    my = out["Dist_to_Neuron_um"].to_numpy(copy=True).astype(float)
    clipped_x = mx <= 1e-9
    clipped_y = my <= 1e-9
    mx[clipped_x] = mx[clipped_x] + jx[clipped_x]
    my[clipped_y] = my[clipped_y] + jy[clipped_y]
    out["Dist_to_Muscle_um"] = mx
    out["Dist_to_Neuron_um"] = my
    return out


def _mannwhitney_neuron_distance_nmij_vs_aneural(df, min_per_group=3):
    """NMJ vs Aneural ``Dist_to_Neuron_um``, one-sided (NMJ stochastically smaller). Returns dict or None."""
    from scipy.stats import mannwhitneyu

    if df is None or len(df) == 0:
        return None
    if "BTX signal class" not in df.columns or "Dist_to_Neuron_um" not in df.columns:
        return None
    dist_nmj = df[df["BTX signal class"] == "NMJ"]["Dist_to_Neuron_um"].dropna()
    dist_aneural = df[df["BTX signal class"] == "Aneural AChR clusters"]["Dist_to_Neuron_um"].dropna()
    if len(dist_nmj) < min_per_group or len(dist_aneural) < min_per_group:
        return None
    try:
        _, p_val = mannwhitneyu(dist_nmj, dist_aneural, alternative="less")
    except ValueError:
        return None
    return {
        "p_val": float(p_val),
        "med_nmj": float(dist_nmj.median()),
        "med_aneural": float(dist_aneural.median()),
    }


def get_spatial_docking_title(df, label_base="1. Synaptic Docking Precision", n_spots=None, min_per_group=3):
    """
    Synaptic docking precision: compare ``Dist_to_Neuron_um`` (edge distance to neuron) for NMJs
    vs muscle-only (Aneural) clusters. ``alternative='less'`` tests whether NMJs sit closer to
    the neuron channel mask (more precisely docked).
    """
    head = label_base if n_spots is None else f"{label_base} (n={n_spots})"
    if df is None or len(df) == 0:
        return f"{head}\n(No Data)"
    res = _mannwhitney_neuron_distance_nmij_vs_aneural(df, min_per_group=min_per_group)
    if res is None:
        return (
            f"{head}\n(Insufficient clusters for Mann-Whitney; need ≥{min_per_group} NMJ and ≥{min_per_group} Aneural)"
        )
    p_val = res["p_val"]
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    return (
        f"{head}\n(Mann-Whitney P = {p_val:.4g} {sig} | "
        f"NMJ: {res['med_nmj']:.2f} μm vs Aneural: {res['med_aneural']:.2f} μm)"
    )


def spatial_docking_mannwhitneyu_p(df, min_per_group=3):
    """Per-image summary table: p-value or NaN when the test cannot be run."""
    res = _mannwhitney_neuron_distance_nmij_vs_aneural(df, min_per_group=min_per_group)
    return float(res["p_val"]) if res is not None else float("nan")


def draw_proximity_joint(
    ax_main,
    ax_kde_x,
    ax_kde_y,
    df,
    distance_threshold_um,
    title,
    *,
    marginal_alpha=0.35,
    scatter_alpha=0.65,
    scatter_size=None,
    marginal_combined_black=False,
    title_ax=None,
):
    """Scatter with marginal KDEs on muscle (top) and neuron (right). Optionally one black KDE over all spots."""
    if df is not None and len(df) > 0:
        if marginal_combined_black:
            sns.kdeplot(
                data=df,
                x="Dist_to_Muscle_um",
                ax=ax_kde_x,
                color="black",
                fill=True,
                alpha=marginal_alpha,
                warn_singular=False,
            )
            sns.kdeplot(
                data=df,
                y="Dist_to_Neuron_um",
                ax=ax_kde_y,
                color="black",
                fill=True,
                alpha=marginal_alpha,
                warn_singular=False,
            )
        else:
            sns.kdeplot(
                data=df,
                x="Dist_to_Muscle_um",
                hue="BTX signal class",
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE,
                ax=ax_kde_x,
                common_norm=False,
                fill=True,
                alpha=marginal_alpha,
                legend=False,
                warn_singular=False,
            )
            sns.kdeplot(
                data=df,
                y="Dist_to_Neuron_um",
                hue="BTX signal class",
                hue_order=BTX_SIGNAL_CLASS_ORDER,
                palette=BTX_SIGNAL_CLASS_PALETTE,
                ax=ax_kde_y,
                common_norm=False,
                fill=True,
                alpha=marginal_alpha,
                legend=False,
                warn_singular=False,
            )
    df_scatter = _scatter_dataframe_with_clip_jitter(df)
    scatter_kw = dict(
        data=df_scatter,
        x="Dist_to_Muscle_um",
        y="Dist_to_Neuron_um",
        hue="BTX signal class",
        hue_order=BTX_SIGNAL_CLASS_ORDER,
        palette=BTX_SIGNAL_CLASS_PALETTE,
        ax=ax_main,
    )
    if scatter_size is not None:
        scatter_kw["s"] = scatter_size
        scatter_kw["alpha"] = scatter_alpha
    else:
        # Default per-image plots had no alpha, so multiple spots stacked at the same
        # clipped (0,0) coordinate rendered as a single marker — visually under-counting
        # vs. the yellow-circle overlay panels even though the data was complete.
        scatter_kw["alpha"] = 0.7
    sns.scatterplot(**scatter_kw)
    ax_main.axvline(x=distance_threshold_um, color="black", linestyle="--")
    ax_main.axhline(y=distance_threshold_um, color="black", linestyle="--")
    n_spots = int(len(df)) if df is not None else 0
    full_title = get_spatial_docking_title(df, label_base=title, n_spots=n_spots)
    if title_ax is not None:
        title_ax.clear()
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.5,
            full_title,
            transform=title_ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax_main.set_title("")
    else:
        ax_main.set_title(full_title)
    ax_main.set_xlabel("Distance to Muscle — spot edge (μm)")
    ax_main.set_ylabel("Distance to Neuron — spot edge (μm)")
    ax_kde_x.set_xlabel("")
    ax_kde_x.set_ylabel("Density")
    ax_kde_y.set_ylabel("")
    ax_kde_y.set_xlabel("Density")


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


@st.cache_data(show_spinner=False)
def fast_czi_meta(path):
    """Channel count and X pixel size from CZI metadata only (no image planes read)."""
    czi = None
    try:
        czi = aicspylibczi.CziFile(path)
        dims = czi.get_dims_shape()[0]
        cc = dims.get("C", [0, 4])
        num_channels = int(cc[1]) - int(cc[0])

        pixel_size_um = 1.0
        try:
            for dist in czi.meta.findall(".//Distance"):
                if dist.attrib.get("Id") == "X":
                    val = dist.find("Value")
                    if val is not None:
                        pixel_size_um = float(val.text) * 1e6
                        break
        except Exception:
            pass

        shape_yx = None
        y_rng = dims.get("Y")
        x_rng = dims.get("X")
        if y_rng is not None and x_rng is not None and len(y_rng) >= 2 and len(x_rng) >= 2:
            try:
                shape_yx = (int(y_rng[1]) - int(y_rng[0]), int(x_rng[1]) - int(x_rng[0]))
            except Exception:
                pass

        return num_channels, pixel_size_um, shape_yx
    finally:
        if czi is not None and hasattr(czi, "close"):
            try:
                czi.close()
            except Exception:
                pass


def _czi_channel_zmax_2d(czi, c_idx, dims0):
    """Return a single (Y, X) plane as Z-max projection with *one Z plane in RAM at a time*.

    ``read_image(C=...)`` loads the full Z stack for that channel; streaming Z reduces
    peak memory on deep stacks (critical for Docker / Streamlit OOM limits).
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


def load_czi_image(path, channel_indices=None):
    """Load CZI as 2D Z-max planes per channel.

    When ``channel_indices`` is set, only those channels are read from disk (same strategy
    as the batch app), avoiding loading every channel of large multi-channel files into RAM.
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
                pass

        img, _ = czi.read_image()
        img_sq = np.squeeze(img)
        if img_sq.ndim == 4:
            img_sq = np.max(img_sq, axis=1)
        if img_sq.ndim < 3:
            img_sq = np.expand_dims(img_sq, axis=0)
        if img_sq.ndim == 3 and img_sq.shape[-1] <= 10 and img_sq.shape[-1] < img_sq.shape[0]:
            img_sq = np.moveaxis(img_sq, -1, 0)

        if channel_indices is not None:
            wanted = list(dict.fromkeys(int(c) for c in channel_indices))
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


def estimate_auto_threshold(img_btx_norm, sensitivity="Conservative"):
    """
    Compute an auto DoG threshold from a normalised BTX image.

    sensitivity="Conservative"  →  median + 3×MAD, clip [0.02, 0.12]
        Moderate sensitivity; balances reliability and detection rate.
    sensitivity="High"          →  median + 1×MAD, clip [0.02, 0.12]
        Most permissive; detects the faintest spots above local noise.
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

    k = 1.0 if sensitivity == "High" else 3.0
    return float(np.clip(median + k * std_est, 0.02, 0.12))

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
    st.markdown("**DoG Tunning (BTX)**")
    min_diameter_um = st.number_input("Min Spot Diameter (μm)", value=5.00, step=0.10)
    max_diameter_um = st.number_input("Max Spot Diameter (μm)", value=12.00, step=0.10)
    if max_diameter_um <= min_diameter_um:
        st.error("Max Spot Diameter must be larger than Min Spot Diameter.")
        st.stop()
    auto_threshold = st.checkbox(
        "Auto Threshold per image",
        value=False,
        help="Adapts DoG threshold from each image's BTX signal/noise profile.",
    )
    auto_thr_sensitivity = st.radio(
        "Auto Threshold Sensitivity",
        options=["Conservative", "High"],
        index=0,
        horizontal=True,
        disabled=not auto_threshold,
        help="Conservative: median + 3×MAD (balanced). High: median + 1×MAD (most sensitive, higher false-positive risk).",
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
            # Extract only the mapped channels (Z-max per channel; avoids loading all C planes).
            channels = load_czi_image(
                czi_path,
                channel_indices=[muscle_idx, neuron_idx, btx_idx],
            )
            if isinstance(channels, dict):
                img_muscle = channels[muscle_idx]
                img_neuron = channels[neuron_idx]
                img_btx = channels[btx_idx]
            else:
                img_muscle = channels[muscle_idx]
                img_neuron = channels[neuron_idx]
                img_btx = channels[btx_idx]
            img_btx_raw = img_btx.copy()

            # --- Background Subtraction + Robust Normalization ---
            img_btx = remove_muscle_haze(img_btx, pixel_size, btx_bg_radius_um)
            p_high = float(np.percentile(img_btx, 99.9))
            if p_high <= 0:
                p_high = 1e-5
            img_btx_norm = np.clip(img_btx.astype(np.float32, copy=False) / p_high, 0.0, 1.0)
            threshold_used = estimate_auto_threshold(img_btx_norm, sensitivity=auto_thr_sensitivity) if auto_threshold else float(threshold)
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
                    "is_NMJ": (d_m_um <= distance_threshold_um) and (d_n_um <= distance_threshold_um)
                })

            df_spots = pd.DataFrame(spots_data)

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
            df_spots["Resolution_Class"] = np.where(
                float(pixel_size) > RESOLUTION_CLASS_LOWRES_UM_PER_PIXEL,
                "Low-Res",
                "High-Res",
            )

            # Outputs
            nmj_count = df_spots['is_NMJ'].sum()
            formation_rate = nmj_count / total_spots * 100

            near_m_only = len(df_spots[(df_spots['Dist_to_Muscle_um'] <= distance_threshold_um) & (df_spots['Dist_to_Neuron_um'] > distance_threshold_um)])
            near_n_only = len(df_spots[(df_spots['Dist_to_Neuron_um'] <= distance_threshold_um) & (df_spots['Dist_to_Muscle_um'] > distance_threshold_um)])
            orphaned = len(df_spots[(df_spots['Dist_to_Muscle_um'] > distance_threshold_um) & (df_spots['Dist_to_Neuron_um'] > distance_threshold_um)])

            docking_p = spatial_docking_mannwhitneyu_p(df_spots)

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
            if np.isnan(docking_p):
                st.markdown(
                    "- **Synaptic docking precision (Mann–Whitney):** not computed "
                    "(need ≥3 NMJ and ≥3 Aneural clusters in this image)."
                )
            else:
                docking_sig = "***" if docking_p < 0.001 else "**" if docking_p < 0.01 else "*" if docking_p < 0.05 else "ns"
                st.markdown(
                    f"- **Synaptic docking precision (Mann–Whitney, one-sided):** `{docking_p:.4g}` {docking_sig} "
                    "— tests whether NMJs are closer to the neuron signal than muscle-only (aneural) clusters "
                    "(smaller edge distance to neuron mask)."
                )
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
            fig = plt.figure(figsize=(24, 24))
            outer = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

            ax_scatter, ax_prox_kde_x, ax_prox_kde_y, ax_prox_title = proximity_joint_axes(
                fig, outer[0, 0], title_first=True
            )
            ax_size_kde = fig.add_subplot(outer[0, 1])
            ax_circ_kde = fig.add_subplot(outer[0, 2])
            ax_overlap_kde = fig.add_subplot(outer[1, 0])
            ax_intensity_kde = fig.add_subplot(outer[1, 1])
            ax_btx_clean = fig.add_subplot(outer[1, 2])
            ax_btx_marked = fig.add_subplot(outer[2, 0])
            ax_comp_marked = fig.add_subplot(outer[2, 1])
            ax_comp_arrows = fig.add_subplot(outer[2, 2])
            
            
            # Graph 6: Roundness KDE (≥MIN pixels; NMJ / Aneural / Neuron-associated only — matches batch logic)
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
            ax_circ_kde.set_title("3. NMJ Roundness KDE (1 − eccentricity)")
            ax_circ_kde.set_xlabel("Roundness (1 = circle)")
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
            
            # Graph 1: Scatter NMJ + marginal KDEs
            draw_proximity_joint(
                ax_scatter,
                ax_prox_kde_x,
                ax_prox_kde_y,
                df_spots,
                distance_threshold_um,
                "1. NMJ Proximity Analysis",
                marginal_combined_black=True,
                title_ax=ax_prox_title,
            )

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
            fig.savefig(out_img, bbox_inches="tight")
            fig.clf()
            plt.close(fig)

            st.success(f"Files saved: `{out_csv}` and `{out_img}`")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            import traceback
            st.code(traceback.format_exc())

