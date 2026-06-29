"""Timestamped run folders under ``output/`` for analysis artifacts (not raw CZI inputs)."""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime
from typing import Any, Iterator

OUTPUT_ROOT = "output"
CHANNEL_MAPPING_CONFIG_FILENAME = "channel_mapping_config.json"


def channel_config_state_key(folder_rel: str, czi_file: str, field: str) -> str:
    """Streamlit session_state key scoped to one dataset folder and image file."""
    safe_folder = folder_rel.replace("/", "__").replace("\\", "__")
    return f"cfg_{field}__{safe_folder}__{czi_file}"


def read_channel_mapping_config(folder_path: str) -> dict[str, Any] | None:
    cfg_path = os.path.join(folder_path, CHANNEL_MAPPING_CONFIG_FILENAME)
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as jf:
            loaded = json.load(jf)
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def apply_channel_mapping_to_session_keys(
    folder_rel: str,
    mapping: dict[str, Any],
    czi_files: list[str],
) -> dict[str, Any]:
    """Build session_state key/value pairs from a folder's channel_mapping_config.json."""
    updates: dict[str, Any] = {}
    for cf in czi_files:
        if cf not in mapping:
            continue
        fc = mapping[cf]
        updates[channel_config_state_key(folder_rel, cf, "m")] = fc["m"]
        updates[channel_config_state_key(folder_rel, cf, "n")] = fc["n"]
        updates[channel_config_state_key(folder_rel, cf, "b")] = fc["b"]
        updates[channel_config_state_key(folder_rel, cf, "p")] = fc.get("p", 1.0)
        updates[channel_config_state_key(folder_rel, cf, "skip")] = fc.get("skip", False)
    return updates


def export_channel_mapping_from_session(
    folder_rel: str,
    czi_files: list[str],
    session_values: dict[str, Any],
) -> dict[str, Any]:
    """Serialize in-memory channel mapping for one folder to JSON-ready dict."""
    export_data: dict[str, Any] = {}
    for fname in czi_files:
        key_m = channel_config_state_key(folder_rel, fname, "m")
        if key_m not in session_values:
            continue
        export_data[fname] = {
            "m": session_values.get(channel_config_state_key(folder_rel, fname, "m"), 0),
            "n": session_values.get(channel_config_state_key(folder_rel, fname, "n"), 0),
            "b": session_values.get(channel_config_state_key(folder_rel, fname, "b"), 0),
            "p": session_values.get(channel_config_state_key(folder_rel, fname, "p"), 1.0),
            "skip": session_values.get(channel_config_state_key(folder_rel, fname, "skip"), False),
        }
    return export_data


def resolve_channel_mapping_for_folder(
    folder_rel: str,
    *,
    disk_mapping: dict[str, Any] | None,
    cache_mapping: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer in-session cache over on-disk JSON when re-entering a dataset folder."""
    if cache_mapping:
        return cache_mapping
    return disk_mapping


def default_channel_config_for_file(*, n_ch: int, pixel_size_um: float) -> dict[str, Any]:
    n_ch_safe = max(int(n_ch), 1)
    return {
        "m": 0,
        "n": min(1, n_ch_safe - 1),
        "b": min(3, n_ch_safe - 1),
        "p": float(pixel_size_um),
        "skip": False,
    }


def create_run_output_dir(*, root: str = OUTPUT_ROOT, when: datetime | None = None) -> str:
    when = when or datetime.now()
    run_name = when.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(root, run_name))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def mirror_dataset_output_path(
    run_dir: str, data_root: str, dataset_dir: str, filename: str
) -> str:
    """Place a file under ``run_dir`` mirroring its dataset folder relative to ``data_root``."""
    data_root = os.path.normpath(os.path.abspath(data_root))
    dataset_dir = os.path.normpath(os.path.abspath(dataset_dir))
    rel_parent = os.path.relpath(dataset_dir, data_root)
    if rel_parent.startswith(".."):
        rel_parent = os.path.basename(dataset_dir)
    out_dir = os.path.join(run_dir, rel_parent)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def snapshot_channel_mappings(
    data_root: str,
    target_dirs: list[str],
    *,
    active_folder_path: str | None = None,
    active_file_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Copy channel maps from ``data/`` plus any in-memory UI overrides for the active folder."""
    data_root = os.path.normpath(os.path.abspath(data_root))
    active_abs = os.path.normpath(os.path.abspath(active_folder_path)) if active_folder_path else None
    snapshot: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    for target_d in target_dirs:
        td = os.path.normpath(os.path.abspath(target_d))
        if td in seen:
            continue
        seen.add(td)
        try:
            rel_folder = os.path.relpath(td, data_root)
        except ValueError:
            rel_folder = os.path.basename(td)

        folder_map: dict[str, Any] = {}
        loaded = read_channel_mapping_config(td)
        if loaded:
            folder_map = loaded

        if active_abs and td == active_abs and active_file_configs:
            for fname, conf in active_file_configs.items():
                folder_map[fname] = {
                    "m": conf["muscle"],
                    "n": conf["neuron"],
                    "b": conf["btx"],
                    "p": conf.get("pixel_size", 1.0),
                    "skip": conf.get("skip", False),
                }

        if folder_map:
            snapshot[rel_folder] = folder_map

    return snapshot


def save_run_config_files(
    run_dir: str,
    run_config: dict[str, Any],
    channel_mapping_snapshot: dict[str, dict[str, Any]],
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as jf:
        json.dump(run_config, jf, indent=2)
    with open(os.path.join(run_dir, "channel_mapping_config.json"), "w", encoding="utf-8") as jf:
        json.dump(channel_mapping_snapshot, jf, indent=2)


def iter_run_files(run_dir: str) -> Iterator[tuple[str, str]]:
    run_dir = os.path.abspath(run_dir)
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        for name in sorted(filenames):
            abspath = os.path.join(dirpath, name)
            rel = os.path.relpath(abspath, run_dir)
            yield rel, abspath


def build_run_zip_bytes(run_dir: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel, abspath in iter_run_files(run_dir):
            zf.write(abspath, arcname=rel)
    buf.seek(0)
    return buf.getvalue()


def _guess_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".png": "image/png",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")


def render_streamlit_download_section(st, run_dir: str, *, title: str = "📥 Download outputs") -> None:
    """Streamlit download buttons for an entire run folder."""
    if not run_dir or not os.path.isdir(run_dir):
        return

    files = list(iter_run_files(run_dir))
    if not files:
        return

    st.subheader(title)
    st.caption(f"Run folder: `{run_dir}`")

    zip_name = f"{os.path.basename(run_dir)}.zip"
    st.download_button(
        label=f"Download all ({len(files)} files) as ZIP",
        data=build_run_zip_bytes(run_dir),
        file_name=zip_name,
        mime="application/zip",
        key=f"zip_{run_dir}",
    )

    priority = [rel for rel, _ in files if rel in ("run_config.json", "channel_mapping_config.json")]
    priority += [
        rel
        for rel, _ in files
        if rel.endswith(".csv") or ("SUMMARY" in rel and rel.endswith(".png")) or "MASTER_RESULTS" in rel or "STAT_SUMMARY" in rel or "IMAGE_LEVEL_MEDIANS" in rel or "INTENSITY_PAIRED_IMAGES" in rel or "PAIRED_OTSU_SPOT_CHANGE" in rel or "OTSU_DIM_NOISE" in rel
    ]
    priority = list(dict.fromkeys(priority))

    for rel in priority:
        abspath = os.path.join(run_dir, rel)
        if not os.path.isfile(abspath):
            continue
        with open(abspath, "rb") as fh:
            st.download_button(
                label=f"Download {rel}",
                data=fh.read(),
                file_name=os.path.basename(abspath),
                mime=_guess_mime(abspath),
                key=f"dl_{run_dir}_{rel}",
            )

    with st.expander(f"All files in this run ({len(files)})"):
        for rel, abspath in files:
            with open(abspath, "rb") as fh:
                st.download_button(
                    label=rel,
                    data=fh.read(),
                    file_name=os.path.basename(abspath),
                    mime=_guess_mime(abspath),
                    key=f"dl_all_{run_dir}_{rel}",
                )
