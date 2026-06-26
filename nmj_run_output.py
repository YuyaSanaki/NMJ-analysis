"""Timestamped run folders under ``output/`` for analysis artifacts (not raw CZI inputs)."""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime
from typing import Any, Iterator

OUTPUT_ROOT = "output"


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
        cfg_path = os.path.join(td, "channel_mapping_config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as jf:
                    loaded = json.load(jf)
                if isinstance(loaded, dict):
                    folder_map = loaded
            except (OSError, json.JSONDecodeError):
                pass

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
        if rel.endswith(".csv") or ("SUMMARY" in rel and rel.endswith(".png")) or "MASTER_RESULTS" in rel
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
