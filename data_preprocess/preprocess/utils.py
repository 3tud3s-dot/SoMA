"""Shared helpers for the SoMA preprocessing pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_python_path(path: str | Path | None) -> None:
    if not path:
        return
    path = str(Path(path).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def run_command(args: list[str], cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("[run]", " ".join(args))
    subprocess.run(args, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def copy_file(src: str | Path, dst: str | Path) -> None:
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_pickle(path: str | Path) -> Any:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def save_mask(mask: np.ndarray, save_path: str | Path) -> None:
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask should be 2D after squeeze, got shape {mask.shape}")
    mask = (mask * 255).astype(np.uint8)
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    Image.fromarray(mask, mode="L").save(save_path)


def sorted_scene_dirs(data_root: str | Path, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None) -> list[Path]:
    data_root = Path(data_root).expanduser().resolve()
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    scenes = []
    for item in sorted(data_root.iterdir()):
        if not item.is_dir():
            continue
        if include_set and item.name not in include_set:
            continue
        if item.name in exclude_set:
            continue
        scenes.append(item)
    return scenes


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        base_dir = Path.cwd()
    return Path(base_dir).expanduser().resolve() / path
