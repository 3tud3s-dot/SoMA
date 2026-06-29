"""Command-line entrypoint for SoMA data preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .utils import load_config, load_pickle, resolve_path, save_json, sorted_scene_dirs


def run_augmentation(seq_dir: Path, data_root: Path, cfg: dict[str, Any]) -> None:
    from .reconstruction import clean_the_desk, compute_desk, pcd_align

    fixed_cam_id = int(cfg.get("fixed_cam_id", 0))
    aligned_scene = cfg.get("aligned_scene")
    if not aligned_scene:
        raise ValueError("augmentation requires aligned_scene in the config")

    desk_pcd_path = seq_dir / "pi3" / "observation_desk.ply"
    desk_clean_path = seq_dir / "pi3" / "observation_desk_clean.ply"
    clean_the_desk(desk_pcd_path, desk_clean_path)

    c2w = load_pickle(seq_dir / "calibrate.pkl")[fixed_cam_id]
    scene_info = compute_desk(desk_clean_path, c2w)

    gs_iter = cfg.get("gaussian", {}).get("iterations", 10000)
    aligned_pcd_path = data_root / aligned_scene / "pi3" / "gs" / "point_cloud" / f"iteration_{gs_iter}" / "point_cloud.ply"
    current_pcd_path = seq_dir / "pi3" / "gs" / "point_cloud" / f"iteration_{gs_iter}" / "point_cloud.ply"
    scene_info["transformation"] = pcd_align(aligned_pcd_path, current_pcd_path).tolist()
    save_json(scene_info, seq_dir / "scene_info.json")


def process_scene(seq_dir: Path, data_root: Path, cfg: dict[str, Any]) -> None:
    steps = cfg.get("steps", [])
    print(f"========== Processing {seq_dir.name} ==========")

    if "segmentation" in steps:
        print("[1/6] segmentation")
        from .segmentation import run_segmentation

        run_segmentation(seq_dir, cfg)

    if "reconstruction" in steps:
        print("[2/6] reconstruction")
        from .reconstruction import run_reconstruction

        run_reconstruction(seq_dir, cfg)

    if "gaussian" in steps:
        print("[3/6] gaussian")
        from .gaussian import run_gaussian

        run_gaussian(seq_dir, cfg)

    if "augmentation" in steps:
        print("[4/6] augmentation")
        run_augmentation(seq_dir, data_root, cfg)

    if "robot_alignment" in steps:
        print("[5/6] robot alignment")
        from .robot import align_robot

        robot_dir_value = cfg.get("robot", {}).get("robot_dir") or (data_root.parent / "robot")
        robot_dir = Path(robot_dir_value)
        align_robot(str(robot_dir), str(seq_dir), int(cfg.get("fixed_cam_id", 0)), cfg.get("dataset_name", data_root.name))

    if "robot_tracking" in steps:
        print("[6/6] robot tracking")
        from .robot import track_robot

        particle_num = int(cfg.get("robot", {}).get("particle_num", 30))
        track_robot(str(seq_dir), particle_num=particle_num)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run SoMA data preprocessing.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--data-root", default=None, help="Override data_root from the config.")
    parser.add_argument("--steps", nargs="*", default=None, help="Override stage list, e.g. --steps robot_alignment robot_tracking")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    cfg["_repo_dir"] = str(config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.steps is not None:
        cfg["steps"] = args.steps

    data_root = resolve_path(cfg["data_root"], cfg["_repo_dir"])
    scene_filter = cfg.get("scene_filter", {})
    scenes = sorted_scene_dirs(data_root, scene_filter.get("include"), scene_filter.get("exclude"))
    if not scenes:
        raise RuntimeError(f"No scene directories found under {data_root}")

    for seq_dir in scenes:
        process_scene(seq_dir, data_root, cfg)


if __name__ == "__main__":
    main()
