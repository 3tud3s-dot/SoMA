"""Camera/point-cloud reconstruction utilities for SoMA preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .utils import ensure_dir, resolve_path


def pcd_align(source_pcd_path: str | Path, target_pcd_path: str | Path) -> np.ndarray:
    source_pcd = o3d.io.read_point_cloud(str(source_pcd_path))
    target_pcd = o3d.io.read_point_cloud(str(target_pcd_path))
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        0.05,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result.transformation


def downsample_pcd(
    source_path: str | Path,
    target_path: str | Path,
    downsample_type: str = "voxel",
    downsample_rate: float = 0.1,
    voxel_size: float = 0.006,
) -> None:
    source_pcd = o3d.io.read_point_cloud(str(source_path))
    if downsample_type == "random":
        total_points = len(source_pcd.points)
        keep_points = max(1, int(total_points * downsample_rate))
        indices = np.random.choice(total_points, keep_points, replace=False)
        source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source_pcd.points)[indices])
        if source_pcd.has_colors():
            source_pcd.colors = o3d.utility.Vector3dVector(np.asarray(source_pcd.colors)[indices])
    elif downsample_type == "voxel":
        source_pcd = source_pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        raise ValueError(f"Unknown downsample_type: {downsample_type}")
    ensure_dir(Path(target_path).parent)
    o3d.io.write_point_cloud(str(target_path), source_pcd)


def run_pi3(seq_dir: str | Path, cfg: dict[str, Any]) -> bool:
    """Run Pi3 using the SoMA adapter vendored in this package."""
    third_party = cfg.get("third_party", {})
    pi3_repo_value = third_party.get("pi3_repo")
    if not pi3_repo_value:
        raise RuntimeError(
            "Pi3 reconstruction requires third_party.pi3_repo. "
            "Run scripts/install_third_party.sh or set this path in the config."
        )
    from .pi3_adapter import run_pi3 as soma_run_pi3

    return soma_run_pi3(
        str(seq_dir),
        mounted_cam=cfg.get("mounted_cam_id", 0),
        valid_cam_ids=cfg.get("valid_cam_ids"),
        fixe_cam_pos=cfg.get("fix_camera_pose", True),
        pi3_repo=resolve_path(pi3_repo_value, cfg.get("_repo_dir", cfg.get("_config_dir"))),
        pi3_model=cfg.get("pi3_model", "yyfz233/Pi3"),
    )


def run_reconstruction(seq_dir: str | Path, cfg: dict[str, Any]) -> None:
    seq_dir = Path(seq_dir)
    run_pi3(seq_dir, cfg)
    downsample_cfg = cfg.get("downsample", {})
    downsample_pcd(
        seq_dir / "pi3" / "observation_original.ply",
        seq_dir / "pi3" / "observation.ply",
        downsample_type=downsample_cfg.get("type", "voxel"),
        downsample_rate=float(downsample_cfg.get("rate", 0.1)),
        voxel_size=float(downsample_cfg.get("voxel_size", 0.006)),
    )


def clean_the_desk(desk_pcd_path: str | Path, save_pcd_path: str | Path, voxel_size: float = 0.005) -> None:
    desk_pcd = o3d.io.read_point_cloud(str(desk_pcd_path))
    desk_pcd = desk_pcd.voxel_down_sample(voxel_size=voxel_size)
    desk_pcd, _ = desk_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    _, inliers = desk_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    clean_desk_pcd = desk_pcd.select_by_index(inliers)
    ensure_dir(Path(save_pcd_path).parent)
    o3d.io.write_point_cloud(str(save_pcd_path), clean_desk_pcd)


def compute_desk(desk_pcd_path: str | Path, c2w: np.ndarray) -> dict[str, Any]:
    from scipy.spatial.transform import Rotation as R

    pcd = o3d.io.read_point_cloud(str(desk_pcd_path))
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 3:
        raise ValueError("Point cloud has fewer than 3 points; cannot estimate a desk plane.")

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    normal = np.array(plane_model[:3], dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    centroid = pts[inliers].mean(axis=0) if len(inliers) > 0 else pts.mean(axis=0)

    cam_pos = np.array(c2w[:3, 3], dtype=np.float64).reshape(3)
    if np.dot(normal, centroid - cam_pos) < 0:
        normal = -normal

    a, b, c, d_raw = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    norm = np.linalg.norm(n) + 1e-9
    n = n / norm
    d = d_raw / norm

    pts_centered = pts - centroid
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    axis_long = eigvecs[:, 0]
    axis_long = axis_long / (np.linalg.norm(axis_long) + 1e-9)
    axis_short = np.cross(normal, axis_long)
    axis_short = axis_short / (np.linalg.norm(axis_short) + 1e-9)

    coords_long = pts_centered @ axis_long
    coords_short = pts_centered @ axis_short
    length = float(coords_long.max() - coords_long.min())
    width = float(coords_short.max() - coords_short.min())

    normal = normal / (np.linalg.norm(normal) + 1e-9)
    d_plus = np.linalg.norm((centroid + normal) - cam_pos)
    d_minus = np.linalg.norm((centroid - normal) - cam_pos)
    gravity_dir = normal if d_plus >= d_minus else -normal

    if np.dot(n, gravity_dir) > 0:
        n = -n
        d = -d
    a, b, c = n.tolist()
    plane_params = [a, b, c, d]

    target = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)
    source = gravity_dir.reshape(1, 3)
    scene_rot, _ = R.align_vectors(target, source)
    scene_quat = scene_rot.as_quat().astype(np.float64)

    return {
        "plane_params": plane_params,
        "length": length,
        "width": width,
        "gravity_rot_quat": scene_quat.tolist(),
    }
