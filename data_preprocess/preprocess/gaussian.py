"""3D Gaussian generation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import copy_file, resolve_path, run_command


def prepare_gaussian_inputs(seq_dir: str | Path, max_cam: int) -> Path:
    work_dir = Path(seq_dir) / "pi3"
    for cam_id in range(max_cam):
        copy_file(work_dir / "images" / f"{cam_id}.jpg", work_dir / f"{cam_id}.png")
        copy_file(work_dir / "masks" / f"{cam_id}.jpg", work_dir / f"mask_{cam_id}.png")
    return work_dir


def run_gaussian(seq_dir: str | Path, cfg: dict[str, Any]) -> None:
    work_dir = prepare_gaussian_inputs(seq_dir, int(cfg.get("max_cam", 3)))
    gaussian_cfg = cfg.get("gaussian", {})
    script_dir_value = gaussian_cfg.get("script_dir")
    if not script_dir_value:
        raise RuntimeError("Gaussian stage requires gaussian.script_dir to point to the SoMA Gaussian adapter containing gs_train.py and gs_render.py.")
    script_dir = resolve_path(script_dir_value, cfg.get("_repo_dir", cfg.get("_config_dir")))
    train_script = script_dir / "gs_train.py"
    render_script = script_dir / "gs_render.py"
    missing = [str(path) for path in (train_script, render_script) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Gaussian adapter script(s): " + ", ".join(missing))
    model_dir = work_dir / "gs"

    train_args = [
        "python",
        str(train_script),
        "-s",
        str(work_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(gaussian_cfg.get("iterations", 10000)),
        "--lambda_depth",
        str(gaussian_cfg.get("lambda_depth", 0)),
        "--lambda_normal",
        str(gaussian_cfg.get("lambda_normal", 0)),
        "--lambda_anisotropic",
        str(gaussian_cfg.get("lambda_anisotropic", 0.0)),
        "--use_masks",
        "--lambda_seg",
        str(gaussian_cfg.get("lambda_seg", 0.5)),
        "--gs_init_opt",
        str(gaussian_cfg.get("gs_init_opt", "hybrid")),
        "--percent_dense",
        str(gaussian_cfg.get("percent_dense", 0.03)),
        "--densify_until_iter",
        str(gaussian_cfg.get("densify_until_iter", 7000)),
        "--densify_grad_threshold",
        str(gaussian_cfg.get("densify_grad_threshold", 0.0008)),
        "--sh_degree",
        str(gaussian_cfg.get("sh_degree", 0)),
        "--isotropic",
    ]
    if gaussian_cfg.get("use_high_res", False):
        for cam_id in range(int(cfg.get("max_cam", 3))):
            for suffix in (f"{cam_id}_high.png", f"mask_{cam_id}_high.png"):
                high_res_path = work_dir / suffix
                if not high_res_path.exists():
                    raise FileNotFoundError(f"High-resolution Gaussian input is enabled but {high_res_path} is missing.")
        train_args.append("--use_high_res")
    run_command(train_args)
    run_command(["python", str(render_script), "-s", str(work_dir), "-m", str(model_dir)])
