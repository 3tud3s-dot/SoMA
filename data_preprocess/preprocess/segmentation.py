"""Segmentation stage for SoMA preprocessing.

This module intentionally keeps GroundingDINO/SAM2 imports lazy, because those
repositories and checkpoints are large optional dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .utils import copy_file, ensure_dir, resolve_path, save_mask


class GroundedSamSegmenter:
    def __init__(self, cfg: dict[str, Any]):
        third_party = cfg.get("third_party", {})
        checkpoints = cfg.get("checkpoints", {})
        config_dir = cfg.get("_repo_dir", cfg.get("_config_dir"))
        sam2_root = third_party.get("grounded_sam_root")
        grounding_dino_root = third_party.get("grounding_dino_root")
        if not sam2_root or not grounding_dino_root:
            raise RuntimeError(
                "Segmentation requires third_party.grounded_sam_root and "
                "third_party.grounding_dino_root. Run scripts/install_third_party.sh "
                "or set these paths in the config."
            )
        from .segment_utils import GroundedSamVideoSegmenter

        self.segmenter = GroundedSamVideoSegmenter(
            sam2_root=resolve_path(sam2_root, config_dir),
            grounding_dino_root=resolve_path(grounding_dino_root, config_dir),
            sam2_checkpoint=resolve_path(checkpoints["sam2"], config_dir),
            grounding_dino_config=resolve_path(checkpoints["grounding_dino_config"], config_dir),
            grounding_dino_checkpoint=resolve_path(checkpoints["grounding_dino"], config_dir),
            sam2_model_config=cfg.get("sam2_model_config", "configs/sam2.1/sam2.1_hiera_l.yaml"),
            box_threshold=float(cfg.get("box_threshold", 0.35)),
            text_threshold=float(cfg.get("text_threshold", 0.25)),
            device=cfg.get("device"),
        )
        self.upscale_cfg = cfg.get("upscale", {})

    def segment_video(self, image_dir: str | Path, prompt: str):
        return self.segmenter.segment_video(image_dir, prompt)

    def upscale_image(self, image_path: str | Path, prompt: str):
        from .image_upscale import upscale_image

        return upscale_image(
            image_path=image_path,
            object_name=prompt,
            model_id=self.upscale_cfg.get("model_id", "stabilityai/stable-diffusion-x4-upscaler"),
            device=self.upscale_cfg.get("device", "cuda"),
            noise_level=self.upscale_cfg.get("noise_level"),
            guidance_scale=self.upscale_cfg.get("guidance_scale"),
        )

def _find_object_id(id_to_objects: dict, label: str) -> int | None:
    for key, value in id_to_objects.items():
        if value == label:
            return int(key)
    return None


def _inverse_union(frame_masks: dict[int, np.ndarray]) -> np.ndarray:
    union_mask = None
    for mask in frame_masks.values():
        mask_bool = np.asarray(mask) > 0
        union_mask = mask_bool if union_mask is None else np.logical_or(union_mask, mask_bool)
    if union_mask is None:
        raise ValueError("No masks are available for inverse-union desk mask.")
    return np.logical_not(union_mask).astype(np.float32)


def run_segmentation(seq_dir: str | Path, cfg: dict[str, Any]) -> None:
    seq_dir = Path(seq_dir)
    max_cam = int(cfg.get("max_cam", 3))
    prompt = cfg["prompts"]["all"]
    object_label = cfg["prompts"]["object"]
    desk_label = cfg["prompts"].get("desk", "table")

    segmenter = GroundedSamSegmenter(cfg)
    pi3_dir = ensure_dir(seq_dir / "pi3")
    mask_dir = ensure_dir(seq_dir / "mask")
    ensure_dir(pi3_dir / "images")
    ensure_dir(pi3_dir / "masks")
    ensure_dir(pi3_dir / "desk_masks")

    for cam_id in range(max_cam):
        static_dir = seq_dir / "static" / str(cam_id)
        static_segments, static_objects = segmenter.segment_video(static_dir, prompt)
        object_id = _find_object_id(static_objects, object_label)
        desk_id = _find_object_id(static_objects, desk_label)
        if object_id is None:
            raise RuntimeError(f"Could not find object label {object_label!r} in {static_dir}")

        copy_file(static_dir / "0.jpg", pi3_dir / "images" / f"{cam_id}.jpg")
        save_mask(static_segments[0][object_id], pi3_dir / "masks" / f"{cam_id}.jpg")
        if desk_id is None:
            save_mask(_inverse_union(static_segments[0]), pi3_dir / "desk_masks" / f"{cam_id}.jpg")
        else:
            save_mask(static_segments[0][desk_id], pi3_dir / "desk_masks" / f"{cam_id}.jpg")

        if cfg.get("gaussian", {}).get("use_high_res", False):
            upscaled_dir = ensure_dir(pi3_dir / "upscaled_images" / str(cam_id))
            upscaled_image_path = upscaled_dir / "0.jpg"
            upscaled_image = segmenter.upscale_image(static_dir / "0.jpg", object_label)
            upscaled_image.save(upscaled_image_path)
            upscaled_segments, upscaled_objects = segmenter.segment_video(upscaled_dir, prompt)
            upscaled_object_id = _find_object_id(upscaled_objects, object_label)
            if upscaled_object_id is None:
                raise RuntimeError(f"Could not find object label {object_label!r} in {upscaled_dir}")
            save_mask(upscaled_segments[0][upscaled_object_id], pi3_dir / f"mask_{cam_id}_high.png")
            copy_file(upscaled_image_path, pi3_dir / f"{cam_id}_high.png")

        dynamic_dir = seq_dir / "color" / str(cam_id)
        dynamic_segments, dynamic_objects = segmenter.segment_video(dynamic_dir, prompt)
        with open(mask_dir / f"mask_info_{cam_id}.json", "w", encoding="utf-8") as f:
            json.dump(dynamic_objects, f, indent=2)
        for frame_idx, frame_masks in dynamic_segments.items():
            for mask_idx, mask_key in enumerate(dynamic_objects.keys()):
                save_mask(frame_masks[mask_key], mask_dir / str(cam_id) / str(mask_idx) / f"{frame_idx}.jpg")
