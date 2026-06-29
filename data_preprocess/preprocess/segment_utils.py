"""Grounded-SAM video segmentation adapter used by SoMA preprocessing."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.ops import box_convert

from .utils import add_python_path


class GroundedSamVideoSegmenter:
    def __init__(
        self,
        sam2_root: str | Path,
        grounding_dino_root: str | Path,
        sam2_checkpoint: str | Path,
        grounding_dino_config: str | Path,
        grounding_dino_checkpoint: str | Path,
        sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.sam2_root = Path(sam2_root).expanduser().resolve()
        self.grounding_dino_root = Path(grounding_dino_root).expanduser().resolve()
        self.sam2_checkpoint = str(Path(sam2_checkpoint).expanduser().resolve())
        self.grounding_dino_config = str(Path(grounding_dino_config).expanduser().resolve())
        self.grounding_dino_checkpoint = str(Path(grounding_dino_checkpoint).expanduser().resolve())
        self.sam2_model_config = sam2_model_config
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        add_python_path(self.sam2_root)
        add_python_path(self.grounding_dino_root)
        from sam2.build_sam import build_sam2, build_sam2_video_predictor  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        from groundingdino.util.inference import load_image, load_model, predict  # type: ignore

        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        grounding_model = load_model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device,
        )
        self._loaded = {
            "build_sam2": build_sam2,
            "build_sam2_video_predictor": build_sam2_video_predictor,
            "SAM2ImagePredictor": SAM2ImagePredictor,
            "load_image": load_image,
            "predict": predict,
            "sam2_model": sam2_model,
            "grounding_model": grounding_model,
        }
        return self._loaded

    @staticmethod
    def _frame_names(video_path: str | Path) -> list[str]:
        frame_names = [
            p
            for p in Path(video_path).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not frame_names:
            raise FileNotFoundError(f"No frames found in {video_path}")
        return [p.name for p in sorted(frame_names, key=lambda p: int(p.stem))]

    def segment_video(self, video_path: str | Path, text_prompt: str):
        loaded = self._load()
        frame_names = self._frame_names(video_path)
        video_path = str(video_path)

        video_predictor = loaded["build_sam2_video_predictor"](self.sam2_model_config, self.sam2_checkpoint)
        image_predictor = loaded["SAM2ImagePredictor"](loaded["sam2_model"])
        inference_state = video_predictor.init_state(video_path=video_path)

        ann_frame_idx = 0
        img_path = str(Path(video_path) / frame_names[ann_frame_idx])
        image_source, image = loaded["load_image"](img_path)
        boxes, confidences, labels = loaded["predict"](
            model=loaded["grounding_model"],
            image=image,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        if len(labels) == 0:
            raise RuntimeError(f"GroundingDINO found no objects for prompt {text_prompt!r} in {img_path}")

        h, w, _ = image_source.shape
        boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        image_predictor.set_image(image_source)

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        with amp_context:
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        for object_id, (label, box) in enumerate(zip(labels, input_boxes)):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        id_to_objects = {i: obj for i, obj in enumerate(labels)}
        return video_segments, id_to_objects

    def segment_video_with_points(self, video_path: str | Path, points, point_labels=None):
        loaded = self._load()
        frame_names = self._frame_names(video_path)
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (N, 3) with [x, y, frame_idx]")
        if point_labels is None:
            point_labels = np.ones((points.shape[0],), dtype=np.int64)
        point_labels = np.asarray(point_labels).astype(np.int64)
        if point_labels.shape[0] != points.shape[0]:
            raise ValueError("point_labels length mismatch points")

        video_predictor = loaded["build_sam2_video_predictor"](self.sam2_model_config, self.sam2_checkpoint)
        inference_state = video_predictor.init_state(video_path=str(video_path))
        frame_to_points = {}
        for i, (x, y, fidx) in enumerate(points):
            fidx = int(fidx)
            if 0 <= fidx < len(frame_names):
                frame_to_points.setdefault(fidx, []).append((x, y, point_labels[i]))

        for fidx, plist in frame_to_points.items():
            pts_xy = np.array([[p[0], p[1]] for p in plist], dtype=np.float32)
            labels = np.array([p[2] for p in plist], dtype=np.int64)
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=fidx,
                obj_id=0,
                points=pts_xy,
                labels=labels,
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments, {0: "object"}
