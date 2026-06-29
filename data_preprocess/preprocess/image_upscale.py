"""High-resolution image upscaling used by the Gaussian preprocessing stage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_PIPELINE = None
_PIPELINE_KEY = None


def _load_pipeline(model_id: str, device: str):
    global _PIPELINE, _PIPELINE_KEY
    key = (model_id, device)
    if _PIPELINE is not None and _PIPELINE_KEY == key:
        return _PIPELINE
    import torch
    from diffusers import StableDiffusionUpscalePipeline

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    _PIPELINE = pipeline
    _PIPELINE_KEY = key
    return pipeline


def _crop_with_mask(image: Image.Image, mask_path: str | Path) -> Image.Image:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    bbox_pixels = np.argwhere(mask > 0.8 * 255)
    if bbox_pixels.size == 0:
        return image
    x0, y0, x1, y1 = (
        np.min(bbox_pixels[:, 1]),
        np.min(bbox_pixels[:, 0]),
        np.max(bbox_pixels[:, 1]),
        np.max(bbox_pixels[:, 0]),
    )
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    size = int(max(x1 - x0, y1 - y0) * 1.2)
    crop_box = (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + size // 2),
        int(center[1] + size // 2),
    )
    return image.crop(crop_box)


def upscale_image(
    image_path: str | Path,
    object_name: str,
    mask_path: str | Path | None = None,
    noise_level: int | None = None,
    guidance_scale: float | None = None,
    model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
    device: str = "cuda",
):
    low_res_img = Image.open(image_path).convert("RGB")
    if mask_path is not None:
        low_res_img = _crop_with_mask(low_res_img, mask_path)
    prompt = f"A {object_name} is placed on a desk."
    pipeline = _load_pipeline(model_id, device)
    if noise_level is None:
        return pipeline(prompt=prompt, image=low_res_img).images[0]
    return pipeline(
        prompt=prompt,
        image=low_res_img,
        noise_level=noise_level,
        guidance_scale=guidance_scale,
    ).images[0]
