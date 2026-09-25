#!/usr/bin/env python3
"""T14 only: sequential RGB export; unique cameras shared by both configurations.

CPU dependencies: numpy, opencv-python, Pillow. Refuses existing output paths.
Example:
  python -B tools/deform360_adapter/export_rgb.py --workspace /path/to/tcgs \
    --output /path/to/t14_rgb --contract /path/to/rgb_export_contract.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--contract', required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    repo = workspace / 'SoMA'
    base = repo / 'docs/deform360/contracts/008-pink-cloth/episode_0'
    names = ['frame_manifest.json', 'camera_intrinsics_contract.json',
             'camera_config_2cam.json', 'camera_config_3cam.json', 'split_contract.json']
    hashes = {name: sha(base / name) for name in names}
    docs = {name: json.loads((base / name).read_text()) for name in names}
    mapping = docs['frame_manifest.json']
    intrinsics = docs['camera_intrinsics_contract.json']
    configs = {key: docs[name] for key, name in
               [('A', 'camera_config_2cam.json'), ('B', 'camera_config_3cam.json')]}
    pairs = [(row['source_frame'], row['local_frame']) for row in mapping['frames']]
    require(pairs == [(i + 113, i) for i in range(194)], 'Unexpected T4 mapping')
    require(not mapping['tactile_filter_applied'], 'Tactile filtering is forbidden')
    geometry = intrinsics['geometry']
    require(geometry['source_resolution_WH'] == [1280, 720] and
            geometry['target_resolution_WH'] == [640, 360], 'Unexpected T9 resolution')
    cameras = list(dict.fromkeys(cam for config in configs.values() for cam in config['camera_ids']))
    for key, config in configs.items():
        require(config['camera_ids'] == intrinsics['configurations'][key]['camera_ids'], 'T7/T9 order mismatch')
        require([entry['camera_id'] for entry in config['camera_order']] == config['camera_ids'], 'Invalid order')
    require(len(cameras) == 3, 'Expected three unique cameras')
    episode = workspace / mapping['source_episode']
    output = args.output.resolve()
    require(not output.exists() and not args.contract.exists(), 'Refusing to overwrite existing artifacts')
    # Preflight all sources before creating derived data. Timestamps are not reinterpreted.
    source_hashes = {}
    for camera in cameras:
        video = episode / camera / 'undistorted.mp4'
        timestamps = episode / camera / 'aligned_timestamps.txt'
        relative = str(timestamps.relative_to(workspace))
        require(sha(timestamps) == mapping['timestamp_file_sha256'][relative], 'Timestamp hash mismatch')
        tokens = timestamps.read_text().splitlines()
        require(all(tokens[src].strip() == row['timestamps'][camera]
                    for (src, _), row in zip(pairs, mapping['frames'])), 'Timestamp mapping mismatch')
        source_hashes[camera] = sha(video)
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for camera in cameras:
        destination = output / camera
        destination.mkdir()
        video = episode / camera / 'undistorted.mp4'
        cap = cv2.VideoCapture(str(video))
        require(cap.isOpened(), 'Video open failed: ' + camera)
        manifest = []
        source_index = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                require(bgr.shape == (720, 1280, 3) and bgr.dtype == np.uint8, 'Invalid decoded source')
                if 113 <= source_index <= 306:
                    local = source_index - 113
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    small = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
                    target = destination / f'{local}.png'
                    with target.open('xb') as handle:
                        Image.fromarray(small).save(handle, format='PNG')
                    with Image.open(target) as image:
                        require(image.mode == 'RGB', 'Incorrect PNG color mode')
                        decoded = np.asarray(image)
                        require(np.array_equal(decoded, small), 'PNG RGB roundtrip mismatch')
                    manifest.append({'local_frame': local, 'source_frame': source_index,
                                     'path': str(target.relative_to(output)), 'sha256': sha(target),
                                     'rgb_pixel_sha256': hashlib.sha256(small.tobytes()).hexdigest(),
                                     'min': int(small.min()), 'max': int(small.max())})
                source_index += 1
        finally:
            cap.release()
        require(source_index == 357 and len(manifest) == 194, 'Unexpected decoded frame count')
        require({p.name for p in destination.iterdir()} == {f'{i}.png' for i in range(194)}, 'File numbering error')
        require(len({row['rgb_pixel_sha256'] for row in manifest}) == 194, 'Duplicate RGB content requires review')
        # Separate full sequential decode. Independent 2x2 box-average checks
        # cover all exported frames, not just the requested representative frames.
        cap = cv2.VideoCapture(str(video))
        require(cap.isOpened(), 'Verification video open failed')
        frame = 0
        max_error = 0
        samples = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if 113 <= frame <= 306:
                    rgb = bgr[..., ::-1]
                    reference = rgb.reshape(360, 2, 640, 2, 3).mean(axis=(1, 3))
                    with Image.open(destination / f'{frame - 113}.png') as image:
                        pixels = np.asarray(image)
                        require(pixels.shape == (360, 640, 3) and pixels.dtype == np.uint8, 'Invalid output')
                        error = float(np.max(np.abs(pixels.astype(float) - reference)))
                    require(error <= 0.5, 'Source correspondence/resize/channel mismatch')
                    max_error = max(max_error, error)
                    if frame in [113, 123, 200, 267, 268, 306]:
                        samples.append({'source_frame': frame, 'local_frame': frame - 113,
                                        'max_error_vs_independent_box_average_uint8_units': error})
                frame += 1
        finally:
            cap.release()
        require(frame == 357, 'Verification decode incomplete')
        require(sha(video) == source_hashes[camera], 'Input changed during export')
        results[camera] = {'source_video': str(video), 'source_sha256': source_hashes[camera],
                           'source_decoded_frame_count': source_index, 'output_directory': str(destination),
                           'output_count': 194, 'unique_pixel_hash_count': 194, 'manifest': manifest,
                           'validation': {'all_frames_redecoded_sequentially': True,
                                          'all_png_rgb_roundtrips_exact': True,
                                          'max_error_vs_box_average': max_error, 'samples': samples}}
        print(camera, 'PASS: 194 RGB PNG; verified all frames', flush=True)
    require(all(sha(base / name) == h for name, h in hashes.items()), 'Input contract changed')
    contract = {'task': 'T14', 'status': 'PASS', 'dataset': '008-pink-cloth/episode_0',
                'source_frame_range_inclusive': [113, 306], 'local_frame_range_inclusive': [0, 193],
                'frame_count': 194, 'source_resolution_WH': [1280, 720], 'target_resolution_WH': [640, 360],
                'resize': 'OpenCV INTER_AREA full-frame 2x downsample; 2x2 box support centered at 2*u+0.5, matching T9 half-pixel convention; no crop, warp or undistortion',
                'color': 'OpenCV decoded BGR explicitly converted to RGB; Pillow writes RGB PNG',
                'dtype': 'uint8', 'value_range': [0, 255], 'video_seek_used': False,
                'tactile_filter_applied': False, 'split_changed': False, 'mask_processed': False,
                'canonical_output_root': str(output), 'canonical_camera_ids': cameras,
                'configurations': {key: {'camera_ids': config['camera_ids'],
                    'assets_in_order': [str(output / cam) for cam in config['camera_ids']]}
                    for key, config in configs.items()}, 'cameras': results,
                'provenance': {'contract_sha256': hashes, 'tool_sha256': sha(Path(__file__)),
                               'opencv_version': cv2.__version__, 'numpy_version': np.__version__},
                'limitations': 'CPU decoding/encoding and pixel checks only; no scene packaging or loader/training execution. Source is compressed video; validation compares decoded pixels, not uncompressed camera originals.'}
    with args.contract.open('x') as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print('PASS contract:', args.contract, flush=True)


if __name__ == '__main__':
    main()
