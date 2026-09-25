#!/usr/bin/env python3
"""T15 CPU-only mask export. Refuses existing outputs; no segmentation or RGB edits."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    a = ap.parse_args()
    w = a.workspace.resolve()
    base = w / 'SoMA/docs/deform360/contracts/008-pink-cloth/episode_0'
    names = ['frame_manifest.json', 'rgb_export_contract.json', 'camera_intrinsics_contract.json',
             'camera_config_2cam.json', 'camera_config_3cam.json', 'split_contract.json']
    hashes = {n: sha(base / n) for n in names}
    docs = {n: json.loads((base / n).read_text()) for n in names}
    rgb = docs['rgb_export_contract.json']
    frames = docs['frame_manifest.json']['frames']
    check([(f['source_frame'], f['local_frame']) for f in frames] == [(113+i, i) for i in range(194)], 'T4 mapping')
    cameras = rgb['canonical_camera_ids']
    check(cameras == ['brics-odroid-023_cam0', 'brics-odroid-009_cam1', 'brics-odroid-014_cam1'], 'Camera set')
    for key, name in [('A', 'camera_config_2cam.json'), ('B', 'camera_config_3cam.json')]:
        ids = docs[name]['camera_ids']
        check(ids == rgb['configurations'][key]['camera_ids'] == docs['camera_intrinsics_contract.json']['configurations'][key]['camera_ids'], 'Camera order')
    check(docs['camera_intrinsics_contract.json']['geometry']['target_resolution_WH'] == [640, 360], 'T9 size')
    for path in [a.output, a.contract, a.review]:
        check(not path.exists(), 'Refusing existing output: ' + str(path))
    episode = w / docs['frame_manifest.json']['source_episode']
    source_hashes = {c: sha(episode / c / 'mask_refined.h5') for c in cameras}
    # Verify all RGB assets before writing, preventing accidental pairing to stale exports.
    for c in cameras:
        entries = rgb['cameras'][c]['manifest']
        check(len(entries) == 194, 'RGB manifest count')
        for i, entry in enumerate(entries):
            check((entry['local_frame'], entry['source_frame']) == (i, i+113), 'RGB mapping')
            check(sha(Path(rgb['canonical_output_root']) / entry['path']) == entry['sha256'], 'RGB hash')
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    review = Image.new('RGB', (4*320, 3*200), 'white')
    draw = ImageDraw.Draw(review)
    samples = [113, 200, 267, 306]
    for ci, camera in enumerate(cameras):
        dest = a.output / camera
        dest.mkdir()
        records = []
        with h5py.File(episode / camera / 'mask_refined.h5', 'r') as h:
            data = h['data']
            check(data.shape == (357, 720, 1280) and data.dtype == np.uint8, 'Source shape/dtype')
            for local in range(194):
                source = local + 113
                mask = data[source]
                check(set(np.unique(mask)).issubset({0, 1}), 'Unexpected source labels; no semantic remapping allowed')
                image = Image.fromarray(mask * np.uint8(255)).resize((640, 360), Image.Resampling.NEAREST)
                target = dest / f'{local}.png'
                with target.open('xb') as handle:
                    image.save(handle, format='PNG')
                with Image.open(target) as saved:
                    pixels = np.asarray(saved)
                    check(saved.mode == 'L' and pixels.shape == (360, 640) and pixels.dtype == np.uint8, 'Output format')
                    check(set(np.unique(pixels)).issubset({0, 255}) and np.any(pixels), 'Output labels/empty foreground')
                    # Half-pixel nearest, ties to higher index: floor(2*u+1) = 2*u+1.
                    check(np.array_equal(pixels, data[source][1::2, 1::2] * np.uint8(255)), 'Independent nearest/index verification')
                records.append({'source_frame': source, 'local_frame': local, 'path': f'{camera}/{local}.png',
                                'sha256': sha(target), 'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                                'foreground_pixels': int(np.count_nonzero(pixels))})
                if source in samples:
                    entry = rgb['cameras'][camera]['manifest'][local]
                    with Image.open(Path(rgb['canonical_output_root']) / entry['path']) as im:
                        colors = np.asarray(im).copy()
                    fg = pixels > 0
                    colors[fg] = np.rint(.6*colors[fg] + .4*np.array([0, 255, 0])).astype(np.uint8)
                    overlay = Image.fromarray(colors).resize((320, 180), Image.Resampling.BOX)
                    x, y = samples.index(source)*320, ci*200
                    review.paste(overlay, (x, y+20))
                    draw.text((x+3, y+3), f'{camera.replace("brics-odroid-", "")} source {source} / local {local}', fill='black')
        check({p.name for p in dest.iterdir()} == {f'{i}.png' for i in range(194)}, 'Numbering')
        check(len({p['pixel_sha256'] for p in records}) == 194, 'Repeated mask content requires review')
        check(sha(episode / camera / 'mask_refined.h5') == source_hashes[camera], 'Input mask changed')
        results[camera] = {'source_hdf5': str(episode / camera / 'mask_refined.h5'), 'source_sha256': source_hashes[camera],
                           'output_directory': str(dest), 'output_count': 194, 'unique_pixel_hash_count': 194,
                           'foreground_min_max': [min(x['foreground_pixels'] for x in records), max(x['foreground_pixels'] for x in records)],
                           'manifest': records}
        print(camera, 'PASS 194 masks', flush=True)
    review.save(a.review, format='JPEG', quality=90)
    check(all(sha(base / n) == h for n, h in hashes.items()), 'Input contract modified')
    for c in cameras:
        for entry in rgb['cameras'][c]['manifest']:
            check(sha(Path(rgb['canonical_output_root']) / entry['path']) == entry['sha256'], 'RGB changed')
    contract = {'task': 'T15', 'status': 'PASS', 'hdf5_key': 'data', 'source_resolution_WH': [1280, 720],
                'target_resolution_WH': [640, 360], 'source_frame_range_inclusive': [113, 306], 'local_frame_range_inclusive': [0, 193],
                'resize': {'method': 'Pillow Resampling.NEAREST', 'crop': False, 'warp': False,
                           'pixel_mapping': 'T9 half-pixel inverse x_src=2*x_dst+0.5; nearest tie to higher index; sampled source [1::2,1::2]',
                           'validation': 'All PNG arrays exactly equal independently indexed source data [1::2,1::2] * 255'},
                'value_convention': {'source': [0, 1], 'output': [0, 255], 'background': 0, 'foreground': 255, 'mode': 'L', 'dtype': 'uint8'},
                'canonical_output_root': str(a.output), 'canonical_camera_ids': cameras, 'cameras': results,
                'configurations': {k: {'camera_ids': v['camera_ids'], 'assets_in_order': [str(a.output / c) for c in v['camera_ids']]} for k,v in rgb['configurations'].items()},
                'validation': {'all_frames_finite_binary_nonempty': True, 'counts_and_numbering': True, 'duplicate_content': False,
                               'T14_RGB_hashes_unchanged': True, 'source_HDF5_hashes_unchanged': True, 'representative_source_frames': samples,
                               'overlay_review': str(a.review), 'overlay_sha256': sha(a.review), 'visual_review': 'pending'},
                'provenance': {'input_contract_hashes': hashes, 'tool_sha256': sha(Path(__file__)), 'h5py_version': h5py.__version__},
                'boundaries': 'No segmentation, mask morphology, tactile filtering, split changes, RGB modification, scene packaging, loader/model/config edits, training or T16.'}
    with a.contract.open('x') as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
