#!/usr/bin/env python3
"""T16.5: lossless, per-frame zero-SH removal; run inside a Slurm GPU job.

Preflight the entire T4 window before creating output. Never overwrite artifacts.
A failure leaves any already written derived files for inspection, never resumes
silently. No cross-frame correspondence or rollout reset semantics are implied.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from plyfile import PlyData, PlyElement

REST = [f'f_rest_{i}' for i in range(45)]
PREFIX = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
SUFFIX = ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
EXPECTED = PREFIX + REST + SUFFIX
CONTRACT_DIR = 'docs/deform360/contracts/008-pink-cloth/episode_0'
EPISODE = 'datasets/deform360/processed/008-pink-cloth/episode_0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_source(path):
    ply = PlyData.read(path)
    require([e.name for e in ply.elements] == ['vertex'], f'{path}: unexpected elements')
    v = ply['vertex'].data
    require(list(v.dtype.names) == EXPECTED, f'{path}: property schema/order mismatch')
    require(len(v) > 0, f'{path}: empty vertex array')
    for name in EXPECTED:
        require(v[name].dtype == np.dtype('<f4'), f'{path}: unexpected dtype {name}')
        require(np.isfinite(v[name]).all(), f'{path}: nonfinite {name}')
    for name in REST:
        require((v[name] == 0).all(), f'{path}: NONZERO {name}; conversion forbidden')
    # This episode uses fixed-size binary records: reject trailing/truncated data.
    require(not ply.text and ply.byte_order == '<', f'{path}: unexpected PLY format')
    with path.open('rb') as f:
        while True:
            line = f.readline()
            require(bool(line), f'{path}: missing end_header')
            if line.strip() == b'end_header':
                break
        require(path.stat().st_size - f.tell() == v.nbytes, f'{path}: payload size mismatch')
    return ply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    repo = root / 'SoMA'
    require(os.environ.get('SLURM_JOB_ID'), 'Run in Slurm GPU allocation, not login node')
    require(not args.output.exists(), 'Output already exists; refusing overwrite/resume')
    require(not args.contract.exists(), 'Contract already exists; refusing overwrite')
    manifest_path = repo / CONTRACT_DIR / 'frame_manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    pairs = [(r['source_frame'], r['local_frame']) for r in manifest['frames']]
    require(pairs == [(s, s - 113) for s in range(113, 307)], 'T4 mapping mismatch')
    rows = []
    for source, local in pairs:
        path = root / EPISODE / 'splatfacto' / f'splat_{source}.ply'
        digest = sha(path)
        ply = read_source(path)
        require(sha(path) == digest, f'{path}: changed during preflight')
        rows.append(dict(source_frame=source, local_frame=local,
                         source_path=str(path.relative_to(root)), source_sha256=digest,
                         vertex_count=len(ply['vertex'].data), schema_check='PASS',
                         finite_check='PASS', zero_SH_check='PASS'))
    print('PREFLIGHT PASS: 194 sources, f_rest_0..44 strictly zero', flush=True)

    # All CUDA operations (including this diagnostic) are allocation-only.
    smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True)
    import torch
    require(torch.cuda.is_available(), 'CUDA unavailable inside allocation')
    loader = repo / 'gaussian-splatting/scene/gaussian_model.py'
    loader_hash = sha(loader)
    sys.path.insert(0, str(repo / 'gaussian-splatting'))
    spec = importlib.util.spec_from_file_location('t16_5_original_gaussian_model', loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=False)
    retained = PREFIX + SUFFIX
    shapes = {'_xyz': [3], '_features_dc': [1, 3], '_features_rest': [0, 3],
              '_opacity': [1], '_scaling': [3], '_rotation': [4]}
    for row in rows:
        source_path = root / row['source_path']
        require(sha(source_path) == row['source_sha256'], 'Source changed after preflight')
        ply = read_source(source_path)
        v = ply['vertex'].data
        reduced = np.empty(len(v), dtype=[(n, v.dtype.fields[n][0]) for n in retained])
        for name in retained:
            reduced[name] = v[name]
        derived = PlyData([PlyElement.describe(reduced, 'vertex', comments=ply['vertex'].comments)],
                          text=ply.text, byte_order=ply.byte_order,
                          comments=ply.comments, obj_info=ply.obj_info)
        out = args.output / f"splat_{row['source_frame']}_sh0.ply"
        with out.open('xb') as f:
            derived.write(f)
        reread = PlyData.read(out)
        rv = reread['vertex'].data
        require(len(rv) == len(v) and list(rv.dtype.names) == retained, 'Derived schema/count mismatch')
        require(reread.comments == ply.comments and reread.obj_info == ply.obj_info,
                'PLY metadata changed')
        for name in retained:
            require(rv[name].dtype == v[name].dtype, f'Dtype changed: {name}')
            require(rv[name].tobytes() == v[name].tobytes(), f'Values/order changed: {name}')
            require(np.isfinite(rv[name]).all(), f'Derived nonfinite: {name}')
        with torch.no_grad():
            model = module.GaussianModel(0)
            model.load_ply(str(out))
            for attr, tail in shapes.items():
                tensor = getattr(model, attr)
                require(list(tensor.shape) == [len(v)] + tail, f'Loader shape: {attr}')
                require(tensor.dtype == torch.float32 and tensor.is_cuda, f'Loader dtype/device: {attr}')
                require(torch.isfinite(tensor).all().item(), f'Loader nonfinite: {attr}')
            del tensor, model
        torch.cuda.empty_cache()
        row.update(output_path=str(out.relative_to(root)), output_sha256=sha(out),
                   output_bytes=out.stat().st_size, retained_fields_bitwise_equal=True,
                   vertex_order_preserved=True, derived_f_rest_count=0, loader_validation='PASS')
        print(f"PASS source={row['source_frame']} local={row['local_frame']} vertices={len(v)}", flush=True)
    for row in rows:
        require(sha(root / row['source_path']) == row['source_sha256'], 'Source changed during run')
        require(sha(root / row['output_path']) == row['output_sha256'], 'Output changed during run')
    require(sha(loader) == loader_hash, 'Loader changed during run')
    require(sha(manifest_path) == manifest_hash, 'T4 manifest changed during run')
    contract = dict(
        schema_version=1, task='T16.5', status='PASS',
        workspace_path_base=str(root), repository_head=subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        frame_mapping_reference=f'SoMA/{CONTRACT_DIR}/frame_manifest.json',
        frame_mapping_sha256=manifest_hash, source_range_inclusive=[113, 306],
        local_range_inclusive=[0, 193], frame_count=194,
        prior_evidence='SoMA/docs/deform360/SoMA-D360-v0-implementation-todo.md#t2 and #t3',
        conversion_rule='Remove only f_rest_0..44 after full-window strict-zero/finite/schema preflight; preserve remaining field dtype/order/bytes and vertex order.',
        source_schema=[[n, '<f4'] for n in EXPECTED],
        derived_schema=[[n, '<f4'] for n in retained],
        file_format='binary_little_endian; comments/obj_info preserved',
        identity_boundary='Within-frame conversion only. No fixed cross-frame Gaussian identity is assumed; future reconstructed PLY must not reset rollout state.',
        output_directory=str(args.output.relative_to(root)), server_only=True,
        output_total_bytes=sum(r['output_bytes'] for r in rows),
        validation=dict(all_194_strict_zero=True, all_194_lossless=True,
                        original_sources_unchanged=True, loader_unchanged=True),
        loader_validation=dict(status='PASS', count=194, sh_degree=0,
            loader_path='SoMA/gaussian-splatting/scene/gaussian_model.py', sha256=loader_hash,
            slurm_job_id=os.environ['SLURM_JOB_ID'], gpu=smi.strip(), torch_version=torch.__version__,
            cuda_available=True, tensor_shapes_per_vertex=shapes, dtype='float32', device='cuda'),
        tool=dict(repository_path='tools/deform360_adapter/convert_gaussian_sequence_sh0.py', sha256=sha(Path(__file__))),
        frames=rows)
    with args.contract.open('x') as f:
        json.dump(contract, f, indent=2)
        f.write('\n')
    print(f"COMPLETE: 194 PASS; {contract['output_total_bytes']} bytes; {args.contract}", flush=True)


if __name__ == '__main__':
    main()
