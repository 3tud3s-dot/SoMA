#!/usr/bin/env python3
"""T17 CPU-only scene packaging; existing canonical assets are linked, not copied.

No EmbodiedDataset/Camera/model imports, clustering, rendering or training.
The JSON interface is an explicit integration contract, not an executable config.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess

import joblib
import numpy as np
from PIL import Image

CONTRACTS = 'docs/deform360/contracts/008-pink-cloth/episode_0'
NAMES = ['frame_manifest', 'split_contract', 'camera_config_2cam', 'camera_config_3cam',
         'camera_extrinsics_contract', 'camera_intrinsics_contract',
         'controller_trajectory_contract', 'controller_grouping_contract',
         'gravity_world_frame_contract', 'rgb_export_contract', 'mask_export_contract',
         'supervision_mask_contract', 'gaussian_sequence_sh0_contract']


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
    check(json.loads(path.read_text()) == value, f'JSON roundtrip {path}')


def link(path, target):
    check(target.exists(), f'Missing target {target}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent))
    check(path.resolve() == target.resolve(), f'Link mismatch {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    a = p.parse_args()
    root = a.workspace.resolve()
    repo = root / 'SoMA'
    out = a.output.absolute()
    check(not out.exists() and not a.contract.exists(), 'Refuse overwrite')
    cp = repo / CONTRACTS
    docs = {n: json.loads((cp / (n + '.json')).read_text()) for n in NAMES}
    inputs = {str((cp / (n + '.json')).relative_to(root)): sha(cp / (n + '.json')) for n in NAMES}
    mapping = docs['frame_manifest']['frames']
    check([(r['source_frame'], r['local_frame']) for r in mapping] == [(s, s-113) for s in range(113,307)], 'T4 mapping')
    split = docs['split_contract']
    check(split['local_split_half_open'] == {'train': [0,155], 'test': [155,194]}, 'T5 split')
    traj = docs['controller_trajectory_contract']['output']
    traj_path = root / traj['path_relative_to_workspace']
    check(sha(traj_path) == traj['sha256'], 'T11 hash')
    arr = np.load(traj_path, allow_pickle=False)
    check(arr.shape == (194,30,3) and arr.dtype == np.float32 and np.isfinite(arr).all(), 'T11 array')
    check(hashlib.sha256(arr.tobytes()).hexdigest() == traj['array_bytes_sha256'], 'T11 values')
    grouping = docs['controller_grouping_contract']
    check(grouping['hierarchy_counts'] == [30,10,2,1], 'T12 hierarchy')
    grav = docs['gravity_world_frame_contract']
    check(grav['raw_gravity_world'] == [0.,0.,-9.8] and grav['rot_est_xyzw'] == [0.,0.,0.,1.], 'T13 convention')
    np.testing.assert_allclose(np.array(grav['raw_gravity_world'])*4, grav['final_external_gravity'])
    gs = docs['gaussian_sequence_sh0_contract']['frames'][0]
    check(gs['source_frame'] == 113 and gs['local_frame'] == 0, 'Initial Gaussian')
    gs_path = root / gs['output_path']
    check(sha(gs_path) == gs['output_sha256'], 'Initial SH0 hash')
    # Preflight all shared image assets once; full manifests remain in T14/T15.
    checked_assets = {}
    for name, mode in [('rgb_export_contract','RGB'), ('mask_export_contract','L')]:
        d = docs[name]
        for cam, info in d['cameras'].items():
            check(len(info['manifest']) == 194, 'Image count')
            check([(f['source_frame'],f['local_frame']) for f in info['manifest']] == [(s,s-113) for s in range(113,307)], 'Image mapping')
            for f in info['manifest']:
                path = Path(d['canonical_output_root']) / f['path']
                check(sha(path) == f['sha256'], f'Image hash {path}')
                with Image.open(path) as im:
                    check(im.size == (640,360) and im.mode == mode, f'Image schema {path}')
            checked_assets[f'{name}:{cam}'] = 194
    # Check the actual loader string replacement before any output is written.
    for name, count in [('config_a_2cam',2), ('config_b_3cam',3)]:
        for i in range(count):
            mask_dir = out / name / 'mask' / str(i)
            expected = mask_dir.parent / f'mask_info_{i}.json'
            check(str(mask_dir).replace(f'/{i}', f'/mask_info_{i}.json') == str(expected),
                  'Output parent collides with loader mask-info string replacement')
    out.mkdir(parents=True)
    shared = out / 'shared'
    shared.mkdir()
    track = shared / 'track_process_data.pkl'
    with track.open('xb') as f:
        pickle.dump({'controller_points': arr}, f, protocol=4)
    with track.open('rb') as f:
        loaded = pickle.load(f)
    check(list(loaded) == ['controller_points'] and loaded['controller_points'].tobytes() == arr.tobytes(), 'Track PKL fidelity')
    check(loaded['controller_points'].dtype == np.float32, 'Track dtype')
    configurations = {}
    for key, name in [('A','config_a_2cam'), ('B','config_b_3cam')]:
        scene = out / name
        scene.mkdir()
        ext = docs['camera_extrinsics_contract']['configurations'][key]
        intr = docs['camera_intrinsics_contract']['configurations'][key]
        ids = ext['camera_ids']
        check(ids == intr['camera_ids'], 'T8/T9 identity/order')
        for n in ['rgb_export_contract','mask_export_contract']:
            check(ids == docs[n]['configurations'][key]['camera_ids'], 'T14/T15 identity/order')
        config = docs['camera_config_2cam' if key == 'A' else 'camera_config_3cam']
        check(ids == config['camera_ids'], 'T7 identity/order')
        c2w = np.asarray(ext['c2w'], dtype=np.float64)
        check(c2w.shape == (len(ids),4,4) and np.isfinite(c2w).all(), 'Calibration shape')
        with (scene / 'calibrate.pkl').open('xb') as f:
            pickle.dump(c2w, f, protocol=4)
        check(np.array_equal(joblib.load(scene/'calibrate.pkl'),c2w), 'Calibration joblib roundtrip')
        metadata = {'intrinsics': intr['K'], 'serial_numbers': ids, 'fps': 30,
                    'WH': [640,360], 'frame_num': 194}
        write_json(scene/'metadata.json',metadata)
        write_json(scene/'scene_info.json',{'gravity_rot_quat': grav['rot_est_xyzw']})
        link(scene/'track_process_data.pkl',track)
        link(scene/'pi3/gs/point_cloud/iteration_10000/point_cloud.ply',gs_path)
        for i, cam in enumerate(ids):
            rgb = Path(docs['rgb_export_contract']['cameras'][cam]['output_directory'])
            mask = Path(docs['mask_export_contract']['cameras'][cam]['output_directory'])
            link(scene/'color'/str(i),rgb)
            link(scene/'mask'/str(i)/'1',mask)
            write_json(scene/'mask'/f'mask_info_{i}.json',{'1':'cloth'})
            for local in range(194):
                check((scene/'color'/str(i)/f'{local}.png').is_file(), 'RGB path')
                check((scene/'mask'/str(i)/'1'/f'{local}.png').is_file(), 'Mask path')
            # Exact io.py convention; sole object and no fabricated obstacle masks.
            actual = str(scene/'mask'/str(i)).replace(f'/{i}',f'/mask_info_{i}.json')
            check(json.loads(Path(actual).read_text()) == {'1':'cloth'}, 'Mask label contract')
        interface = {
            'purpose': 'T17 partial interface contract; not a runnable training config or executed dataset.',
            'env_cfg_common': {
                'data_dir': str(out), 'scene_list': [name], 'sub_img_dir': 'color',
                'sub_video_dir': 'color', 'cam_transform_fn': None, 'resolution': [360,640],
                'load_imgs': False, 'scale_x_angle': 1.0, 'max_cam_num': len(ids),
                'max_cam_total': len(ids), 'max_seq': 1, 'pad_cam': False, 'llffhold': 0,
                'tracking_pcd': 'track_process_data.pkl', 'prompt_dict': {'object':['cloth'],'obstacle':[]},
                'bounding_box': None, 'use_random_background': {name:False},
                'const_white_bg': {name:False}, 'controller_cfg': {name:grouping['controller_scheme']},
                'cluster_type': grouping['cluster_type'], 'gravity': grav['raw_gravity_world'],
                'rot_est': {name:grav['rot_est_xyzw']}, 'dt': grav['time_scaling_rule']['comp_dt'],
                'real_dt': {name:grav['time_scaling_rule']['real_dt']}},
            'stage_dataset_frame_gap': {'stage1':10,'stage2':1},
            'train_split_list': {name:[[0,155]]},
            'continuous_evaluation_split_list': {name:[[0,194]]},
            'score_local_half_open': [155,194], 'warmup_local_half_open': [0,155],
            'evaluation_warning': 'Scoring mask must be integrated later; never train on evaluation range or use [155,194] as a reset segment.',
            'initial_gaussian': {'model_path':str(scene/'pi3/gs/point_cloud/iteration_10000/point_cloud.ply'),'sh_degree':0,'source_frame':113,'local_frame':0},
            'future_reconstructed_gaussian_reset_allowed': False,
            'gravity_result_after_dataset_only': grav['final_external_gravity'],
            'gravity_limitation': 'User-approved -Z engineering convention, not measured physical vertical; same for tactile/no-tactile.',
            'grouping_counts': grouping['hierarchy_counts'],
            'not_generated': ['Gaussian/controller cluster caches','graph','Stage-1 cache','training configuration'],
            'remaining_training_config': 'Model/object cluster_cfg/volume_scalar and runtime schedule belong to later tasks; this interface does not fabricate them.'}
        write_json(scene/'scene_interface.json',interface)
        configurations[key] = {'scene_directory': str(scene.relative_to(root)), 'camera_ids':ids,
                              'numeric_camera_mapping':{str(i):cam for i,cam in enumerate(ids)}, 'metadata': metadata,
                              'interface': interface}
    generated = {}
    links = {}
    for path in out.rglob('*'):
        if path.is_symlink():
            check(path.exists(),f'Broken link {path}')
            links[str(path.relative_to(root))] = str(path.resolve().relative_to(root))
        elif path.is_file():
            generated[str(path.relative_to(root))] = {'sha256':sha(path),'bytes':path.stat().st_size}
    check(sha(traj_path) == traj['sha256'] and sha(gs_path) == gs['output_sha256'], 'Inputs modified')
    for path,digest in inputs.items():
        check(sha(root/path)==digest,'Contract modified')
    code_paths = ['mmgs/datasets/embodied_dataset.py','mmgs/datasets/utils/io.py','configs/SoMA/cloth_lift_stage1.py']
    contract = {
        'task':'T17','status':'PASS','validation_scope':'Static schema/path/hash only; no dataset/Camera/model instantiation, CUDA, render, forward or training.',
        'workspace_path_base':str(root),'package_root':str(out.relative_to(root)),
        'provenance':{'repository_head':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
                      'input_contract_sha256':inputs,'loader_source_sha256':{p:sha(repo/p) for p in code_paths},
                      'packager_sha256':sha(Path(__file__))},
        'configurations':configurations,'generated_files':generated,'symlink_mapping':links,
        'controller':{'canonical_input':traj,'packaged_file':str(track.relative_to(root)), 'array_values_and_dtype_unchanged':True},
        'initial_gaussian':gs,'checked_image_assets':checked_assets,
        'frame_mapping_reference':f'SoMA/{CONTRACTS}/frame_manifest.json',
        'split_reference':f'SoMA/{CONTRACTS}/split_contract.json',
        'validation':{'all_paths_resolve':True,'camera_order_preserved':True,'WH':[640,360],'frames':194,
                      'controller_shape':[194,30,3],'PKL_JSON_roundtrip':True,'input_hashes_match':True,
                      'sole_object_label':{'1':'cloth'},'large_assets_copied':False},
        'loader_evidence':{'camera':'embodied_dataset.py:211-260,976-1024; c2w array, intrinsics/WH/frame_num',
                           'controller':'embodied_dataset.py:611-626,918-919; controller_points only',
                           'gaussian':'embodied_dataset.py:513-518; fixed pi3 path',
                           'mask':'utils/io.py:367-456; numeric camera/mask_info_ID.json and object label directory',
                           'gravity':'embodied_dataset.py:921-934; raw gravity times ratio squared once, inverse identity rotation',
                           'sample':'datasets/soma_sample/soma_data_sample/cloth_lift/left_lift_1; actual JSON/PKL schemas read before packaging'},
        'limitations':['PASS is packaging only, not a training/loader execution PASS.',
                       'Training loader may permute camera batching; stored calibration and asset order remain fixed.',
                       'T16 no-obstacle-mask semantics retained; object mask is not a robot mask.',
                       'Package path avoids /0,/1,/2 parent collisions in current mask-info string replacement.',
                       'Only initial source113 PLY linked; no future reconstruction/reset paths.',
                       'Continuous evaluation scoring integration remains a later task.']}
    write_json(a.contract,contract)
    print(json.dumps({'status':'PASS','package':str(out),'files':len(generated),'links':len(links),
                      'new_file_bytes':sum(v['bytes'] for v in generated.values())},indent=2))


if __name__ == '__main__':
    main()
