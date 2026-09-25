"""T11 only: config-driven controller NPY plus a small provenance contract.

Run with the existing server NumPy environment. No CUDA, tactile, loader, PKL,
cluster or graph imports. Both output paths must be new.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_opening_function(path):
    """Extract just the pure reference function/constants, not its module imports."""
    names = {'MIN_OPENING_M', 'MAX_OPENING_M', 'JOINT_UPPER_M', 'JOINT_LOWER_M'}
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'opening_to_umi_joints':
            nodes.append(node)
    scope = {'np': np, 'Dict': dict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['opening_to_umi_joints']


def origin_matrix(joint):
    origin = joint.find('origin')
    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    result = np.eye(4)
    result[:3, :3] = np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ])
    result[:3, 3] = xyz
    return result


def urdf_anchor_root(link, joints_by_child, positions, root_link):
    """Independent XML joint-chain FK, including the original fixed anchor."""
    result = np.eye(4)
    while link != root_link:
        joint = joints_by_child[link]
        move = np.eye(4)
        value = positions.get(joint.get('name'), 0.0)
        if joint.get('type') == 'prismatic':
            move[:3, 3] = np.fromstring(joint.find('axis').get('xyz'), sep=' ') * value
        elif joint.get('type') in ('revolute', 'continuous'):
            require(value == 0, 'This baseline validation supports zero finger revolute angles only')
        else:
            require(joint.get('type') == 'fixed', 'Unsupported URDF joint')
        result = origin_matrix(joint) @ move @ result
        link = joint.find('parent').get('link')
    return result[:3, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, required=True)
    parser.add_argument('--contract-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root, directory = args.workspace_root.resolve(), args.contract_dir.resolve()
    output, report = args.output.resolve(), args.report.resolve()
    require(output != report and not output.exists() and not report.exists(), 'Output/report already exists or paths overlap')
    require(output.suffix == '.npy', 'Output must be a standalone NPY')
    paths = {name: directory/name for name in ('controller_geometry_config.json', 'controller_single_frame_contract.json', 'frame_manifest.json')}
    cfg, t10, mapping = (json.loads(paths[n].read_text()) for n in paths)
    require(cfg['canonical_candidate_id'] == cfg['default_candidate_id'] == 'candidate_7mm', 'T11 requires the human-confirmed canonical candidate_7mm')
    candidate = next(c for c in cfg['candidates'] if c['candidate_id'] == cfg['canonical_candidate_id'])
    require(candidate['is_canonical_for_v0_baseline'] and candidate['offset_mm'] == cfg['surface_offset_mm'], 'Canonical selection mismatch')
    require(cfg['num_points'] == 30 and cfg['left_indices'] == list(range(15)) and cfg['right_indices'] == list(range(15,30)), 'Unexpected point identity groups')
    require(cfg['coordinate_frame'] == 'world' and cfg['controller_dtype'] == 'float32', 'Unexpected output convention')
    require(sha(paths['controller_single_frame_contract.json']) == cfg['relation_to_t10_contract']['sha256'], 'T10 hash mismatch')
    require(sha(paths['frame_manifest.json']) == t10['provenance']['frame_manifest_sha256'], 'T4 hash mismatch')
    records = mapping['frames']
    sources = [r['source_frame'] for r in records]
    require([r['local_frame'] for r in records] == list(range(194)) and sources == list(range(113,307)), 'T4 must map source 113..306 to local 0..193')
    require(mapping['frame_count'] == 194 and not mapping['tactile_filter_applied'], 'T4 count/filter mismatch')
    robot_path = root/cfg['source']['robot_pose']['path_relative_to_workspace']
    input_paths = {str(p.relative_to(root)): p for p in paths.values()}
    for role in ('robot_pose', 'opening_mapping', 'urdf_geometry', 'geometry_reference'):
        item = cfg['source'][role]
        p = root/item['path_relative_to_workspace']
        require(sha(p) == item['sha256'], 'Input hash mismatch: '+str(p))
        input_paths[str(p.relative_to(root))] = p
    hashes_before = {k: sha(p) for k,p in input_paths.items()}
    robot = np.load(robot_path, allow_pickle=True).item()
    require(not robot['bimanual'], 'Expected single-gripper episode')
    pose = np.asarray(robot['T_worlds'][sources], dtype=np.float64)
    opening = np.asarray(robot['openings'][sources], dtype=np.float64)
    require(pose.shape == (194,4,4) and opening.shape == (194,), 'Robot shape mismatch')
    require(np.isfinite(pose).all() and np.isfinite(opening).all(), 'Non-finite robot data')
    require(np.array_equal(pose[:,3,:], np.tile([0,0,0,1], (194,1))), 'Invalid homogeneous row')
    rotations = pose[:,:3,:3]
    ortho_error = float(np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3))))
    det_error = float(np.max(np.abs(np.linalg.det(rotations)-1)))
    require(max(ortho_error,det_error)<1e-8, 'Invalid robot rotation')
    grid, kin = cfg['fixed_grid'], cfg['kinematics']
    om = kin['opening_mapping']
    lo, hi = om['clip_bounds_m']
    clipped = np.clip(opening,lo,hi)
    joint = om['joint_upper_m']-(clipped-lo)/(hi-lo)*(om['joint_upper_m']-om['joint_lower_m'])
    official = official_opening_function(root/cfg['source']['opening_mapping']['path_relative_to_workspace'])
    reference_joints = [official(float(o)) for o in opening]
    opening_error = max(abs(joint[i]-r['joint_left']) for i,r in enumerate(reference_joints))
    require(opening_error<1e-15 and all(abs(r['joint_right']+joint[i])<1e-15 for i,r in enumerate(reference_joints)), 'Config and official opening mapping differ')
    require(all(v == 0 for v in kin['finger_revolute_positions_rad'].values()), 'Nonzero finger revolute angle')
    ids, local_points, bases, signs = [], [], [], []
    for side, sign in (('left',1),('right',-1)):
        direction = np.asarray(cfg['offset_direction'][side]['unit_vector'])
        for row in grid['rows']:
            for col in grid['columns']:
                p = direction * (candidate['offset_mm']/1000)
                p[1] += grid['y0_m']+grid['y_step_m']*col
                p[2] += grid['z_pitch_m']*(grid['z_row_reference']-row)
                local_points.append(p)
                bases.append(kin['finger_base_'+side+'_m']);signs.append(sign)
                ids.append(f'{side}_r{row:02d}_c{col:02d}')
    require(ids == grid['point_ids'] == [p['point_id'] for p in t10['point_ordering']], 'Point ordering mismatch')
    require(np.allclose(local_points,[p['finger_local_position_m'] for p in t10['point_ordering']],atol=1e-15,rtol=0), 'Canonical local anchors differ from T10')
    # Construct the fixed grid once. Only per-frame joints and poses are applied.
    root_points = np.broadcast_to(np.asarray(bases)+np.asarray(local_points)@np.asarray(kin['finger_to_root_rotation']).T,(194,30,3)).copy()
    # Match the T10 operation order for the local-X coordinate.
    root_points[:,:,0] = np.asarray(bases)[None,:,0]+joint[:,None]*np.asarray(signs)[None,:]+np.asarray(local_points)[None,:,0]
    world = np.matmul(root_points,rotations.transpose(0,2,1))+pose[:,None,:3,3]
    trajectory = np.asarray(world,dtype='<f4',order='C')
    require(trajectory.shape == (194,30,3) and np.isfinite(trajectory).all(), 'Invalid trajectory')
    require(np.array_equal(trajectory[0],np.asarray(t10['controller_points'],dtype=np.float32)), 'First frame differs from T10')
    require(all(len(np.unique(frame,axis=0))==30 for frame in trajectory), 'Duplicate controller positions')
    joints_xml = {j.find('child').get('link'):j for j in ET.parse(root/cfg['source']['urdf_geometry']['path_relative_to_workspace']).getroot().findall('joint')}
    sample_indices = [0,1,27,57,87,120,154,155,193]
    independent = []
    for i in sample_indices:
        positions = dict(reference_joints[i],**kin['finger_revolute_positions_rad'])
        reference_root = np.array([urdf_anchor_root(p['urdf_anchor_link'],joints_xml,positions,cfg['source']['urdf_geometry']['root_link']) for p in t10['point_ordering']])
        reference_world = reference_root@pose[i,:3,:3].T+pose[i,:3,3]
        error = float(np.max(np.abs(reference_world-world[i])))
        float_error = float(np.max(np.abs(reference_world.astype(np.float32)-trajectory[i])))
        require(error<1e-12 and float_error<1e-7, 'Independent URDF FK mismatch')
        independent.append({'local_frame':i,'source_frame':sources[i],'world_float64_max_abs_error_m':error,'float32_max_abs_error_m':float_error,'float32_equal':bool(np.array_equal(reference_world.astype(np.float32),trajectory[i]))})
    require(hashes_before == {k:sha(p) for k,p in input_paths.items()}, 'Input changed during generation')
    output.parent.mkdir(parents=True,exist_ok=True)
    report.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:
        np.save(f,trajectory,allow_pickle=False)
    reloaded = np.load(output,allow_pickle=False)
    require(reloaded.dtype==np.float32 and np.array_equal(reloaded,trajectory), 'Saved NPY roundtrip mismatch')
    gap_mm = (root_points[:,15,0]-root_points[:,0,0])*1000
    contract = {
        'schema_version':1,'task':'T11: fixed controller trajectory','status':'PASS',
        'authority':'Repository-owned trajectory provenance/validation contract; the NPY remains server-only.',
        'canonical_geometry_candidate':candidate['candidate_id'],'surface_offset_mm':candidate['offset_mm'],
        'input_sha256_relative_to_workspace':hashes_before,
        'source_frames_inclusive':[sources[0],sources[-1]],'local_frames_inclusive':[0,len(sources)-1],
        'frame_count':len(sources),'mapping_reference':'frame_manifest.json','mapping':'source_frame = local_frame + 113',
        'coordinate_frame':cfg['coordinate_frame'],'length_unit':cfg['length_unit'],
        'point_ordering':{'point_ids':ids,'left_indices':cfg['left_indices'],'right_indices':cfg['right_indices'],'definition_reference':'controller_single_frame_contract.json:point_ordering','fixed_grid_reference':'controller_geometry_config.json:fixed_grid','resampling':False,'reordering':False},
        'output':{'path_relative_to_workspace':str(output.relative_to(root)),'server_absolute_path':str(output),'format':'NPY; no pickle','shape':list(reloaded.shape),'dtype':str(reloaded.dtype),'bytes':output.stat().st_size,'sha256':sha(output),'array_bytes_sha256':hashlib.sha256(reloaded.astype('<f4',copy=False).tobytes(order='C')).hexdigest(),'git_policy':'server-only; do not add NPY to Git'},
        'validation':{'finite':True,'all_frames_30_unique_points':True,'point_identity_order_matches_t10':True,'first_frame_float32_bitwise_equal_t10':True,'npy_roundtrip_equal':True,'all_194_opening_values_match_official_function':True,'opening_mapping_max_abs_error_m':float(opening_error),'opening_clipped_frame_count':int(np.count_nonzero(opening!=clipped)),'pose_rotation_orthogonality_max_abs':ortho_error,'pose_rotation_determinant_max_error':det_error,'independent_method':'Compose XML URDF joint chains using the unchanged official opening_to_umi_joints, original anchor links and raw source pose; independent of the config closed-form geometry.','independent_samples':independent,'independent_float64_tolerance_m':1e-12,'independent_float32_tolerance_m':1e-7,'inputs_hashes_unchanged':True,'tactile_read':False,'frames_deleted':False},
        'known_limitation':{'description':'Retain canonical 7 mm signed-gap limitation; not a collision or physical-contact validation.','negative_gap_frame_count':int(np.count_nonzero(gap_mm<0)),'gap_min_median_max_mm':[float(np.min(gap_mm)),float(np.median(gap_mm)),float(np.max(gap_mm))],'gap_corrected':False},
        'execution':{'python':sys.version.split()[0],'numpy':np.__version__,'device':'CPU only','soma_head':subprocess.check_output(['git','-C',str(root/'SoMA'),'rev-parse','HEAD'],text=True).strip(),'generator_repository_path':'tools/deform360_adapter/build_controller_trajectory.py','generator_sha256':sha(Path(__file__)),'argv':sys.argv,'commit_push_performed':False,'pkl_generated':False,'graph_or_cluster_built':False},
    }
    with report.open('x') as f:
        json.dump(contract,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS','output':contract['output'],'validation':contract['validation'],'known_limitation':contract['known_limitation'],'report':str(report)},indent=2))


if __name__ == '__main__':
    main()
