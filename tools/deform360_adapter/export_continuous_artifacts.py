"""T31.1 serialization only: receives already-computed tensors, never runs a model."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def tensor_hash(value):
    return hashlib.sha256(np.ascontiguousarray(array(value)).tobytes()).hexdigest()

class ArtifactWriter:
    def __init__(self, output, repository):
        self.out, self.repo = Path(output), Path(repository)
        self.cameras = ['brics-odroid-023_cam0', 'brics-odroid-009_cam1']
        self.short = ['023_cam0', '009_cam1']
        self.base = self.repo / 'docs/deform360/contracts/008-pink-cloth/episode_0'
        self.rgb = json.loads((self.base / 'rgb_export_contract.json').read_text())
        self.mask = json.loads((self.base / 'mask_export_contract.json').read_text())
        self.states, self.renders = {}, {}
        (self.out / 'states').mkdir()
        for camera in self.short:
            (self.out / 'renders' / camera).mkdir(parents=True)

    def state(self, local, pos, cov, origin):
        assert local not in self.states
        pos, cov = array(pos), array(cov)
        assert pos.shape == (12861, 3) and cov.shape == (12861, 6)
        assert pos.dtype == cov.dtype == np.float32
        assert np.isfinite(pos).all() and np.isfinite(cov).all()
        path = self.out / 'states' / f'frame_{local:03d}.npz'
        np.savez_compressed(path, pred_pos=pos, pred_cov=cov)
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved['pred_pos'], pos) and np.array_equal(saved['pred_cov'], cov)
        self.states[local] = dict(local_frame=local, source_frame=113 + local,
            state_origin=origin, path=str(path.relative_to(self.out)), sha256=sha(path),
            pos_hash=tensor_hash(pos), cov_hash=tensor_hash(cov), dtype='float32',
            pred_pos_shape=list(pos.shape), pred_cov_shape=list(cov.shape))

    def render(self, local, images, origin):
        assert local not in self.renders and len(images) == 2
        records = []
        for i, image in enumerate(images):
            raw = array(image)
            assert raw.shape == (3, 360, 640) and np.isfinite(raw).all()
            # Display encoding only: never write into raw or feed quantized values back.
            pixels = np.rint(np.clip(raw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            path = self.out / 'renders' / self.short[i] / f'frame_{local:03d}.png'
            Image.fromarray(pixels).save(path)
            with Image.open(path) as im:
                assert im.mode == 'RGB' and im.size == (640, 360)
                assert np.array_equal(np.asarray(im), pixels)
            cam = self.cameras[i]
            refs = {}
            for kind, contract in [('rgb', self.rgb), ('mask', self.mask)]:
                item = contract['cameras'][cam]['manifest'][local]
                assert item['local_frame'] == local and item['source_frame'] == local + 113
                ref = Path(contract['canonical_output_root']) / item['path']
                assert sha(ref) == item['sha256']
                refs[kind + '_source'] = str(ref)
                refs[kind + '_sha256'] = item['sha256']
            records.append(dict(local_frame=local, source_frame=113 + local, camera_id=cam,
                render=str(path.relative_to(self.out)), sha256=sha(path), origin=origin,
                raw_tensor_sha256=tensor_hash(raw), raw_min=float(raw.min()), raw_max=float(raw.max()),
                display_clipped_values=int(((raw < 0) | (raw > 1)).sum()), **refs))
        self.renders[local] = records

    def prediction(self, local, pos, cov, images):
        self.state(local, pos, cov, 'model prediction')
        self.render(local, images, 'existing encode_decode.pred_img_list')

    def finalize(self, report):
        assert set(self.states) == set(self.renders) == set(range(194))
        assert len(report['steps']) == 193
        provenance = [dict(local_frame=0, source_frame=113, state_origin='canonical initial; exact T27 frame_0',
            previous_state_hash=self.states[0]['pos_hash'], current_state_hash=self.states[0]['pos_hash'],
            cache_key='frame_0', boundary_update=False, template_origin='canonical initial',
            consumer_previous_state_reset=False)]
        metrics = []
        for row in report['steps']:
            local = row['step']
            assert self.states[local]['pos_hash'] == row['prediction_hash']
            assert self.states[local]['cov_hash'] == row['covariance_hash']
            assert self.states[local - 1]['pos_hash'] == row['current_hash']
            provenance.append(dict(local_frame=local, source_frame=113 + local, state_origin='model prediction',
                previous_state_hash=row['current_hash'], current_state_hash=row['prediction_hash'],
                cache_key=f"frame_{row['template_local']}", boundary_update=local % 10 == 0,
                template_origin=row['template_origin'],
                consumer_previous_state_reset=row['previous_state_is_current_at_template_boundary']))
            record = dict(local_frame=local, source_frame=113 + local, covariance_PD_count=row['PD_count'],
                          pred_pos_finite=row['pred_pos_finite'], pred_cov_finite=row['pred_cov_finite'])
            record.update(row['metrics'])
            record.update({'displacement_' + k: v for k, v in row['displacement'].items()})
            metrics.append(record)
        for filename, rows in [('provenance.csv', provenance), ('metrics.csv', metrics)]:
            with (self.out / filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        cache = json.loads((self.base / 'stage1_cache_contract.json').read_text())['canonical']['frames'][0]
        assert cache['local_frame'] == 0 and sha(cache['path']) == cache['sha256']
        manifest = dict(schema_version=1, task='T31.1', status='PASS', exploratory=True,
            checkpoint=report['checkpoint'], config=dict(git_commit=report['head'],
                path='configs/SoMA/deform360_v0_stage2.py',
                sha256=sha(self.repo / 'configs/SoMA/deform360_v0_stage2.py'),
                camera_config='camera_config_2cam.json', camera_config_sha256=sha(self.base / 'camera_config_2cam.json'),
                evaluation_overrides=report['protocol']['eval_overrides'],
                T27_cache_initial=dict(path=cache['path'], sha256=cache['sha256']),
                T27_contract_sha256=sha(self.base / 'stage1_cache_contract.json')),
            sequence=dict(start_source=113, end_source=306, num_steps=193, total_frames=194,
                          initial_frame=0, predicted_local_frames=[1,193]),
            artifact=dict(render_count=388, predicted_render_count=386, initial_render_count=2,
                          state_count=194, predicted_state_count=193, initial_state_count=1,
                          camera_order=self.cameras),
            conversion='PNG: CHW RGB -> HWC; clip[0,1], round(value*255), uint8; no model feedback; metrics use original floats',
            initial_render='First existing encode_decode_render_only result, template frame0; not a predicted transition',
            provenance_columns_note='previous_state_hash is prior dense POSITION hash; consumer_previous_state_reset separately records known velocity-history reassignment. Later frame_N cache_key names online prediction templates, not disk-cache reads.',
            limitations=[report['previous_state_boundary_reassignment'],
                'This is a new same-protocol inference for export, not recovery of tensors from the old exited process.',
                'No bitwise equality claim against old inference; registered state invariant checked.',
                'Display PNG quantization/clipping is not a replacement for float-domain metrics.',
                'Initial frame has no transition metric; metrics.csv contains193 rows.'],
            future_PLY_read=report['future_PLY_read'], future_cache_injection=report['future_cache_injection'],
            registered_state_unchanged=report['registered_state_unchanged'],
            job_id=report['job_id'], frames=[dict(state=self.states[i], cameras=self.renders[i]) for i in range(194)],
            table_hashes={name:sha(self.out / name) for name in ['provenance.csv','metrics.csv']})
        (self.out / 'README.md').write_text('# T31.1 连续推演导出\n\n193次transition，194帧（initial + 193 predictions）。两相机共388PNG，194NPZ。\n'
            'local0/source113为已有初始render，不是假预测。其余local1–193/source114–306。\n'
            'RGB/mask只引用canonical路径。metrics从本次既有计算结果转换，未从PNG重新评分。\n'
            '每10帧prev_state重设限制原样保留；T31 exploratory结论不改变。\n'
            'manifest记录逐文件hash；manifest.sha256独立记录manifest自身hash，避免自引用。\n')
        manifest['readme_sha256'] = sha(self.out / 'README.md')
        (self.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        (self.out / 'manifest.sha256').write_text(sha(self.out / 'manifest.json') + '  manifest.json\n')
        return {k:manifest[k] for k in ['status','checkpoint','config','sequence','artifact','limitations','job_id']}
