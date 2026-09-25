#!/usr/bin/env python3
"""T19 actual EmbodiedDataset train sample check. Slurm GPU allocation required.

Only dataset-native clustering caches may be generated in derived scenes.
No graph/model construction, prediction, render, backward or training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from PIL import Image


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Requires GPU allocation'
    assert not a.report.exists(), 'Refuse overwrite'
    root=a.workspace.resolve();repo=root/'SoMA';cp=repo/'docs/deform360/contracts/008-pink-cloth/episode_0'
    os.chdir(repo);sys.path.insert(0,str(repo))
    report={'task':'T19','status':'IN_PROGRESS','slurm_job_id':os.environ['SLURM_JOB_ID'],
            'tool_sha256':sha(Path(__file__)),'configurations':{},
            'scope':'Actual EmbodiedDataset constructor and __getitem__(0), no graph/model.',
            'seed_note':'np.random.seed(5) immediately before each constructor yields identity camera permutation for 2/3 cameras in this check. The unmodified training loader generally randomizes camera batching; this does not lock future training order.'}
    code=repo/'mmgs/datasets/embodied_dataset.py';report['loader_sha256']=sha(code)
    try:
        report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
        import torch
        assert torch.cuda.is_available()
        from mmgs.datasets.embodied_dataset import EmbodiedDataset
        assert Path(sys.modules[EmbodiedDataset.__module__].__file__).resolve()==code.resolve()
        package=json.loads((cp/'scene_package_contract.json').read_text())
        grouping=json.loads((cp/'controller_grouping_contract.json').read_text())
        trajectory=json.loads((cp/'controller_trajectory_contract.json').read_text())
        trajpath=root/trajectory['output']['path_relative_to_workspace']
        assert sha(trajpath)==trajectory['output']['sha256']
        canonical=np.load(trajpath,allow_pickle=False)
        report['input_contract_sha256']={n:sha(cp/n) for n in ['scene_package_contract.json','controller_grouping_contract.json','controller_trajectory_contract.json']}
        for rel,info in package['generated_files'].items():assert sha(root/rel)==info['sha256']
        indices=np.arange(0,155,10)
        def describe(x):
            if isinstance(x,(torch.Tensor,np.ndarray)):
                val=x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else x
                assert np.isfinite(val).all()
                return {'shape':list(val.shape),'dtype':str(val.dtype),'finite':True}
            if isinstance(x,dict):return {k:describe(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [describe(v) for v in x]
            if isinstance(x,(int,float,str,bool)) or x is None:return x
            if hasattr(x,'img_path'):
                attrs={n:describe(getattr(x,n)) for n in ['R','T','world_view_transform','projection_matrix','full_proj_transform','camera_center','cam_plane_2_img']}
                attrs.update(img_path=x.img_path,image_height=x.image_height,image_width=x.image_width,FoVx=float(x.FoVx),FoVy=float(x.FoVy))
                assert np.isfinite([x.FoVx,x.FoVy]).all()
                return attrs
            raise TypeError(type(x))
        for key in ['A','B']:
            entry=package['configurations'][key];scene=root/entry['scene_directory'];name=scene.name
            cfg=json.loads((scene/'scene_interface.json').read_text())
            env=cfg['env_cfg_common']
            env.update(split_list=cfg['train_split_list'],frame_gap=10,
                cluster_cfg={name:[{'downsample_rate':0.02},{'downsample_rate':0.2}]},
                volume_scalar={name:512})
            summary={'env_cfg':env,'phase':'train','config_additions_provenance':'Official cloth_lift_stage1.py default_cluster_scheme and default_volume_scalar; dataset-only smoke settings, not newly approved training hyperparameters.'}
            report['configurations'][key]=summary
            before={str(f):sha(f) for dirname in ['cluster_mask','controller_mask'] for f in (scene/dirname).rglob('*.pkl')}
            np.random.seed(5)
            dataset=EmbodiedDataset(env_cfg=env,phase='train')
            assert len(dataset)==1
            with torch.no_grad():sample=dataset[0]
            data=sample['inputs'];meta=sample['meta'];n=len(entry['camera_ids'])
            assert dataset.frame_gap==10 and dataset.split_list[name]==[[0,155]]
            assert dataset.idx_mapping[0][2]==list(range(n)), 'Camera order changed'
            actualids=[]
            for cam in data['cam']:
                index=int(Path(cam.img_path).parent.name)
                assert Path(cam.img_path).stem=='0'
                actualids.append(entry['camera_ids'][index])
            assert actualids==entry['camera_ids']
            assert data['gs_aligned_frame']==0 and meta['seq_num'].item()==10
            assert meta['scene_name']==name and meta['seq_idx'].item()==0
            np.testing.assert_array_equal(data['controller_trajectory'],canonical[indices])
            assert data['controller_trajectory'].shape==(16,30,3) and data['controller_trajectory'].dtype==np.float32
            np.testing.assert_allclose(data['external'],[[0,0,-39.2]],rtol=1e-6)
            assert float(data['volume_scalar'][0,0])==512
            for level,expected in enumerate(grouping['p2c']):
                np.testing.assert_array_equal(data['p2c_mapping'][level][0],expected)
            assert len(data['cam'])==n and len(sample['gt_label'])==n
            for i in range(n):
                assert tuple(data['img'][i].shape)==(3,360,640)
                assert tuple(data['full_gt_label'][i].shape)==(16,3,360,640)
                assert tuple(sample['gt_label'][i].shape)==(16,3,360,640)
                assert tuple(data['controller_img_mask_list'][i].shape)==(16,1,360,640)
                assert torch.count_nonzero(data['controller_img_mask_list'][i]).item()==0
                assert torch.count_nonzero(data['pure_robot_img_mask_list'][i]).item()==0
                # Verify every loaded GT against the exact selected canonical files.
                for t,local in enumerate(indices):
                    rgb=np.asarray(Image.open(scene/'color'/str(i)/f'{local}.png').convert('RGB'))
                    mask=np.asarray(Image.open(scene/'mask'/str(i)/'1'/f'{local}.png'))
                    expected=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
                    assert torch.equal(data['full_gt_label'][i][t],expected)
                    expected=expected*torch.from_numpy((mask>0).copy())[None]
                    assert torch.equal(sample['gt_label'][i][t],expected)
            after={str(f):sha(f) for dirname in ['cluster_mask','controller_mask'] for f in (scene/dirname).rglob('*.pkl')}
            assert all(after[f]==h for f,h in before.items())
            assert sha(scene/'pi3/gs/point_cloud/iteration_10000/point_cloud.ply')==package['initial_gaussian']['output_sha256']
            summary.update(status='PASS',camera_ids=actualids,sampled_local_frames=indices.tolist(),
                sampled_source_frames=(indices+113).tolist(),train_source_max=int(indices[-1]+113),
                seq_num=int(meta['seq_num'].item()),seq_num_semantics='frame stride (10), not frame count',
                gs_aligned_frame=0,initial_source_frame=113,external=data['external'].tolist(),
                controller_hierarchy=[30,10,2,1],sample=describe(sample),
                gaussian_cluster_counts=[len(v[1]) for v in data['p2c_mapping']],
                cache_files={str(Path(f).relative_to(root)):{'sha256':h,'new':f not in before} for f,h in after.items()},
                actual_frames_verified_against_RGB_mask=True,controller_matches_T11_exactly=True)
            print('SAMPLE',key,json.dumps({k:summary[k] for k in ['camera_ids','sampled_source_frames','seq_num','external','controller_hierarchy']}),flush=True)
            del sample,data,dataset
            torch.cuda.empty_cache()
        assert sha(code)==report['loader_sha256']
        assert sha(trajpath)==trajectory['output']['sha256']
        for rel,info in package['generated_files'].items():assert sha(root/rel)==info['sha256']
        report['status']='PASS';report['loader_modified']=False
    except Exception:
        report['status']='FAIL';report['exception']=traceback.format_exc();print(report['exception'],flush=True)
        raise
    finally:
        a.report.write_text(json.dumps(report,indent=2)+'\n')
        print('REPORT',a.report,report['status'],flush=True)


if __name__=='__main__':main()
