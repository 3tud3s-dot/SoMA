#!/usr/bin/env python3
"""T27: official single_gpu_test cache path, independent output and readback audit.
Run only within a Slurm GPU allocation. No training or Stage-2 instantiation.
"""
import argparse, copy, hashlib, json, os, random, subprocess, sys
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID');assert not a.output.exists() and not a.report.exists()
    root=a.workspace;repo=root/'SoMA';os.chdir(repo);sys.path.insert(0,str(repo))
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint, set_random_seed
    from mmgs.datasets import build_dataset,build_dataloader
    from mmgs.models import build_simulator
    from mmgs.utils import wrap_non_distributed_model
    from mmgs.apis import single_gpu_test
    checkpoint=root/'outputs/deform360/stage1/config_a_seed5_normalizer_fix_resume_epoch2/epoch_46.pth'
    expected='d4dbea360fa033bfbf0b1ec338ce3107d49c5549675effb481f817eb672932c4'
    assert sha(checkpoint)==expected
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['meta']['epoch']==46 and ck['meta']['iter']==2300
    assert all(torch.isfinite(t).all() for t in ck['state_dict'].values())
    stats={k:v for k,v in ck['state_dict'].items() if 'normalizer.' in k}
    assert len(stats)==12 and all(v.dtype==torch.float64 for v in stats.values())
    cfgpath=repo/'configs/SoMA/deform360_v0_stage1.py';cfg=Config.fromfile(str(cfgpath))
    set_random_seed(5);cfg.data.test.env_cfg['interact']=False
    assert cfg.data.test.env_cfg.split_list=={'config_a_2cam':[[0,155]]} and cfg.data.test.env_cfg.frame_gap==10
    ds=build_dataset(cfg.data.test,default_args=dict(test_mode=True));assert len(ds)==1
    dl=build_dataloader(ds,samples_per_gpu=1,workers_per_gpu=0,num_gpus=1,dist=False,shuffle=False,round_up=False,sampler_cfg=None)
    mc=copy.deepcopy(cfg.model);mc.pretrained=None;mc.flag_save_gaussian=True;mc.data_dir=str(a.output)
    # T6/T19/T25 dataset stride; Stage-1 uses this attribute only for cache keys.
    mc.frame_gap = cfg.data.test.env_cfg.frame_gap
    model=build_simulator(mc);load_checkpoint(model,str(checkpoint),map_location='cpu',strict=True)
    assert all(torch.equal(model.state_dict()[k].cpu(),v) for k,v in ck['state_dict'].items())
    model=wrap_non_distributed_model(model,device='cuda',device_ids=[0]);model.eval();m=model.module
    before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    scene=m.gs_scene_dict['config_a_2cam'];initial_pos=scene.get_xyz.detach().cpu().clone();initial_cov=scene.get_covariance().detach().cpu().clone()
    # Observe official preprocessing boundaries without modifying inputs or outputs.
    calls=[];original_pre=m._preprocess;original_save=m.save_gaussian;saved={}
    def observe_pre(prev,cur,orig,*args,**kwargs):
        i=len(calls);assert i<15
        assert torch.equal(orig.detach().cpu(),initial_pos)
        assert torch.equal(cur.detach().cpu(),saved[i*10]['pred_pos'])
        assert torch.equal(prev.detach().cpu(),saved[max(0,(i-1)*10)]['pred_pos'])
        calls.append({'step':i+1,'source_target':113+(i+1)*10,'position_from_previous_prediction':i>0})
        return original_pre(prev,cur,orig,*args,**kwargs)
    def observe_save(name,idx,pos,cov):
        assert name=='config_a_2cam' and 0<=idx<=15 and idx*10 not in saved
        assert pos.shape==(12861,3) and cov.shape==(12861,6)
        assert pos.dtype==cov.dtype==torch.float32 and torch.isfinite(pos).all() and torch.isfinite(cov).all()
        saved[idx*10]={'pred_pos':pos.detach().cpu().clone(),'pred_cov':cov.detach().cpu().clone()}
        return original_save(name,idx,pos,cov)
    m._preprocess=observe_pre;m.save_gaussian=observe_save
    single_gpu_test(model,dl,show=False,out_dir=None)
    assert len(calls)==15 and len(saved)==16
    assert all(torch.equal(before[k],v.cpu()) for k,v in m.state_dict().items()),'Registered state or Normalizer changed'
    assert torch.equal(initial_pos,scene.get_xyz.detach().cpu()) and torch.equal(initial_cov,scene.get_covariance().detach().cpu())
    folder=a.output/'config_a_2cam/pred_stage1';assert {f.name for f in folder.iterdir()}=={f'frame_{i}.pth' for i in range(0,151,10)}
    frames=[];previous=initial_pos.double()
    for local in range(0,151,10):
        p=folder/f'frame_{local}.pth';h=sha(p);d=torch.load(p,map_location='cpu',weights_only=False)
        assert set(d)=={'pred_pos','pred_cov'}
        for key in d:assert torch.equal(d[key],saved[local][key]) and torch.isfinite(d[key]).all()
        pos=d['pred_pos'].double();cov=d['pred_cov'].double();mat=torch.zeros((len(cov),3,3),dtype=torch.float64)
        for j,(u,v) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):mat[:,u,v]=cov[:,j];mat[:,v,u]=cov[:,j]
        eig=torch.linalg.eigvalsh(mat);pd=eig[:,0]>0;condition=eig[pd,2]/eig[pd,0];disp=torch.linalg.vector_norm(pos-previous,dim=1)
        assert sha(p)==h
        frames.append(dict(cache_key=f'frame_{local}',local_frame=local,source_frame=113+local,provenance='initial canonical' if local==0 else 'predicted from previous coarse state',path=str(p),sha256=h,bytes=p.stat().st_size,fields={k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':True} for k,v in d.items()},position_min=pos.min(0).values.tolist(),position_max=pos.max(0).values.tolist(),displacement_p50_p95_p99_max=torch.quantile(disp,torch.tensor([.5,.95,.99,1.],dtype=torch.float64)).tolist(),covariance_PD_count=int(pd.sum()),covariance_PD_fraction=float(pd.double().mean()),minimum_eigenvalue=float(eig.min()),condition_p50_p99_max=torch.quantile(condition,torch.tensor([.5,.99,1.],dtype=torch.float64)).tolist()))
        previous=pos
    report=dict(task='T27',status='PASS',job_id=os.environ['SLURM_JOB_ID'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint_path=str(checkpoint),checkpoint_sha256=expected,config_sha256=sha(cfgpath),normalizer_sha256=sha(repo/'mmgs/models/utils/normalization.py'),tool_sha256=sha(__file__),seed=5,camera_ids=['brics-odroid-023_cam0','brics-odroid-009_cam1'],frame_gap=10,cache_server_path=str(folder),file_count=16,total_bytes=sum(f['bytes'] for f in frames),frames=frames,rollout_provenance=calls,registered_state_and_learned_normalizer_unchanged=True,no_future_reconstructed_Gaussian_reset=True,point_order='Official model slices controller prefix only; no Gaussian reordering; N=12861 throughout.',covariance_semantics='Each simple_test step reuses canonical initial covariance; cache stores actual predicted covariance; positions autoregressive.',official_path='mmgs.apis.single_gpu_test -> single_gpu_rollout -> GSSimulatorEmbodied.simple_test -> save_gaussian',stage2_consumer='gs_simulator_embodied_stage2.py:219-241 loads pred_stage1/frame_<local>.pth, required pred_pos/pred_cov; previous position cloned from loaded position; :729-731 indexes gs_aligned_frame',readback_exact=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print('T27_PASS',report['total_bytes'],flush=True)
if __name__=='__main__':main()
