#!/usr/bin/env python3
"""T28 -> T29 gated real Stage2 dataset/cache + one dense optimizer step. Slurm only."""
import argparse,copy,hashlib,json,logging,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert os.environ.get('SLURM_JOB_ID') and not a.output.exists();a.output.mkdir(parents=True)
 root=a.workspace;repo=root/'SoMA';os.chdir(repo);sys.path.insert(0,str(repo))
 import torch
 from PIL import Image
 from mmcv import Config
 from mmcv.runner import load_checkpoint,set_random_seed,build_optimizer,build_runner,OptimizerHook
 from mmgs.datasets import build_dataset,build_dataloader
 from mmgs.models import build_simulator
 from mmgs.utils import wrap_non_distributed_model
 from mmgs.core.lr_updater.hooks import HoodLrUpdaterHook
 import mmgs.core.runner
 cp=repo/'docs/deform360/contracts/008-pink-cloth/episode_0'
 r={'T28':{'status':'IN_PROGRESS'},'T29':{'status':'TODO'},'job_id':os.environ['SLURM_JOB_ID'],'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tool_sha256':sha(__file__)};task='T28'
 def save(): (a.output/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 def finite(x):
  if torch.is_tensor(x):assert torch.isfinite(x).all()
  elif isinstance(x,np.ndarray):assert np.isfinite(x).all()
  elif isinstance(x,dict):
   for v in x.values():finite(v)
  elif isinstance(x,(tuple,list)):
   for v in x:finite(v)
 def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
 try:
  s1=Config.fromfile(str(repo/'configs/SoMA/deform360_v0_stage1.py'));contract=json.loads((cp/'stage1_cache_contract.json').read_text())['canonical'];cache_root=Path(contract['cache_server_path']);scene='config_a_2cam'
  src=repo/'configs/SoMA/cloth_lift_stage2.py';text=src.read_text().replace("data_dir='data_soma_sample/cloth_lift/'",'data_dir='+repr(s1.data.test.env_cfg.data_dir+'/')).replace('"left_lift_1"',repr(scene))
  for base in ['models/gs_simulator_embodied_stage2.py','datasets/gs_soma_dataloader.py','schedules/adam_hood.py','default_runtime.py']:text=text.replace('../_base_/'+base,str(repo/'configs/_base_'/base))
  cfg=Config.fromstring(text,'.py');env=copy.deepcopy(s1.data.test.env_cfg);env.pop('_delete_',None);env.update(frame_gap=cfg.frame_gap_stage2,split_list={scene:[cfg.train_split[0]]},interact=False)
  assert env.frame_gap==1 and env.split_list[scene]==[[0,13]]
  # Only approved D360 data geometry replaces official sample-specific entries.
  mc=copy.deepcopy(cfg.model)
  for k in ['gs_scene','cluster_cfg','controller_cfg']:mc[k]=copy.deepcopy(s1.model[k])
  mc.data_dir=str(cache_root.parent.parent);mc.train_cfg=copy.deepcopy(cfg.model.train_cfg)
  # Explicit one-dense-step smoke override; this is not a Stage2 training protocol.
  mc.train_cfg.update(step_increase_interval=0,max_rollout_step=1)
  r['configuration']={'official_source':str(src),'official_sha256':sha(src),'coarse_gap':mc.frame_gap,'dense_gap':env.frame_gap,'model_dt':mc.dt,'dataset_dt':env.dt,'optimizer':dict(cfg.optimizer),'clip':dict(cfg.optimizer_config),'scheduler':dict(cfg.lr_config),'smoke_epoch_zero_based':cfg.epoch_stage1,'smoke_override':'one dense step via max_rollout_step=1/interval=0; fresh optimizer, Stage1 final learned weights; not formal resume or Stage2 protocol','model':dict(mc),'dataset_env':dict(env)}
  assets=json.loads((repo/'docs/deform360/validation/t26-stage1-20260926/resumed_preflight.json').read_text())['asset_hashes'];assets=dict(assets)
  for f in contract['frames']:assert sha(f['path'])==f['sha256'];assets[f['path']]=f['sha256']
  scene_dir=Path(env.data_dir)/scene
  for ci in range(2):
   for fr in range(13):
    for p in [scene_dir/'color'/str(ci)/f'{fr}.png',scene_dir/'mask'/str(ci)/'1'/f'{fr}.png']:assets[str(p)]=sha(p)
  assert all(sha(p)==h for p,h in assets.items())
  set_random_seed(5);ds=build_dataset(dict(type='EmbodiedDataset',phase='all',env_cfg=env));dl=build_dataloader(ds,samples_per_gpu=1,workers_per_gpu=0,num_gpus=1,dist=False,shuffle=False,round_up=False,sampler_cfg=None);assert len(ds)==1
  batch=next(iter(dl));inp=batch['inputs'];finite(inp);finite(batch['gt_label']);assert int(inp['gs_aligned_frame'])==0
  traj=np.load(root/'datasets/deform360/derived/008-pink-cloth/episode_0/t11_controller_candidate_7mm/controller_points.npy')
  assert torch.equal(inp['controller_trajectory'][0].cpu(),torch.from_numpy(traj[:13]))
  for ci in range(2):
   assert int(Path(inp['cam'][ci].img_path).parent.name)==ci
   for fr in range(13):
    rgb=np.array(Image.open(scene_dir/'color'/str(ci)/f'{fr}.png').convert('RGB'));mask=np.array(Image.open(scene_dir/'mask'/str(ci)/'1'/f'{fr}.png'))>0
    gt=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255*torch.from_numpy(mask.copy())[None]
    assert torch.equal(batch['gt_label'][ci][0,fr].cpu(),gt)
  model=build_simulator(mc);assert model.__class__.__name__=='GsSimulatorEmbodiedS2'
  key=scene+'_frame_0';cachefile=cache_root/'frame_0.pth';cached=torch.load(cachefile,map_location='cpu',weights_only=False)
  assert torch.equal(model.scene_init_pos[key].detach().cpu(),cached['pred_pos']) and torch.equal(model.scene_init_cov[key].detach().cpu(),cached['pred_cov'])
  finite(model.scene_init_pos);finite(model.scene_init_cov);finite(model.scene_init_prev_pos)
  r['T28'].update(status='PASS',cache_key=0,cache_path=str(cachefile),cache_sha256=sha(cachefile),loaded_pred_pos_hash=digest(model.scene_init_pos[key]),loaded_pred_cov_hash=digest(model.scene_init_cov[key]),local_window_inclusive=[0,12],source_window_inclusive=[113,125],dense_target_local=list(range(1,13)),dense_target_source=list(range(114,126)),controller_local_endpoints=[0,12],GT_local_endpoints=[0,12],dense_steps=12,shapes={k:list(v.shape) for k,v in cached.items()},dtypes={k:str(v.dtype) for k,v in cached.items()},camera_ids=contract['camera_ids'],cache_loaded_exact=True,no_tail_truncation=True,all_inputs_finite=True)
  save();print('T28_PASS',flush=True)
  task='T29';r['T29']['status']='IN_PROGRESS';save()
  ckpath=Path(contract['checkpoint_path']);assert sha(ckpath)==contract['checkpoint_sha256'];ck=load_checkpoint(model,str(ckpath),map_location='cpu',strict=True)
  model=wrap_non_distributed_model(model,device='cuda',device_ids=[0]);model.train();m=model.module
  finite(m.state_dict());before={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
  pre=m._preprocess;enc=m.encode_decode;calls=[]
  def observe_pre(prev,cur,template,ctrl_prev,ctrl_cur,ctrl_template,attr,vol,cov,*aa,**kw):
   assert not calls;assert torch.equal(cur.detach().cpu(),cached['pred_pos']) and torch.equal(cov.detach().cpu(),cached['pred_cov'])
   assert torch.equal(ctrl_prev.cpu(),torch.from_numpy(traj[0])) and torch.equal(ctrl_cur.cpu(),torch.from_numpy(traj[1]))
   calls.append({'current_source':113,'target_source':114,'model_input_cache_exact':True})
   return pre(prev,cur,template,ctrl_prev,ctrl_cur,ctrl_template,attr,vol,cov,*aa,**kw)
  def observe_enc(*aa,**kw):
   out=enc(*aa,**kw);finite(out[0]);finite(out[1]);finite(out[4]);finite(out[-1]);assert out[0][30:].shape==(12861,3) and out[1][30:].shape==(12861,6)
   r['T29']['forward']={'pred_pos_shape':list(out[0][30:].shape),'pred_cov_shape':list(out[1][30:].shape),'render_shapes':[list(x.shape) for x in out[4]],'finite':True};return out
  m._preprocess=observe_pre;m.encode_decode=observe_enc
  optimizer=build_optimizer(model,cfg.optimizer);runner=build_runner(dict(type='EpochRunner',max_epochs=cfg.epoch_stage1+1),default_args=dict(model=model,optimizer=optimizer,work_dir=str(a.output),logger=logging.getLogger('t28-t29')));runner._epoch=cfg.epoch_stage1
  lr=HoodLrUpdaterHook(**{k:v for k,v in cfg.lr_config.items() if k!='policy'});lr.before_run(runner);lr.before_train_epoch(runner)
  class GuardedHook(OptimizerHook):
   def clip_grads(self,params):
    params=list(params);gs=[p.grad for p in params if p.grad is not None];counts={'NaN':sum(int(torch.isnan(g).sum()) for g in gs),'+Inf':sum(int(torch.isposinf(g).sum()) for g in gs),'-Inf':sum(int(torch.isneginf(g).sum()) for g in gs)}
    r['T29']['gradients']={'count':len(gs),'finite_count':sum(bool(torch.isfinite(g).all()) for g in gs),'counts':counts,'first_bad_parameter':next((n for n,p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()),None)}
    assert not any(counts.values());norm=lambda:sum(float(g.detach().double().square().sum()) for g in gs)**.5
    r['T29']['preclip_norm']=norm();ret=super().clip_grads(params);r['T29']['clip_return']=float(ret);r['T29']['postclip_norm']=norm();finite(gs);return ret
  runner.run_iter(batch,train_mode=True);finite(runner.outputs['loss']);r['T29']['losses']=runner.outputs['log_vars'];r['T29']['LR']=[g['lr'] for g in optimizer.param_groups]
  GuardedHook(**cfg.optimizer_config).after_train_iter(runner)
  finite(m.state_dict());finite(optimizer.state_dict());finite(m.scene_init_pos);finite(m.scene_init_cov);finite(m.scene_init_prev_pos)
  changed=[n for n,p in m.named_parameters() if n in before and not torch.equal(p,before[n])];assert changed and len(calls)==1
  assert all(sha(p)==h for p,h in assets.items())
  r['T29'].update(status='PASS',step_count=1,updated_parameter_count=len(changed),model_optimizer_runtime_finite=True,cache_and_source_hashes_unchanged=True,provenance=calls,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),normalizer_dtypes={n:str(p.dtype) for n,p in m.named_parameters() if 'normalizer.' in n})
 except Exception:
  r[task]['status']='FAIL';r[task]['exception']=traceback.format_exc();print(r[task]['exception'],flush=True);raise
 finally:save()
if __name__=='__main__':main()
