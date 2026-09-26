#!/usr/bin/env python3
"""T26.5: observe unmodified train.py and real OptimizerHook for ONE batch.
No config/algorithm overrides. Slurm only. Guards precede real clipping/step.
Observer hooks collect detached diagnostics; they return original outputs.
"""
import argparse, ast, hashlib, json, os, runpy, socket, subprocess, sys, time, traceback
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
assert os.environ.get('SLURM_JOB_ID'), 'GPU allocation required'
assert not args.out.exists(), 'No overwrite/retry in an existing directory'
args.out.mkdir(parents=True)
root=Path('/data1/userdata/tcweng/projects/tcgs'); repo=root/'SoMA'
os.chdir(repo); sys.path[:0]=[str(repo),str(repo/'tools')]
import numpy as np
import torch
from mmcv.runner import OptimizerHook
from mmgs.core.runner.epoch_runner import EpochRunner
from mmgs.models.simulators.gs_simulator_embodied import GsSimulatorEmbodied
from mmgs.models.utils.normalization import Normalizer
import mmgs.datasets.embodied_dataset as dataset_module
import diff_gaussian_rasterization as raster

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def run(a): return subprocess.check_output(a,text=True).strip()
# Reuse only pure observational functions, not older audit execution paths.
helpers={
 'check_normalizer_dynamics.py':['array','digest','stats','dist','xyz','full','covstat','fstat','nstate','sanitize','nonfinite'],
 'audit_renderer_degeneracy.py':['fullcov','projection']}
for file,names in helpers.items():
 path=repo/'tools/deform360_adapter'/file
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
config=repo/'configs/SoMA/deform360_v0_stage1.py'
report=dict(task='T26.5',status='IN_PROGRESS',head=run(['git','rev-parse','HEAD']),job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),seed=5,config_sha256=sha(config),tool_sha256=sha(__file__),steps=[],native_backward=[],optimizer_steps=0,events=[],helper_hashes={f:sha(repo/'tools/deform360_adapter'/f) for f in helpers})
assert report['head']=='8b7c4507bc47eb5a033b9f707cbe898c7d76af5c'
assert not run(['git','status','--porcelain'])
assert report['config_sha256']=='e65d1c7848bd222f55c0e18050c464eec5922aef76c1bc62a06813359cca4930'
preflight=json.loads((repo/'docs/deform360/validation/t26-stage1-20260926/resumed_preflight.json').read_text())
assets=preflight['asset_hashes']
assert all(sha(p)==h for p,h in assets.items()), 'Preflight assets mismatch'
expected_traj=np.load(root/'datasets/deform360/derived/008-pink-cloth/episode_0/t11_controller_candidate_7mm/controller_points.npy',allow_pickle=False)[0:155:10]
fields=['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation']
model=None;runner=None;current=None;previous_pred=None;initial_pos=None;initial_cov=None;param_before={};gaussian_before={};buffers={};ctx='init'
class Completed(BaseException): pass

def guard(x,where):
 if torch.is_tensor(x): assert torch.isfinite(x).all().item(), where
 elif isinstance(x,dict):
  for k,v in x.items(): guard(v,where+'/'+str(k))
 elif isinstance(x,(list,tuple)):
  for i,v in enumerate(x): guard(v,where+'/'+str(i))
def values(d): return {k:float(v.detach().mean()) if torch.is_tensor(v) else v for k,v in d.items()}
def norms(): return {n:{**nstate(m),'statistics':{k:nonfinite(getattr(m,k)) for k in ['_acc_count','_num_accumulations','_acc_sum','_acc_sum_squared']}} for n,m in model.named_modules() if isinstance(m,Normalizer)}
def gradients():
 rows={n:nonfinite(p.grad) for n,p in model.named_parameters() if p.grad is not None}
 bad=[n for n,v in rows.items() if not v['finite']]
 r=dict(tensors=len(rows),finite_tensors=len(rows)-len(bad),nonfinite_tensors=len(bad),first_bad_parameter=bad[0] if bad else None,nan=sum(v['nan'] for v in rows.values()),posinf=sum(v['posinf'] for v in rows.values()),neginf=sum(v['neginf'] for v in rows.values()),unused=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None])
 if not bad:
  ns={n:float(p.grad.detach().double().norm()) for n,p in model.named_parameters() if p.grad is not None}
  r.update(global_norm=float(np.linalg.norm(list(ns.values()))),max_parameter_norm=max(ns.values()),max_parameter=max(ns,key=ns.get))
 return r
orig_reader=dataset_module.read_video_image_rgba_cv2_mask
def reader(video_dir,mask_dir,frame_range,*a,**kw):
 assert list(frame_range)==list(range(0,155,10))
 report.setdefault('video_ranges',{})[str(video_dir)]=[113+i for i in frame_range]
 return orig_reader(video_dir,mask_dir,frame_range,*a,**kw)
dataset_module.read_video_image_rgba_cv2_mask=reader
orig_run=EpochRunner.run_iter
def run_iter(self,data_batch,train_mode,**kw):
 global model,runner,param_before,ctx
 assert self.epoch==0 and self.iter==0 and train_mode and not report['steps']
 runner=self;model=self.model.module;ctx='forward'
 assert model.train_cfg.step_initial==3 and model.train_cfg.step_increase_magnitude==3
 assert len(self.data_loader)==50
 report['lr']=self.current_lr();assert all(abs(v-.000404)<1e-12 for v in report['lr'])
 report['initial_normalizers']=norms()
 for m in model.modules():
  if isinstance(m,Normalizer):
   for k in ['_acc_count','_num_accumulations','_acc_sum','_acc_sum_squared']:
    t=getattr(m,k);assert t.dtype==torch.float64 and bool((t==0).all())
 optids={id(p) for g in self.optimizer.param_groups for p in g['params']}
 for n,g in model.gs_scene_dict.items():
  assert all(id(getattr(g,k)) not in optids for k in fields)
  gaussian_before[n]={k:digest(getattr(g,k)) for k in fields}
 param_before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
 report['optimizer_class']=type(self.optimizer).__name__;assert report['optimizer_class']=='Adam' and not self.optimizer.state
 report['optimizer_groups']=[{k:v for k,v in g.items() if k!='params'} for g in self.optimizer.param_groups]
 out=orig_run(self,data_batch,train_mode,**kw)
 guard(self.outputs['loss'],'combined loss');report['losses']=dict(self.outputs['log_vars'])
 assert len(report['steps'])==3
 return out
EpochRunner.run_iter=run_iter
orig_forward=GsSimulatorEmbodied.forward_train
def forward(self,inputs,gt_label,**kw):
 assert kw.get('num_epoch')==0 and kw.get('num_iter')==0
 assert int(inputs['gs_aligned_frame'].item())==0
 assert np.array_equal(array(inputs['controller_trajectory'].squeeze(0)),expected_traj)
 assert np.allclose(array(inputs['external']),[0,0,-39.2],rtol=0,atol=2e-6)
 assert len(gt_label)==2 and all(g.shape[1]==16 for g in gt_label)
 grouping=json.loads((repo/'docs/deform360/contracts/008-pink-cloth/episode_0/controller_grouping_contract.json').read_text())
 for a,b in zip(inputs['p2c_mapping'],grouping['p2c']): assert array(a[0].reshape(-1)).tolist()==b
 report['sample']=dict(source_frames=list(range(113,268,10)),targets=[123,133,143],controller=stats(inputs['controller_trajectory']),grouping='30→10→2→1',external=array(inputs['external']).tolist(),GT_shapes=[list(g.shape) for g in gt_label],target_hashes=[[digest(g[0,j]) for g in gt_label] for j in [1,2,3]])
 return orig_forward(self,inputs,gt_label,**kw)
GsSimulatorEmbodied.forward_train=forward
orig_pre=GsSimulatorEmbodied._preprocess
def preprocess(self,*a,**kw):
 global current,initial_pos,initial_cov
 idx=len(report['steps']);assert idx<3
 if idx==0: initial_pos=a[1].detach().clone();initial_cov=a[8].detach().clone()
 assert torch.equal(a[1],initial_pos if idx==0 else previous_pred)
 assert torch.equal(a[2],initial_pos) and torch.equal(a[8],initial_cov)
 assert np.array_equal(array(a[3]),expected_traj[idx]) and np.array_equal(array(a[4]),expected_traj[idx+1])
 current=dict(step=idx+1,target_source=123+10*idx,input_source='canonical source 113' if idx==0 else 'previous prediction (detached, exact)',covariance_input='initial covariance reused by unchanged formal source',input_position_sha256=digest(a[1]),input_covariance_sha256=digest(a[8]),controller_sources=[113+10*idx,123+10*idx],rasterizer=[])
 report['steps'].append(current)
 return orig_pre(self,*a,**kw)
GsSimulatorEmbodied._preprocess=preprocess
orig_encode=GsSimulatorEmbodied.encode_decode
def encode(self,*a,**kw):
 global previous_pred
 assert kw['rollout_size']==3 and kw['register_norm'] is True
 assert [digest(x) for x in kw['gt_label']]==report['sample']['target_hashes'][current['step']-1]
 before=a[0][0].ndata['cur_state'][30:].detach().clone()
 out=orig_encode(self,*a,**kw)
 guard([out[0],out[1],out[4],out[5],out[-1]],'step outputs')
 previous_pred=out[0][30:].detach().clone();cov=out[1][30:]
 current.update(pred_pos=stats(previous_pred),pred_cov=stats(cov),displacement=dist(np.linalg.norm(array(previous_pred-before).astype(np.float64),axis=1)),displacement_from_initial=dist(np.linalg.norm(array(previous_pred-initial_pos).astype(np.float64),axis=1)),F=[fstat(f) for f in out[2]],covariance=covstat(full(array(cov))),renders=[stats(x) for x in out[4]],regularization_losses=values(out[-1]))
 assert current['covariance']['PD_count']==12861, 'Non-PD covariance; stop before backward'
 # Gross scene-scale tripwire only; actual displacement distributions are reported.
 assert current['displacement_from_initial']['quantiles']['max'] < 10*xyz(initial_pos)['bbox_diagonal'], 'Scene-scale explosion'
 return out
GsSimulatorEmbodied.encode_decode=encode
orig_loss=GsSimulatorEmbodied._encode_decode_train
def loss(self,*a,**kw):
 out=orig_loss(self,*a,**kw);guard(out,'step loss');current['loss_components']=values(out);return out
GsSimulatorEmbodied._encode_decode_train=loss
orig_native=raster._C.rasterize_gaussians
def native(*a):
 out=orig_native(*a);rr=array(out[2]);ci=len(current['rasterizer']);assert ci<2
 buffers[out[3].data_ptr()]=(current['step'],ci)
 invalid=[];dets=[];p,c,v,pr=map(array,[a[1],a[7],a[8],a[9]])
 for row in np.where(rr>0)[0]:
  z=projection(p[row],c[row],v,pr,a[10],a[11],a[13],a[12],np.float32)
  dets.append(float(z['det']))
  if not np.isfinite(z['det']) or z['det']<=0 or z['eigenvalues'][0]<=0 or not np.isfinite(z['conic']).all():invalid.append(int(row))
 rec=dict(camera=['brics-odroid-023_cam0','brics-odroid-009_cam1'][ci],positive_radius=int((rr>0).sum()),radius=dist(rr[rr>0]),render=stats(out[1]),projected_covariance_invalid_rows=invalid,projected_det=dist(dets),projection_method='CPU fp32 recomputation of audited native formula, not kernel intermediate readback')
 current['rasterizer'].append(rec);assert not invalid and rec['render']['finite']
 return out
raster._C.rasterize_gaussians=native
orig_nb=raster._C.rasterize_gaussians_backward
def native_backward(*a):
 step,ci=buffers[a[18].data_ptr()];out=orig_nb(*a)
 rec=dict(step=step,camera_index=ci,incoming_color=nonfinite(a[13]),incoming_depth=nonfinite(a[14]),returned={n:nonfinite(t) for n,t in zip(['means2D','colors','opacity','means3D','covariance','SH','scales','rotations'],out)})
 report['native_backward'].append(rec);guard(out,'renderer backward step '+str(step));return out
raster._C.rasterize_gaussians_backward=native_backward
orig_clip=OptimizerHook.clip_grads
def clip(self,params):
 global ctx
 ctx='pre-clip';report['pre_clip']=gradients();assert not report['pre_clip']['nonfinite_tensors']
 assert self.grad_clip.get('max_norm')==1.0 and self.grad_clip.get('norm_type',2)==2
 report['clip_config']=self.grad_clip.copy();ctx='clip'
 result=orig_clip(self,list(params));report['clip_return']=float(result);report['post_clip']=gradients()
 assert np.isfinite(report['clip_return']) and not report['post_clip']['nonfinite_tensors']
 assert report['post_clip']['global_norm']<=1.00001
 ctx='optimizer.step';return result
OptimizerHook.clip_grads=clip
orig_opt=OptimizerHook.after_train_iter
def opt(self,r):
 global ctx
 ctx='backward';orig_opt(self,r);report['optimizer_steps']=1;ctx='post-step'
 guard(model.state_dict(),'post-step model');guard(r.optimizer.state_dict(),'post-step optimizer')
 report['updated_parameters']=[n for n,p in model.named_parameters() if p.requires_grad and not torch.equal(p.detach(),param_before[n])]
 report['trainable_parameter_count']=len(param_before)
 report['adam']=dict(state_count=len(r.optimizer.state),steps=[float(s['step']) for s in r.optimizer.state.values()],exp_avg_finite=all(torch.isfinite(s['exp_avg']).all().item() for s in r.optimizer.state.values()),exp_avg_sq_finite=all(torch.isfinite(s['exp_avg_sq']).all().item() for s in r.optimizer.state.values()))
 assert all(s==1 for s in report['adam']['steps']) and report['updated_parameters']
 report['final_normalizers']=norms()
 assert len(report['final_normalizers'])==3
 for n in report['final_normalizers'].values():
  assert all(s['finite'] and s['dtype']=='torch.float64' for s in n['statistics'].values())
 report['model_finite']=True;report['runner_counter_at_stop']=dict(epoch=r.epoch,iter=r.iter,explanation='Stop after optimizer hook, before runner increments iter; one optimizer step completed in epoch1/iteration1.')
 raise Completed()
OptimizerHook.after_train_iter=opt
start=time.monotonic()
try:
 torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();report['gpu']=torch.cuda.get_device_name(0)
 assert 'RTX 5090' in report['gpu'] and torch.cuda.device_count()==1
 entry=repo/'tools/train.py';sys.argv=[str(entry),str(config),'--seed','5','--gpus','1','--launcher','none','--work_dir',str(args.out/'runner')]
 report['entrypoint_argv']=sys.argv.copy();runpy.run_path(str(entry),run_name='__main__')
 raise AssertionError('Training escaped one-step stop')
except Completed: report['status']='PASS'
except BaseException:
 report['status']='FAIL';report['failure_stage']=ctx;report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
finally:
 report['wall_seconds']=time.monotonic()-start
 report['memory']=dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
 report['assets_verified_count']=len(assets);report['assets_unchanged']=all(sha(p)==h for p,h in assets.items())
 report['gaussian_fields_unchanged']=all(digest(getattr(model.gs_scene_dict[n],k))==h for n,d in gaussian_before.items() for k,h in d.items()) if model else None
 report['config_unchanged']=sha(config)==report['config_sha256'];report['server_clean']=not run(['git','status','--porcelain'])
 if report['status']=='PASS':
  assert report['assets_unchanged'] and report['gaussian_fields_unchanged'] and report['config_unchanged'] and report['server_clean']
 (args.out/'report.json').write_text(json.dumps(sanitize(report),indent=2,allow_nan=False)+'\n')
 print('T26.5',report['status'],flush=True)
if report['status']!='PASS': sys.exit(2)
