#!/usr/bin/env python3
"""T26.1 fresh-process backward probes on the unmodified formal train.py path.
Only diagnostic horizon/selected backward objective vary. Never clip/update/save.
"""
import argparse, contextlib, hashlib, json, math, os, runpy, sys, time, traceback
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tag',required=True)
ap.add_argument('--rollout',type=int,required=True);ap.add_argument('--step',type=int,default=0)
ap.add_argument('--component',choices=['combined','momentum','l2','ssim'],default='combined')
ap.add_argument('--anomaly',action='store_true');ap.add_argument('--native-trace',action='store_true');args=ap.parse_args()
assert os.environ.get('SLURM_JOB_ID') and args.rollout in [1,2,3]
assert not args.out.exists();args.out.mkdir(parents=True)
root=Path('/data1/userdata/tcweng/projects/tcgs');repo=root/'SoMA'
os.chdir(repo);sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'tools'))
import numpy as np
import torch
from mmcv.runner import OptimizerHook
from mmgs.core.runner.epoch_runner import EpochRunner
from mmgs.models.simulators.gs_simulator_embodied import GsSimulatorEmbodied
from mmgs.models.heads.acc_decoder import AccDecoder

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def stat(t):
 if t is None:return None
 t=t.detach();finite=torch.isfinite(t);values=t[finite].double()
 return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'finite':bool(finite.all()),
 'nan':int(torch.isnan(t).sum()),'posinf':int(torch.isposinf(t).sum()),'neginf':int(torch.isneginf(t).sum()),
 'finite_abs_max':float(values.abs().max()) if values.numel() else None,
 'finite_abs_min':float(values.abs().min()) if values.numel() else None}
def nested_hash(x):
 if torch.is_tensor(x):return {'sha256':digest(x),'shape':list(x.shape),'dtype':str(x.dtype)}
 if isinstance(x,np.ndarray):return {'sha256':hashlib.sha256(x.tobytes()).hexdigest(),'shape':list(x.shape),'dtype':str(x.dtype)}
 if isinstance(x,dict):return {k:nested_hash(v) for k,v in x.items() if k!='cam'}
 if isinstance(x,(list,tuple)):return [nested_hash(v) for v in x]
 if isinstance(x,(str,float,int,bool)) or x is None:return x
 return str(type(x))
def safe(x):
 if isinstance(x,float) and not math.isfinite(x):return str(x)
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 return x
report={'task':'T26.1','tag':args.tag,'rollout':args.rollout,'selected_step':args.step,'component':args.component,'anomaly':args.anomaly,'native_trace':args.native_trace,
 'job_id':os.environ['SLURM_JOB_ID'],'seed':5,'config_sha256':sha(repo/'configs/SoMA/deform360_v0_stage1.py'),
 'tool_sha256':sha(__file__),'optimizer_step':False,'gradient_clipping':False,'formal_training':False,'steps':[],'backward_boundaries':[]}
start=time.monotonic();step_losses=[];model=None;step=0
class ProbeComplete(BaseException):pass

def tap(t,name):
 if (args.anomaly or args.native_trace) and torch.is_tensor(t) and t.requires_grad:
  report['backward_boundaries'].append({'event':'forward','name':name,**stat(t)})
  def hook(g):report['backward_boundaries'].append({'event':'backward','name':name,**stat(g)})
  t.register_hook(hook)

# Installed native boundary targeted only after anomaly identified rasterizer backward.
if args.native_trace:
 import diff_gaussian_rasterization as raster
 report['native_source']={'python_path':raster.__file__,'python_sha256':sha(raster.__file__),'extension_path':raster._C.__file__,'extension_sha256':sha(raster._C.__file__)}
 report['native_calls']=[];camera_by_view={}
 native_forward=raster._C.rasterize_gaussians
 native_backward=raster._C.rasterize_gaussians_backward
 def desc(x):
  if torch.is_tensor(x):
   if x.dtype==torch.uint8:return {'shape':list(x.shape),'dtype':str(x.dtype),'opaque_buffer':True}
   return stat(x)
  return x
 def forward_native(*a):
  out=native_forward(*a)
  key=digest(a[8]);camera_by_view.setdefault(key,len(camera_by_view))
  report['native_calls'].append({'direction':'forward','camera_index':camera_by_view[key],
    'args':{k:desc(v) for k,v in zip(['background','means3D','colors','opacity','scales','rotations','scale_modifier','covariance','viewmatrix','projmatrix','tanfovx','tanfovy','height','width','SH','SHdegree','camera_center','prefiltered','antialiasing','debug'],a)},
    'num_rendered':out[0],'image':stat(out[1]),'radii':stat(out[2]),'positive_radii_count':int((out[2]>0).sum()),'depth':stat(out[6])})
  return out
 def backward_native(*a):
  record={'direction':'backward','camera_index':camera_by_view[digest(a[9])],
   'args':{k:desc(v) for k,v in zip(['background','means3D','radii','colors','opacity','scales','rotations','scale_modifier','covariance','viewmatrix','projmatrix','tanfovx','tanfovy','grad_out_color','grad_out_depth','SH','SHdegree','camera_center','geomBuffer','num_rendered','binningBuffer','imgBuffer','antialiasing','debug'],a)}}
  out=native_backward(*a)
  record['returned_gradients']={k:stat(v) for k,v in zip(['means2D','colors','opacity','means3D','covariance','SH','scales','rotations'],out)}
  record['first_bad_rows']={k:torch.nonzero(~torch.isfinite(v).reshape(v.shape[0],-1).all(-1)).flatten()[:20].cpu().tolist() for k,v in zip(['means2D','colors','opacity','means3D','covariance','SH','scales','rotations'],out) if v.numel() and not torch.isfinite(v).all()}
  report['native_calls'].append(record)
  return out
 raster._C.rasterize_gaussians=forward_native
 raster._C.rasterize_gaussians_backward=backward_native

original_run=EpochRunner.run_iter
def run_iter(self,data_batch,train_mode,**kwargs):
 global model
 assert self.epoch==0 and self.iter==0 and train_mode
 model=self.model.module
 report['initial_state']={k:{'sha256':digest(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in model.state_dict().items()}
 report['input_hashes']=nested_hash(data_batch)
 report['camera_inputs']=[{'img_path':str(c.img_path),'world_view':nested_hash(c.world_view_transform),'projection':nested_hash(c.full_proj_transform)} for c in data_batch['inputs']['cam']]
 report['initial_Gaussian']={n:{k:nested_hash(getattr(g,k)) for k in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']} for n,g in model.gs_scene_dict.items()}
 assert len(self.data_loader)==50
 # In-memory diagnostic horizon only; frozen file and all other settings unchanged.
 model.train_cfg.step_initial=args.rollout
 return original_run(self,data_batch,train_mode,**kwargs)
EpochRunner.run_iter=run_iter
original_pre=GsSimulatorEmbodied._preprocess
def preprocess(self,*a,**kw):
 global step
 step+=1;assert step<=args.rollout
 return original_pre(self,*a,**kw)
GsSimulatorEmbodied._preprocess=preprocess
original_encode=GsSimulatorEmbodied.encode_decode
def encode(self,*a,**kw):
 out=original_encode(self,*a,**kw)
 tensors=[out[0],out[1],*out[2],*out[4],*out[5],*out[-1].values()]
 assert all(bool(torch.isfinite(t.detach()).all()) for t in tensors)
 step_losses.append(dict(out[-1]))
 report['steps'].append({'step':step,'input_source':113+10*(step-1),'target_source':113+10*step,'all_observed_outputs_finite':True,'pred_pos':stat(out[0]),'pred_cov':stat(out[1])})
 tap(out[0],f'step{step}.pred_pos');tap(out[1],f'step{step}.pred_cov')
 return out
GsSimulatorEmbodied.encode_decode=encode
original_loss=GsSimulatorEmbodied._encode_decode_train
def loss(self,*a,**kw):
 out=original_loss(self,*a,**kw);step_losses[-1].update(out)
 assert all(bool(torch.isfinite(t.detach()).all()) for t in out.values())
 report['steps'][-1]['losses']={k:float(v.detach().mean()) for k,v in step_losses[-1].items()}
 return out
GsSimulatorEmbodied._encode_decode_train=loss
original_render=AccDecoder.pre_render
def render(self,*a,**kw):
 if args.anomaly or args.native_trace:
  for key in ['pos','cov3D_precomp','opacity']:
   if key in kw:tap(kw[key],f'step{step}.cam{a[0].img_path}.renderer_input.{key}')
 out=original_render(self,*a,**kw)
 if args.anomaly or args.native_trace:tap(out[0],f'step{step}.cam{a[0].img_path}.render_output')
 return out
AccDecoder.pre_render=render

def backward_probe(self,runner):
 report['full_total_loss']=float(runner.outputs['loss'].detach())
 report['full_components']=dict(runner.outputs['log_vars'])
 if args.step:
  candidates={k:v for k,v in step_losses[args.step-1].items() if k.startswith('decode.loss')}
  if args.component!='combined':
   key={'momentum':'decode.loss_mse_momentum','l2':'decode.loss_l2_render','ssim':'decode.loss_ssim_render'}[args.component]
   candidates={key:candidates[key]}
  chosen=sum(v.mean() for v in candidates.values())
  report['selected_loss_keys']=list(candidates)
 else:chosen=runner.outputs['loss'];report['selected_loss_keys']='formal aggregate loss'
 report['selected_loss']=float(chosen.detach());assert torch.isfinite(chosen)
 assert all(p.grad is None for p in model.parameters())
 chosen.backward()
 gradients={};missing=[]
 for n,p in model.named_parameters():
  if not p.requires_grad:continue
  if p.grad is None:missing.append(n)
  else:gradients[n]=stat(p.grad)
 bad=[n for n,v in gradients.items() if not v['finite']]
 counts={key:sum(v[key] for v in gradients.values()) for key in ['nan','posinf','neginf']}
 report['gradients']={'parameters':gradients,'missing':missing,'finite_tensor_count':len(gradients)-len(bad),'nonfinite_tensor_count':len(bad),'first_bad_parameter':bad[0] if bad else None,**counts}
 if not bad:
  report['gradients']['global_norm']=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None))
 report['result']='NONFINITE' if bad else 'FINITE'
 raise ProbeComplete()
OptimizerHook.after_train_iter=backward_probe
try:
 torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
 report['gpu']=torch.cuda.get_device_name(0)
 sys.argv=[str(repo/'tools/train.py'),'configs/SoMA/deform360_v0_stage1.py','--seed','5','--gpus','1','--launcher','none','--work_dir',str(args.out/'runner')]
 with (torch.autograd.detect_anomaly(check_nan=True) if args.anomaly else contextlib.nullcontext()):
  runpy.run_path(sys.argv[0],run_name='__main__')
except ProbeComplete:pass
except BaseException:
 report['result']='EXCEPTION';report['traceback']=traceback.format_exc()
 print(report['traceback'],flush=True)
finally:
 report['wall_seconds']=time.monotonic()-start
 report['memory']={'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
 if model is not None:report['final_model_finite']=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
 (args.out/'report.json').write_text(json.dumps(safe(report),indent=2)+'\n')
 print('PROBE_RESULT',args.tag,report.get('result'),report.get('gradients',{}).get('nonfinite_tensor_count'),flush=True)
if report.get('result')=='EXCEPTION':sys.exit(2)
