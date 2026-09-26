"""Execute unmodified tools/train.py with observational boundaries and fail-fast guards.
No config overrides, extra forward, new detach, optimizer change, or backend flags.
All outputs are server-only. Only Slurm allocation may invoke this file.
"""
import hashlib, json, math, os, runpy, socket, subprocess, sys, time, traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID'), 'Slurm GPU allocation required'
ROOT=Path('/data1/userdata/tcweng/projects/tcgs'); REPO=ROOT/'SoMA'
CONTROL=ROOT/'outputs/deform360/t26-control-20260926'
WORK=ROOT/'outputs/deform360/stage1/config_a_seed5'
os.chdir(REPO);sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'tools'))
import numpy as np
import torch
from mmcv.runner import OptimizerHook, EpochBasedRunner
from mmgs.core.runner.epoch_runner import EpochRunner
from mmgs.core.evaluation.eval_hooks import EvalHook
from mmgs.models.simulators.gs_simulator_embodied import GsSimulatorEmbodied
import mmgs.datasets.embodied_dataset as dataset_module

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(cmd):return subprocess.check_output(cmd,text=True).strip()
def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return str(x)
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 return x
started=time.monotonic();runner_ref=None;model_ref=None
ctx={'phase':'initialization','epoch':None,'iteration':None,'rollout':None,'step':None}
report={'task':'T26','status':'IN_PROGRESS','head':run(['git','rev-parse','HEAD']),
 'job_id':os.environ['SLURM_JOB_ID'],'node':socket.gethostname(),'pid':os.getpid(),'seed':5,
 'observer_sha256':sha(__file__),'config_sha256':sha(REPO/'configs/SoMA/deform360_v0_stage1.py'),
 'optimizer_steps_completed':0,'epochs_completed':0,'checkpoints':[],
 'policy':'Original train.py, frozen config, normal evaluation enabled; stop first OOM/nonfinite. No automatic recovery.',
 'command':[sys.executable,str(Path(__file__)),*sys.argv[1:]],
 'train_source_frames':list(range(113,268,10)),
 'environment_flags':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
assert report['head']=='ff02eaf9fceaf6f9bf1f53ebd615eef545ddd922'
assert not run(['git','status','--porcelain'])
assert not WORK.exists(), 'No overwrite'
assert not (CONTROL/'training_report.json').exists(), 'No rerun'
preflight=json.loads((CONTROL/'resumed_preflight.json').read_text());assert preflight['status']=='PASS'
assert preflight['config_sha256']==report['config_sha256']
events=(CONTROL/'events.jsonl').open('x',buffering=1)

def memory():
 free,total=torch.cuda.mem_get_info()
 return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
 'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
 'free':free,'total':total}
def save():
 (CONTROL/'training_report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
def event(kind,**values):
 events.write(json.dumps(clean({'event':kind,'seconds':time.monotonic()-started,**ctx,**values}),allow_nan=False)+'\n')
def bad_tensors(x,prefix=''):
 bad=[]
 if torch.is_tensor(x):
  if not torch.isfinite(x.detach()).all().item():bad.append(prefix)
 elif isinstance(x,dict):
  for k,v in x.items():bad+=bad_tensors(v,prefix+'/'+str(k))
 elif isinstance(x,(list,tuple)):
  for i,v in enumerate(x):bad+=bad_tensors(v,prefix+'/'+str(i))
 return bad
class NonfiniteError(RuntimeError):pass
def guard(x,where):
 bad=bad_tensors(x)
 if bad:
  report['first_nonfinite']={'where':where,'paths':bad,**ctx};event('nonfinite',where=where,paths=bad)
  raise NonfiniteError(where+': '+str(bad[:10]))
def loss_values(x):return {k:float(v.detach().mean()) if torch.is_tensor(v) else v for k,v in x.items()}
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
fields=['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation'];gaussian_before={}

# Record actual dataset video ranges; return original reader output unchanged.
original_reader=dataset_module.read_video_image_rgba_cv2_mask
def read_frames(video_dir,mask_dir,frame_range,*args,**kwargs):
 assert list(frame_range)==list(range(0,155,10)), ('supervision frame mismatch',frame_range)
 if 'verified_video_ranges' not in report:report['verified_video_ranges']={}
 report['verified_video_ranges'][str(video_dir)]=[113+i for i in frame_range]
 return original_reader(video_dir,mask_dir,frame_range,*args,**kwargs)
dataset_module.read_video_image_rgba_cv2_mask=read_frames

original_run_iter=EpochRunner.run_iter
def run_iter(self,data_batch,train_mode,**kwargs):
 global runner_ref,model_ref
 runner_ref=self;model_ref=self.model.module
 ctx.update(phase='train' if train_mode else 'val',epoch=self.epoch+1,iteration=self.iter+1,step=0)
 ctx['rollout']=min(3+3*self.epoch,15)
 if not gaussian_before:
  opt_ids={id(p) for g in self.optimizer.param_groups for p in g['params']}
  for name,g in model_ref.gs_scene_dict.items():
   assert all(id(getattr(g,k)) not in opt_ids for k in fields)
   gaussian_before[name]={k:tensor_hash(getattr(g,k)) for k in fields}
  assert len(self.data_loader)==50
 event('iteration_start',lr=self.current_lr(),memory=memory())
 value=original_run_iter(self,data_batch,train_mode,**kwargs)
 guard(self.outputs['loss'],'forward aggregate loss')
 event('forward_complete',losses=self.outputs['log_vars'],memory=memory())
 return value
EpochRunner.run_iter=run_iter

original_forward=GsSimulatorEmbodied.forward_train
expected_traj=np.load(ROOT/'datasets/deform360/derived/008-pink-cloth/episode_0/t11_controller_candidate_7mm/controller_points.npy',allow_pickle=False)[0:155:10]
def forward(self,inputs,gt_label,**kwargs):
 ctx['step']=0
 assert int(inputs['gs_aligned_frame'].item())==0
 assert np.array_equal(inputs['controller_trajectory'].squeeze(0).detach().cpu().numpy(),expected_traj)
 assert inputs['external'].shape[-1]==3
 assert np.allclose(inputs['external'].detach().cpu().numpy(),[0,0,-39.2],rtol=0,atol=2e-6)
 assert len(gt_label)==2 and gt_label[0].shape[1]==16
 grouping=json.loads((REPO/'docs/deform360/contracts/008-pink-cloth/episode_0/controller_grouping_contract.json').read_text())
 for actual,expected in zip(inputs['p2c_mapping'],grouping['p2c']):
  assert actual[0].reshape(-1).detach().cpu().tolist()==expected
 return original_forward(self,inputs,gt_label,**kwargs)
GsSimulatorEmbodied.forward_train=forward

original_pre=GsSimulatorEmbodied._preprocess
def preprocess(self,*args,**kwargs):
 ctx['step']+=1;ctx['operation']='graph/preprocess'
 event('rollout_step',target_source=113+10*ctx['step'],memory=memory())
 return original_pre(self,*args,**kwargs)
GsSimulatorEmbodied._preprocess=preprocess
original_encode=GsSimulatorEmbodied.encode_decode
def encode(self,*args,**kwargs):
 ctx['operation']='encode_decode/render'
 result=original_encode(self,*args,**kwargs)
 guard([result[0],result[1],result[-1]],'predicted state/covariance/regularization')
 event('step_prediction',losses=loss_values(result[-1]),memory=memory())
 return result
GsSimulatorEmbodied.encode_decode=encode
original_render_loss=GsSimulatorEmbodied._encode_decode_train
def render_loss(self,*args,**kwargs):
 ctx['operation']='image loss';result=original_render_loss(self,*args,**kwargs)
 guard(result,'render loss');event('step_render_loss',losses=loss_values(result))
 return result
GsSimulatorEmbodied._encode_decode_train=render_loss

original_clip=OptimizerHook.clip_grads
def clip(self,params):
 params=list(params);ctx['operation']='pre-clip gradient finite guard'
 guard({n:p.grad for n,p in model_ref.named_parameters() if p.grad is not None},'raw gradients before clip/optimizer')
 norm=original_clip(self,params)
 if norm is not None:guard(norm,'gradient norm');report['last_grad_norm']=float(norm)
 return norm
OptimizerHook.clip_grads=clip
original_opt_hook=OptimizerHook.after_train_iter
def opt_hook(self,runner):
 ctx['operation']='backward'
 result=original_opt_hook(self,runner)
 report['optimizer_steps_completed']+=1
 ctx['operation']='post-step finite guard'
 guard(model_ref.state_dict(),'post-step model/normalizer')
 guard(runner.optimizer.state_dict(),'post-step Adam')
 event('optimizer_step_complete',step_count=report['optimizer_steps_completed'],grad_norm=report.get('last_grad_norm'),losses=runner.outputs['log_vars'],memory=memory())
 report['last_completed_iteration']={**ctx,'losses':runner.outputs['log_vars'],'lr':runner.current_lr()};save()
 return result
OptimizerHook.after_train_iter=opt_hook

original_save=EpochBasedRunner.save_checkpoint
def checkpoint(self,*args,**kwargs):
 guard(model_ref.state_dict(),'pre-save model/normalizer');guard(self.optimizer.state_dict(),'pre-save Adam')
 result=original_save(self,*args,**kwargs)
 p=Path(self.work_dir)/('epoch_%d.pth'%(self.epoch+1))
 info={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'epoch':self.epoch+1,'iter':self.iter}
 report['checkpoints'].append(info);report['epochs_completed']=self.epoch+1
 event('checkpoint',checkpoint=info,memory=memory());save()
 return result
EpochBasedRunner.save_checkpoint=checkpoint
original_eval=EvalHook.after_train_epoch
def evaluate(self,runner):
 ctx.update(phase='train-only evaluation',step=0,operation='evaluation',rollout=15)
 result=original_eval(self,runner)
 values=dict(runner.log_buffer.output)
 for k,v in values.items():
  if isinstance(v,(float,int)) and not math.isfinite(v):raise NonfiniteError('evaluation metric '+k)
 event('evaluation_complete',metrics=values,memory=memory());return result
EvalHook.after_train_epoch=evaluate

try:
 torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
 report['gpu']=run(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'])
 assert 'RTX 5090' in torch.cuda.get_device_name(0) and torch.cuda.device_count()==1
 report['torch']=torch.__version__;report['gpu_visible_name']=torch.cuda.get_device_name(0);save()
 event('start',memory=memory())
 sys.argv[0]=str(REPO/'tools/train.py');report['official_entrypoint_argv']=sys.argv.copy()
 runpy.run_path(sys.argv[0],run_name='__main__')
 assert report['optimizer_steps_completed']==2300 and report['epochs_completed']==46
 report['status']='PASS'
except BaseException as error:
 report['status']='BLOCKED/OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else ('FAIL/nonfinite' if isinstance(error,NonfiniteError) else 'FAIL')
 report['failure_context']=ctx.copy();report['traceback']=traceback.format_exc()
 trace=[];tb=error.__traceback__
 while tb:
  f=tb.tb_frame;trace.append({'file':f.f_code.co_filename,'line':tb.tb_lineno,'function':f.f_code.co_name,'scalars':{k:f.f_locals[k] for k in ['frame_idx','num_epoch','num_iter','num_frame','rollout_size'] if k in f.f_locals and isinstance(f.f_locals[k],(int,float,str))}});tb=tb.tb_next
 report['traceback_stages']=trace
 report['failure_memory']=memory()
 report['gpu_process_memory']=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
 if runner_ref is not None:
  report['model_nonfinite_at_stop']=bad_tensors(model_ref.state_dict())
  report['optimizer_nonfinite_at_stop']=bad_tensors(runner_ref.optimizer.state_dict())
 raise
finally:
 report['wall_seconds']=time.monotonic()-started;report['memory']=memory()
 report['config_unchanged']=sha(REPO/'configs/SoMA/deform360_v0_stage1.py')==report['config_sha256']
 report['asset_hashes_unchanged']=all(sha(Path(p))==h for p,h in preflight['asset_hashes'].items())
 report['gaussian_fields_unchanged']=all(tensor_hash(getattr(model_ref.gs_scene_dict[n],k))==h for n,v in gaussian_before.items() for k,h in v.items()) if model_ref else None
 report['server_working_tree_clean']=not run(['git','status','--porcelain'])
 if report['status']=='PASS' and not all([report['config_unchanged'],report['asset_hashes_unchanged'],report['gaussian_fields_unchanged'],report['server_working_tree_clean']]):report['status']='FAIL/integrity'
 save();event('finish',status=report['status'],memory=report['memory']);events.close()
 print('T26_RESULT',json.dumps(clean({k:v for k,v in report.items() if k not in ['traceback','traceback_stages']})),flush=True)
