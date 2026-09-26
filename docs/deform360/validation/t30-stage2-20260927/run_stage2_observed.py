"""Execute unmodified tools/train.py with observational boundaries and fail-fast guards.
No config overrides, extra forward, new detach, optimizer change, or backend flags.
All outputs are server-only. Only Slurm allocation may invoke this file.
"""
import hashlib, json, math, os, runpy, socket, subprocess, sys, time, traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID'), 'Slurm GPU allocation required'
ROOT=Path('/data1/userdata/tcweng/projects/tcgs'); REPO=ROOT/'SoMA'
CONTROL=ROOT/'outputs/deform360/t30-stage2-control-20260927'
WORK=ROOT/'outputs/deform360/stage2/config_a_seed5'
INITIAL=ROOT/'outputs/deform360/stage1/config_a_seed5_normalizer_fix_resume_epoch2/epoch_46.pth'
os.chdir(REPO);sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'tools'))
import numpy as np
import torch
from mmcv.runner import OptimizerHook, EpochBasedRunner
from mmgs.core.runner.epoch_runner import EpochRunner
from mmgs.core.evaluation.eval_hooks import EvalHook
from mmgs.models.simulators.gs_simulator_embodied_stage2 import GsSimulatorEmbodiedS2 as Simulator
import mmgs.datasets.embodied_dataset as dataset_module
from mmgs.models.utils.normalization import Normalizer
import diff_gaussian_rasterization as raster
native_context={}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(cmd):return subprocess.check_output(cmd,text=True).strip()
def clean(x):
 # Logging only: promote NumPy scalars before Python type/nonfinite handling.
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,float) and not math.isfinite(x):return str(x)
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 return x
started=time.monotonic();runner_ref=None;model_ref=None
authorized_head='1d54006ed2a7a8d3f1c247232b484186f9a16363'
ctx={'phase':'initialization','epoch':None,'iteration':None,'requested_rollout':None,'effective_rollout':None,'step':None}
report={'task':'T30','status':'IN_PROGRESS','head':run(['git','rev-parse','HEAD']),
 'job_id':os.environ['SLURM_JOB_ID'],'node':socket.gethostname(),'pid':os.getpid(),'seed':5,
 'observer_sha256':sha(__file__),'config_sha256':sha(REPO/'configs/SoMA/deform360_v0_stage2.py'),
 'optimizer_steps_completed':0,'epochs_completed':0,'checkpoints':[],'epoch_summaries':[],
 'command':[sys.executable,str(Path(__file__)),*sys.argv[1:]],'work_dir':str(WORK),
 'policy':'Frozen61epochs/36600steps; this allocation limited12h; no automatic resubmit at walltime; no numerical/OOM retry',
 'initial_checkpoint_sha256':sha(INITIAL),'fresh_optimizer':True,'fresh_runner_counters':True}
assert report['head']==authorized_head and not run(['git','status','--porcelain'])
assert not WORK.exists() and not (CONTROL/'training_report.json').exists(), 'No overwrite or rerun'
preflight=json.loads((CONTROL/'preflight.json').read_text());assert preflight['status']=='PASS'
assert preflight['config_sha256']==report['config_sha256']
assert all(sha(p)==h for p,h in preflight['asset_hashes'].items())
report['normalizer_sha256']=sha(REPO/'mmgs/models/utils/normalization.py')
assert report['initial_checkpoint_sha256']=='d4dbea360fa033bfbf0b1ec338ce3107d49c5549675effb481f817eb672932c4'
report['cache_contract_sha256']=sha(REPO/'docs/deform360/contracts/008-pink-cloth/episode_0/stage1_cache_contract.json')
cache_contract=json.loads((REPO/'docs/deform360/contracts/008-pink-cloth/episode_0/stage1_cache_contract.json').read_text())['canonical']
cache_files={x['local_frame']:x for x in cache_contract['frames']}
traj=np.load(ROOT/'datasets/deform360/derived/008-pink-cloth/episode_0/t11_controller_candidate_7mm/controller_points.npy',allow_pickle=False)
windows={i:min(i+13,150) for i in range(0,150,10)}
epoch_rows=[];previous_prediction=None;initial_position=None;initial_covariance=None;epoch_started=None
seen_windows=set();cache_hashes={};epoch_start_cache={}
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


def cache_digest(model):
 return {name:{k:tensor_hash(v) for k,v in getattr(model,name).items()} for name in ['scene_init_pos','scene_init_cov','scene_init_prev_pos']}
original_reader=dataset_module.read_video_image_rgba_cv2_mask
def read_frames(video_dir,mask_dir,frame_range,*args,**kwargs):
 fr=list(frame_range);assert fr and fr[0] in windows and fr==list(range(fr[0],windows[fr[0]]))
 assert max(fr)+113<=262
 report.setdefault('verified_video_ranges',{})[str(video_dir)+'/'+str(fr[0])]=[113+i for i in fr]
 return original_reader(video_dir,mask_dir,frame_range,*args,**kwargs)
dataset_module.read_video_image_rgba_cv2_mask=read_frames

original_load=EpochBasedRunner.load_checkpoint
def load(self,filename,*a,**kw):
 assert Path(filename)==INITIAL
 result=original_load(self,filename,*a,**kw)
 ck=torch.load(INITIAL,map_location='cpu',weights_only=False)
 actual=self.model.module.state_dict()
 assert actual.keys()==ck['state_dict'].keys()
 assert all(v.dtype==ck['state_dict'][k].dtype and torch.equal(v.detach().cpu(),ck['state_dict'][k]) for k,v in actual.items())
 assert self.epoch==0 and self.iter==0 and not self.optimizer.state
 normal={k:v for k,v in actual.items() if 'normalizer.' in k}
 assert len(normal)==12 and all(v.dtype==torch.float64 for v in normal.values())
 guard(actual,'initial model/normalizer');report['initialization_exact']=True
 report['initial_counters']={'epoch':self.epoch,'iter':self.iter,'Adam_state_count':len(self.optimizer.state),'FP64_normalizer_count':12}
 save();return result
EpochBasedRunner.load_checkpoint=load

original_run_iter=EpochRunner.run_iter
def run_iter(self,data_batch,train_mode,**kwargs):
 global runner_ref,model_ref,epoch_started,cache_hashes
 runner_ref=self;model_ref=self.model.module
 ctx.update(phase='train' if train_mode else 'val',epoch=self.epoch+1,iteration=self.iter+1,step=0)
 ctx['requested_rollout']=min(3+3*self.epoch,1000)
 if not gaussian_before:
  assert self.epoch==0 and self.iter==0 and not self.optimizer.state and report.get('initialization_exact')
  assert len(self.data_loader)==600 and self._max_epochs==61
  assert not model_ref.flag_update_gaussian and not model_ref.flag_update_gaussian_train and model_ref.test_rollout_mode=='segmented'
  opt_ids={id(p) for g in self.optimizer.param_groups for p in g['params']}
  for name,g in model_ref.gs_scene_dict.items():
   assert all(id(getattr(g,k)) not in opt_ids for k in fields)
   gaussian_before[name]={k:tensor_hash(getattr(g,k)) for k in fields}
  for key,f in cache_files.items():
   d=torch.load(f['path'],map_location='cpu',weights_only=False);tag='config_a_2cam_frame_'+str(key)
   assert torch.equal(model_ref.scene_init_pos[tag].cpu(),d['pred_pos']) and torch.equal(model_ref.scene_init_cov[tag].cpu(),d['pred_cov'])
  cache_hashes=cache_digest(model_ref);report['canonical_cache_loaded_exact']=True
 if self.inner_iter==0:
  epoch_started=time.monotonic();torch.cuda.reset_peak_memory_stats();epoch_rows.clear()
 assert all(abs(v-4e-4*(.5**((self.epoch+1)//2)+.01))<1e-12 for v in self.current_lr())
 native_context.clear();event('iteration_start',lr=self.current_lr(),memory=memory())
 value=original_run_iter(self,data_batch,train_mode,**kwargs)
 guard(self.outputs['loss'],'forward aggregate loss');event('forward_complete',losses=self.outputs['log_vars'],memory=memory())
 return value
EpochRunner.run_iter=run_iter

original_forward=Simulator.forward_train
def forward(self,inputs,gt_label,**kwargs):
 ctx['step']=0
 assert kwargs['num_epoch']==runner_ref.epoch and kwargs['num_iter']==runner_ref.iter
 start=int(inputs['gs_aligned_frame'].item());assert start in windows;end=windows[start]
 ctx.update(window_start=start,cache_key='frame_'+str(start),source_start=start+113,effective_rollout=min(end-start-1,ctx['requested_rollout']))
 assert np.array_equal(inputs['controller_trajectory'].squeeze(0).cpu().numpy(),traj[start:end])
 assert np.allclose(inputs['external'].cpu().numpy(),[0,0,-39.2],rtol=0,atol=2e-6)
 assert len(gt_label)==2 and all(g.shape[1]==end-start for g in gt_label)
 if start not in seen_windows:
  seen_windows.add(start);f=cache_files[start]
  assert sha(f['path'])==f['sha256']
  event('cache_provenance',path=f['path'],sha256=f['sha256'],source_window=[start+113,end+112],future_ply_reset=False)
 report['window_starts_seen']=sorted(seen_windows)
 return original_forward(self,inputs,gt_label,**kwargs)
Simulator.forward_train=forward

original_pre=Simulator._preprocess
def preprocess(self,*args,**kwargs):
 global previous_prediction,initial_position,initial_covariance
 if ctx['phase']=='train':
  ctx['step']+=1;ctx['operation']='graph/preprocess';i=ctx['step']-1;start=ctx['window_start']
  if i==0:
   tag='config_a_2cam_frame_'+str(start)
   assert torch.equal(args[1],self.scene_init_pos[tag]) and torch.equal(args[8],self.scene_init_cov[tag])
   initial_position=args[1].detach().clone();initial_covariance=args[8].detach().clone();previous_prediction=None
  assert torch.equal(args[1],initial_position if i==0 else previous_prediction)
  assert torch.equal(args[2],initial_position) and torch.equal(args[8],initial_covariance)
  assert np.array_equal(args[3].cpu().numpy(),traj[start+i]) and np.array_equal(args[4].cpu().numpy(),traj[start+i+1])
  ctx['target_source']=113+start+ctx['step'];assert ctx['target_source']<=262
  event('rollout_step',memory=memory())
 return original_pre(self,*args,**kwargs)
Simulator._preprocess=preprocess
original_encode=Simulator.encode_decode
def encode(self,*args,**kwargs):
 global previous_prediction
 ctx['operation']='encode_decode/render';result=original_encode(self,*args,**kwargs)
 guard([result[0],result[1],result[4],result[5],result[-1]],'predicted state/covariance/render/regularization')
 if ctx['phase']=='train':previous_prediction=result[0][30:].detach().clone()
 guard({n:p for n,p in self.named_parameters() if 'normalizer' in n},'normalizer after forward')
 return result
Simulator.encode_decode=encode
original_render_loss=Simulator._encode_decode_train
def render_loss(self,*args,**kwargs):
 result=original_render_loss(self,*args,**kwargs);guard(result,'render loss');return result
Simulator._encode_decode_train=render_loss

original_clip=OptimizerHook.clip_grads
def clip(self,params):
 params=list(params);ctx['operation']='pre-clip gradient guard'
 guard({n:p.grad for n,p in model_ref.named_parameters() if p.grad is not None},'raw gradients before clip/optimizer')
 grads=[p.grad.detach() for p in params if p.grad is not None]
 report['last_gradient_tensor_count']=len(grads)
 report['last_preclip_norm']=float(torch.stack([g.double().norm() for g in grads]).norm())
 ctx['operation']='gradient clipping';norm=original_clip(self,params)
 guard(grads,'post-clip gradients');guard(norm,'clip return')
 report['last_postclip_norm']=float(torch.stack([g.double().norm() for g in grads]).norm())
 ctx['operation']='optimizer.step';return norm
OptimizerHook.clip_grads=clip
original_opt=OptimizerHook.after_train_iter
def opt(self,runner):
 ctx['operation']='backward';assert ctx['step']==ctx['effective_rollout']
 value=original_opt(self,runner);report['optimizer_steps_completed']+=1
 ctx['operation']='post-step state guard'
 guard(model_ref.state_dict(),'post-step model/normalizer');guard(runner.optimizer.state_dict(),'post-step Adam')
 assert len(runner.optimizer.state)==275 and all(float(v['step'])==runner.iter+1 for v in runner.optimizer.state.values())
 norms={n:p for n,p in model_ref.named_parameters() if 'normalizer.' in n}
 assert len(norms)==12 and all(p.dtype==torch.float64 for p in norms.values())
 row={**ctx,'losses':dict(runner.outputs['log_vars']),'lr':runner.current_lr(),'preclip_norm':report['last_preclip_norm'],'postclip_norm':report['last_postclip_norm'],'memory':memory(),'gradient_finite':True}
 epoch_rows.append(row);report['last_completed_iteration']=row
 event('optimizer_step_complete',losses=row['losses'],preclip_norm=row['preclip_norm'],postclip_norm=row['postclip_norm'],memory=row['memory']);save()
 return value
OptimizerHook.after_train_iter=opt

original_save=EpochBasedRunner.save_checkpoint
def checkpoint(self,*args,**kwargs):
 guard(model_ref.state_dict(),'pre-save state');guard(self.optimizer.state_dict(),'pre-save Adam')
 result=original_save(self,*args,**kwargs);p=Path(self.work_dir)/('epoch_%d.pth'%(self.epoch+1))
 ck=torch.load(p,map_location='cpu',weights_only=False)
 assert ck['meta']['epoch']==self.epoch+1 and ck['meta']['iter']==self.iter
 assert 'state_dict' in ck and 'optimizer' in ck
 norms={n:v for n,v in ck['state_dict'].items() if 'normalizer.' in n};assert len(norms)==12 and all(v.dtype==torch.float64 and torch.isfinite(v).all() for v in norms.values())
 info={'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'epoch':self.epoch+1,'iter':self.iter,'metadata_readable':True,'model_optimizer_present':True};del ck
 assert len(epoch_rows)==600
 from collections import Counter
 ks=sorted(set().union(*(r['losses'].keys() for r in epoch_rows)))
 summary={'epoch':self.epoch+1,'steps':600,'last_iteration':self.iter,'requested_rollout':ctx['requested_rollout'],'effective_rollout_distribution':dict(Counter(r['effective_rollout'] for r in epoch_rows)),'window_distribution':dict(Counter(r['window_start'] for r in epoch_rows)),'lr':self.current_lr(),'loss_mean':{k:float(np.mean([r['losses'][k] for r in epoch_rows if k in r['losses']])) for k in ks},'preclip_norm_min':min(r['preclip_norm'] for r in epoch_rows),'preclip_norm_max':max(r['preclip_norm'] for r in epoch_rows),'postclip_norm_max':max(r['postclip_norm'] for r in epoch_rows),'all_gradient_finite':True,'training_seconds':time.monotonic()-epoch_started,'training_memory':memory(),'checkpoint':info}
 assert all(n==40 for n in summary['window_distribution'].values()) and len(summary['window_distribution'])==15
 report['epoch_summaries'].append(summary);report['checkpoints'].append(info);report['last_good_checkpoint']=info;report['epochs_completed']=self.epoch+1
 event('checkpoint',checkpoint=info);save();return result
EpochBasedRunner.save_checkpoint=checkpoint
original_eval=EvalHook.after_train_epoch
def evaluate(self,runner):
 ctx.update(phase='train-only evaluation',step=0,operation='evaluation')
 assert cache_digest(model_ref)==cache_hashes
 value=original_eval(self,runner)
 assert cache_digest(model_ref)==cache_hashes, 'evaluation mutated canonical starts'
 values=dict(runner.log_buffer.output)
 for k,v in values.items():
  if isinstance(v,(float,int,np.number)) and not math.isfinite(v):raise NonfiniteError('evaluation metric '+k)
 guard(model_ref.state_dict(),'post-eval state')
 if report['epoch_summaries']:
  report['epoch_summaries'][-1].update(evaluation_metrics=values,evaluation_memory=memory(),epoch_with_evaluation_seconds=time.monotonic()-epoch_started,cache_unchanged_after_evaluation=True)
 event('evaluation_complete',metrics=values,memory=memory());save();return value
EvalHook.after_train_epoch=evaluate

original_native=raster._C.rasterize_gaussians
def native(*a):
 result=original_native(*a)
 guard([result[1],result[6]],'native render output')
 native_context[result[3].data_ptr()]=ctx.copy()
 return result
raster._C.rasterize_gaussians=native
original_native_backward=raster._C.rasterize_gaussians_backward
def native_backward(*a):
 origin=native_context.get(a[18].data_ptr(),{})
 report['backward_origin']=origin
 guard([a[13],a[14]],'native incoming gradients')
 result=original_native_backward(*a)
 guard(result,'native backward returned gradients')
 return result
raster._C.rasterize_gaussians_backward=native_backward
try:
 torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
 report['gpu']=run(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'])
 assert 'RTX 5090' in torch.cuda.get_device_name(0) and torch.cuda.device_count()==1
 report['torch']=torch.__version__;save()
 sys.argv[0]=str(REPO/'tools/train.py');report['official_entrypoint_argv']=sys.argv.copy()
 runpy.run_path(sys.argv[0],run_name='__main__')
 assert report['optimizer_steps_completed']==36600 and report['epochs_completed']==61
 report['status']='PASS'
except BaseException as error:
 report['status']='BLOCKED/OOM' if isinstance(error,torch.cuda.OutOfMemoryError) else ('FAIL/nonfinite' if isinstance(error,NonfiniteError) else 'FAIL')
 report['failure_context']=ctx.copy();report['traceback']=traceback.format_exc();report['failure_memory']=memory()
 report['gpu_process_memory']=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
 report['partial_epoch_iterations']=epoch_rows
 if model_ref is not None:
  report['bad_gradient_statistics']={n:{'nan':int(torch.isnan(p.grad).sum()),'posinf':int(torch.isposinf(p.grad).sum()),'neginf':int(torch.isneginf(p.grad).sum())} for n,p in model_ref.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())}
 raise
finally:
 report['wall_seconds']=time.monotonic()-started;report['memory']=memory()
 report['asset_hashes_unchanged']=all(sha(p)==h for p,h in preflight['asset_hashes'].items())
 report['gaussian_fields_unchanged']=all(tensor_hash(getattr(model_ref.gs_scene_dict[n],k))==h for n,v in gaussian_before.items() for k,h in v.items()) if model_ref else None
 report['server_working_tree_clean']=not run(['git','status','--porcelain'])
 report['config_unchanged']=sha(REPO/'configs/SoMA/deform360_v0_stage2.py')==report['config_sha256']
 if report['status']=='PASS' and not all([report['asset_hashes_unchanged'],report['gaussian_fields_unchanged'],report['server_working_tree_clean'],report['config_unchanged']]):report['status']='FAIL/integrity'
 save();event('finish',status=report['status']);events.close()
 print('T30_RESULT',report['status'],flush=True)
