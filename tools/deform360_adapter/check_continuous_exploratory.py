"""T31 exploratory: actual single_gpu_rollout/simple_test with read-only provenance hooks."""
import argparse,copy,hashlib,json,os,pathlib,subprocess,sys,time,traceback
import numpy as np
ROOT=pathlib.Path('/data1/userdata/tcweng/projects/tcgs');REPO=ROOT/'SoMA'
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=pathlib.Path,default=ROOT/'outputs/deform360/t31-exploratory-epoch22-v2-20260927')
parser.add_argument('--checkpoint',type=pathlib.Path,default=ROOT/'outputs/deform360/stage2/config_a_seed5/epoch_22.pth')
parser.add_argument('--checkpoint-sha256',default='31223b8ca972db83a78412040ddee9dcf7e5836f63fb11cae54b5f91aec9380c')
parser.add_argument('--expected-epoch',type=int,default=22)
parser.add_argument('--expected-iter',type=int,default=13200)
parser.add_argument('--export-artifacts',action='store_true')
args=parser.parse_args()
OUT=args.output
assert os.environ.get('SLURM_JOB_ID') and not OUT.exists()
OUT.mkdir(parents=True);os.chdir(REPO);sys.path.insert(0,str(REPO))
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint,set_random_seed
from mmgs.datasets import build_dataset,build_dataloader
from mmgs.models import build_simulator
from mmgs.utils import wrap_non_distributed_model
from mmgs.apis.test import single_gpu_rollout
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if torch.is_tensor(x):return clean(x.detach().cpu().numpy())
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return str(x)
 return x
def finite(x):
 if torch.is_tensor(x):assert torch.isfinite(x).all()
 elif isinstance(x,np.ndarray):assert np.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(list,tuple)):
  for v in x:finite(v)
def digest(x):
 if torch.is_tensor(x):x=x.detach().cpu().numpy()
 return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def stats(v):return dict(zip(['p50','p95','p99','max'],[float(np.quantile(v,q)) for q in [.5,.95,.99,1]]))
checkpoint=args.checkpoint
expected=args.checkpoint_sha256
r={'task':'T31 exploratory','status':'IN_PROGRESS','intermediate_checkpoint_only':True,'not_final_baseline':True,'T30':'INCOMPLETE/TIMEOUT','job_id':os.environ['SLURM_JOB_ID'],'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'checkpoint':{'path':str(checkpoint),'sha256':sha(checkpoint),'epoch':args.expected_epoch,'iter':args.expected_iter},'steps':[],'online_updates':[],'files_read':[]}
assert r['checkpoint']['sha256']==expected
started=time.monotonic()
def save(): (OUT/'report.json').write_text(json.dumps(clean(r),indent=2,allow_nan=False)+'\n')
ctx={};history={};templates={};row={}
writer=None
if args.export_artifacts:
 from export_continuous_artifacts import ArtifactWriter
 writer=ArtifactWriter(OUT,REPO)
original_load=torch.load
def tracked_load(f,*a,**kw):
 if isinstance(f,(str,pathlib.Path)):
  path=str(f);r['files_read'].append(path)
  if '/pred_stage1/' in path:assert pathlib.Path(path).name=='frame_0.pth','Future coarse cache forbidden'
 return original_load(f,*a,**kw)
torch.load=tracked_load
try:
 set_random_seed(5);cfg=Config.fromfile(str(REPO/'configs/SoMA/deform360_v0_stage2.py'))
 env=copy.deepcopy(cfg.data.test.env_cfg);env.split_list={'config_a_2cam':[[0,194]]};env.frame_gap=1;env.eval_start_frame=0
 mc=copy.deepcopy(cfg.model);mc.test_rollout_mode='continuous';mc.flag_save_gaussian=False
 r['protocol']={'local':[0,193],'source':[113,306],'transitions':193,'train_test_boundary':{'last_train_source':267,'first_test_source':268},'camera_ids':['brics-odroid-023_cam0','brics-odroid-009_cam1'],'eval_overrides':{'split_list':dict(env.split_list),'frame_gap':1,'eval_start_frame':0,'test_rollout_mode':'continuous'},'core_source_changed':False,'loop':'mmgs.apis.test.single_gpu_rollout','model':dict(mc),'dataset_env':dict(env)}
 ds=build_dataset(dict(type='EmbodiedDataset',phase='all',env_cfg=env))
 dl=build_dataloader(ds,samples_per_gpu=1,workers_per_gpu=0,num_gpus=1,dist=False,shuffle=False,round_up=False,sampler_cfg=None);assert len(ds)==1
 m=build_simulator(mc)
 ck=load_checkpoint(m,str(checkpoint),map_location='cpu',strict=True);assert ck['meta']['epoch']==args.expected_epoch and ck['meta']['iter']==args.expected_iter
 m=m.cuda();m.eval();before={k:digest(v) for k,v in m.state_dict().items()}
 assert set(m.scene_init_pos)=={'config_a_2cam_frame_0'}
 cache=ROOT/'datasets/deform360/derived/008-pink-cloth/episode_0/t27_stage1_cache_gap10/config_a_2cam/pred_stage1/frame_0.pth'
 c=torch.load(cache,map_location='cpu',weights_only=False)
 init=m.scene_init_pos['config_a_2cam_frame_0'].detach().cpu().numpy().copy()
 assert np.array_equal(init,c['pred_pos'].detach().numpy()) and torch.equal(m.scene_init_cov['config_a_2cam_frame_0'].cpu(),c['pred_cov'])
 r['initial_canonical_matches_T27_frame0_exact']=True
 history[0]={'pos':init,'cov':c['pred_cov'].detach().numpy()};templates[0]={'pos':digest(init),'cov':digest(c['pred_cov']),'prev':digest(init),'origin':'canonical initial'}
 if writer:writer.state(0,history[0]['pos'],history[0]['cov'],'canonical initial; exact T27 frame_0')
 traj=np.load(ROOT/'datasets/deform360/derived/008-pink-cloth/episode_0/t11_controller_candidate_7mm/controller_points.npy')
 old_pre=m._preprocess;old_enc=m.encode_decode;old_simple=m.simple_test;old_update=m.update_gaussian
 def pre(prev,cur,template,cp,cc,ct,attr,vol,cov,*a,**kw):
  t=ctx['t'];coarse=(t//10)*10
  finite([prev,cur,template,cov]);cur_np=cur.detach().cpu().numpy();prev_np=prev.detach().cpu().numpy()
  assert np.array_equal(cur_np,history[t]['pos']),('current position provenance',t)
  assert digest(template)==templates[coarse]['pos'] and digest(cov)==templates[coarse]['cov']
  expected_prev=t if t%10==0 else t-1
  assert np.array_equal(prev_np,history[expected_prev]['pos']),('previous state provenance',t)
  assert np.array_equal(cp.cpu().numpy(),traj[t]) and np.array_equal(cc.cpu().numpy(),traj[t+1])
  row.update(current_local=t,current_source=113+t,target_source=114+t,current_origin='initial canonical' if t==0 else 'model prediction',previous_state_local=expected_prev,previous_state_source=113+expected_prev,previous_origin='initial canonical' if expected_prev==0 else 'model prediction',template_local=coarse,template_origin=templates[coarse]['origin'],current_hash=digest(cur),previous_hash=digest(prev),template_hash=digest(template),covariance_origin='initial canonical' if coarse==0 else 'online model prediction',template_boundary=t>0 and t%10==0,position_continuity_exact=True,previous_state_is_current_at_template_boundary=t>0 and t%10==0)
  return old_pre(prev,cur,template,cp,cc,ct,attr,vol,cov,*a,**kw)
 def enc(*a,**kw):
  result=old_enc(*a,**kw);pos=result[0][30:];cov=result[1][30:];finite([pos,cov,result[4],result[5],result[-1]])
  assert pos.shape==(12861,3) and cov.shape==(12861,6)
  p=pos.detach().cpu().numpy();v=cov.detach().cpu().double().numpy();mat=np.zeros((len(v),3,3));mat[:,0,0]=v[:,0];mat[:,0,1]=mat[:,1,0]=v[:,1];mat[:,0,2]=mat[:,2,0]=v[:,2];mat[:,1,1]=v[:,3];mat[:,1,2]=mat[:,2,1]=v[:,4];mat[:,2,2]=v[:,5]
  eig=np.linalg.eigvalsh(mat);t=ctx['t'];history[t+1]={'pos':p.copy(),'cov':cov.detach().cpu().numpy().copy()}
  row.update(pred_pos_finite=True,pred_cov_finite=True,PD_count=int((eig[:,0]>0).sum()),N=len(p),min_eigenvalue=float(eig[:,0].min()),displacement=stats(np.linalg.norm(p-history[t]['pos'],axis=1)),from_initial=stats(np.linalg.norm(p-init,axis=1)),centroid=p.mean(0).tolist(),bbox_extent=np.ptp(p,axis=0).tolist(),renderer_finite=True,render_stats=[{'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean())} for x in result[4]],regularization=clean(result[-1]),prediction_hash=digest(p),covariance_hash=digest(cov))
  if writer:writer.prediction(t+1,pos,cov,result[4])
  return result
 def update(scene,frame,pos,cov,prev_pos=None):
  assert frame==ctx['t']+1 and frame%10==0
  assert digest(pos)==digest(history[frame]['pos']) and digest(cov)==digest(history[frame]['cov'])
  value=old_update(scene,frame,pos,cov,prev_pos)
  templates[frame]={'pos':digest(pos),'cov':digest(cov),'prev':digest(prev_pos),'origin':'online model prediction'}
  r['online_updates'].append({'local':frame,'source':113+frame,'pos_hash':digest(pos),'cov_hash':digest(cov),'prev_hash':digest(prev_pos),'prev_equals_current':torch.equal(prev_pos,pos)})
  return value
 def simple(*a,**kw):
  global row
  t=kw['pred_frame_idx'];assert t==len(r['steps']);ctx['t']=t;row={'step':t+1}
  result=old_simple(*a,**kw);finite(result)
  if writer and t==0:writer.render(0,result['cur_img_list'],'existing initial encode_decode_render_only')
  row['metrics']=clean(result['acc']);r['steps'].append(row);save();return result
 m._preprocess=pre;m.encode_decode=enc;m.update_gaussian=update;m.simple_test=simple
 wrapped=wrap_non_distributed_model(m,device='cuda',device_ids=[0])
 torch.cuda.reset_peak_memory_stats();single_gpu_rollout(wrapped,dl,show=False,out_dir=None)
 assert len(r['steps'])==193 and r['steps'][-1]['target_source']==306
 assert all(digest(v)==before[k] for k,v in m.state_dict().items()),'registered state/normalizer mutated during eval'
 assert sha(checkpoint)==expected
 r.update(status='SUCCESS',full_horizon_completed=True,registered_state_unchanged=True,future_PLY_read=False,future_cache_injection=False,position_boundary_reset=False,previous_state_boundary_reassignment='existing continuous implementation replaces prev with same-boundary prediction at each10frame template switch; no external injection, not changed',peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
 if writer:r['artifact_export']=writer.finalize(r)
except Exception:
 r.update(status='FAIL',failure_step=ctx.get('t'),traceback=traceback.format_exc());raise
finally:
 r['wall_seconds']=time.monotonic()-started;save();print('T31_EXPLORATORY',r['status'],flush=True)
