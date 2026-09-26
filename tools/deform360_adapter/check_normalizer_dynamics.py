#!/usr/bin/env python3
"""T26.4 regression observer; optional combined backward, never optimizer.step.
Modes: formal Deform360, existing official sample, historical T22 harness stopped
at the first backward invocation (before executing it). No tensor dumps.
"""
import argparse,hashlib,json,os,runpy,sys,time,traceback
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['d360','official','t22'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fine',action='store_true');ap.add_argument('--backward',action='store_true');args=ap.parse_args();assert not args.backward or args.mode=='d360'
assert os.environ.get('SLURM_JOB_ID') and not args.out.exists();args.out.mkdir(parents=True)
root=Path('/data1/userdata/tcweng/projects/tcgs');repo=root/'SoMA';os.chdir(repo);sys.path[:0]=[str(repo),str(repo/'tools')]
import numpy as np
import torch
from mmcv.runner import OptimizerHook
from mmgs.core.runner.epoch_runner import EpochRunner
from mmgs.models.simulators.gs_simulator_embodied import GsSimulatorEmbodied
from mmgs.models.heads.acc_decoder import AccDecoder
from mmgs.models.utils.normalization import Normalizer
import mmgs.utils.transformation_utils as trans
import diff_gaussian_rasterization as raster

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array(t):return t.detach().cpu().numpy()
def digest(t):return hashlib.sha256(array(t).copy().tobytes()).hexdigest()
def stats(t):
 a=array(t) if torch.is_tensor(t) else np.asarray(t);v=a.astype(np.float64).reshape(-1);f=v[np.isfinite(v)]
 return dict(shape=list(a.shape),dtype=str(a.dtype),finite=bool(np.isfinite(v).all()),abs_max=float(np.max(abs(f))) if len(f) else None,mean=float(f.mean()) if len(f) else None,std=float(f.std()) if len(f) else None,p99_abs=float(np.quantile(abs(f),.99)) if len(f) else None)
def dist(t):
 v=np.asarray(t,dtype=np.float64).reshape(-1);f=v[np.isfinite(v)]
 if not len(f):return dict(count=0)
 return dict(count=len(v),finite=bool(np.isfinite(v).all()),quantiles=dict(zip(['min','p01','p50','p90','p95','p99','max'],np.quantile(f,[0,.01,.5,.9,.95,.99,1]).tolist())))
def xyz(p):
 a=array(p).astype(np.float64);return dict(shape=list(a.shape),min=a.min(0).tolist(),max=a.max(0).tolist(),std=a.std(0).tolist(),bbox_diagonal=float(np.linalg.norm(a.max(0)-a.min(0))))
def full(c):
 a=np.asarray(c);return a[:,[0,1,2,1,3,4,2,4,5]].reshape(-1,3,3)
def covstat(m):
 a=np.asarray(m,dtype=np.float64);up=a[:,[0,0,0,1,1,2],[0,1,2,1,2,2]];u=full(up);e=np.linalg.eigvalsh(u)
 with np.errstate(all='ignore'):co=np.max(abs(e),1)/np.min(abs(e),1)
 return dict(count=len(e),PD_count=int((e[:,0]>0).sum()),PSD_count=int((e[:,0]>=0).sum()),min_eigen=dist(e[:,0]),max_eigen=dist(e[:,2]),condition=dist(co),magnitude=stats(a),asymmetry_absmax=float(abs(a-a.transpose(0,2,1)).max()),eigen_convention='upper-triangle mirrored as actual final packed representation',symmetric_part_PD_count=int((np.linalg.eigvalsh((a+a.transpose(0,2,1))/2)[:,0]>0).sum()))
def fstat(f):
 a=array(f).astype(np.float64);s=np.linalg.svd(a,compute_uv=False)
 return dict(shape=list(a.shape),singular_values=dist(s),max_singular=dist(s[:,0]),min_singular=dist(s[:,-1]),determinant=dist(np.linalg.det(a)))
def nstate(m):return dict(training=m.training,is_training=m.is_training,count=array(m._acc_count).tolist(),accumulations=array(m._num_accumulations).tolist(),mean=array(m._mean()).tolist(),std=array(m._std_with_epsilon()).tolist(),sum=array(m._acc_sum).tolist(),sum_squared=array(m._acc_sum_squared).tolist(),epsilon=float(m._std_epsilon),force_nonorm=m.force_nonorm)
def state(m):return {k:digest(v) for k,v in m.state_dict().items()}
def sanitize(x):
 if isinstance(x,np.generic):return sanitize(x.item())
 if isinstance(x,float) and not np.isfinite(x):return str(x)
 if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [sanitize(v) for v in x]
 return x
report=dict(task='T26.4',mode=args.mode,fine=args.fine,job=os.environ['SLURM_JOB_ID'],seed=5,tool_hash=sha(__file__),backward=False,optimizer_step=False,events=[],normalizers=[],passes=[],init_weights_calls=[])
model=None;current=None;level=None;graph_ids={};graphs=None;initial=None;cov64=None;row_id=None
class Done(BaseException):pass

def record(name,t):report['events'].append(dict(pass_index=len(report['passes']),level=level,name=name,**stats(t)))
def graphfields(g):return {k:stats(g.ndata[k]) for k in ['prev_state','cur_state','template_state','anchor_prev_state','anchor_cur_state','anchor_next_state','anchor_template_state','external','attr'] if k in g.ndata}
def rowfields(g,i):return {k:array(g.ndata[k][i]).tolist() for k in ['cur_state','prev_state','template_state','anchor_next_state','anchor_template_state','hie_anchor_next_state','hie_anchor_template_state','hie_pred_dg_mat','pred_pos','pred_dg','pred_dg_mat'] if k in g.ndata}
def norms(m):return {n:nstate(v) for n,v in m.named_modules() if isinstance(v,Normalizer)}

orig_init=GsSimulatorEmbodied.__init__
def construct(self,*a,**kw):
 global model
 orig_init(self,*a,**kw);model=self
 report['construction_state']=state(self);report['construction_normalizers']=norms(self)
 report['runtime_parameters']=dict(dt=self.dt,backbone_dt=self.backbone.dt,decoder_dt=self.decode_head.dt,decoder_type=type(self.decode_head).__name__,backbone_type=type(self.backbone).__name__,fix_bug=self.fix_bug,forward_last_layer=self.forward_last_layer)
 for name,m in self.named_modules():
  if isinstance(m,Normalizer):
   def pre(mod,a,n=name):report['normalizers'].append(dict(pass_index=len(report['passes']),level=level,name=n,before=nstate(mod),input=stats(a[0])))
   def post(mod,a,o):
    entry=report['normalizers'][-1];entry.update(after=nstate(mod),output=stats(o))
    if args.fine:
     x=array(a[0]).astype(np.float64)
     variance=mod._acc_sum_squared / mod._acc_count - mod._mean().pow(2)
     entry['variance_audit']=dict(statistics_dtype=str(mod._acc_sum.dtype),actual_statistics_raw_variance=array(variance).tolist(),current_batch_centered_variance64=x.var(axis=0).tolist(),current_batch_mean64=x.mean(axis=0).tolist(),current_batch_centered_std64=x.std(axis=0).tolist(),max_centered_residual64=abs(x-x.mean(axis=0)).max(axis=0).tolist())
     if x.shape[0]<=16:entry['variance_audit']['small_coarse_input']=x.tolist()

   m.register_forward_pre_hook(pre);m.register_forward_hook(post)
 for name in ['backbone.node_encoder','backbone.edge_encoder','decode_head.dynamic_proj']:
  m=dict(self.named_modules())[name]
  m.register_forward_pre_hook(lambda m,a,n=name:record(n+'.input',a[0]))
  m.register_forward_hook(lambda m,a,o,n=name:record(n+'.output',o))
  if name=='decode_head.dynamic_proj':
   m.register_forward_hook(lambda m,a,o:record('decoder.raw_log_scales',o[:,4:7]))
 enc=self.backbone.encoder
 enc.register_forward_pre_hook(lambda m,a:record('message_passing.input',a[0].ndata['out_node']))
 enc.register_forward_hook(lambda m,a,o:record('message_passing.output',o.ndata['out_node']))
 # Fine tracing is restricted to the normalization module identified by coarse trace.
GsSimulatorEmbodied.__init__=construct
orig_iw=GsSimulatorEmbodied.init_weights
def iw(self,*a,**kw):
 before=state(self);out=orig_iw(self,*a,**kw);after=state(self)
 report['init_weights_calls'].append(dict(changed_keys=[k for k in before if before[k]!=after[k]],before=before,after=after));return out
GsSimulatorEmbodied.init_weights=iw
orig_run=EpochRunner.run_iter
def run_iter(self,data_batch,train_mode,**kw):
 assert self.epoch==0 and self.iter==0 and train_mode
 report['formal_initial_state']=state(self.model.module)
 report['dataset_summary']={k:stats(v) for k,v in data_batch['inputs'].items() if torch.is_tensor(v)}
 report['sample_meta']={k:(array(v).tolist() if torch.is_tensor(v) else str(v)) for k,v in data_batch['meta'].items()}
 self.model.module.train_cfg.step_initial=1
 out=orig_run(self,data_batch,train_mode,**kw)
 report['losses']=dict(self.outputs['log_vars'])
 loss=self.outputs['loss'];assert torch.isfinite(loss)
 # Check registration/order against the pre-patch real T24 checkpoint and optimizer.
 oldpath=root/'outputs/deform360/t24-finite-recovery-v3-20260926/epoch_1.pth'
 old=torch.load(oldpath,map_location='cpu',weights_only=False)
 names=list(dict(model.named_parameters()));trainable=[n for n,p in model.named_parameters() if p.requires_grad]
 reference=json.loads((repo/'docs/deform360/validation/t24-finite-recovery-20260926/save_report.json').read_text())['optimizer_parameter_names']
 assert names==list(old['state_dict']) and trainable==reference
 probe=torch.optim.Adam([p for p in model.parameters() if p.requires_grad]);probe.load_state_dict(old['optimizer'])
 matched=0
 for group in probe.param_groups:
  for p in group['params']:
   st=probe.state[p]
   if st:
    assert st['exp_avg'].shape==p.shape and st['exp_avg_sq'].shape==p.shape;matched+=1
 param_names={id(p):n for n,p in model.named_parameters()}
 report['optimizer_mapping']=dict(all_registered_names_same_as_old_checkpoint=True,trainable_order_same_as_T24=True,trainable_count=len(trainable),loaded_Adam_states=matched,formal_group_names=[[param_names[id(p)] for p in g['params']] for g in self.optimizer.param_groups],no_step=True)
 del probe,old
 if args.backward:
  assert all(p.grad is None for p in model.parameters())
  original_tensor_backward(loss)
  grads={n:nonfinite(p.grad) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
  bad=[n for n,v in grads.items() if not v['finite']]
  report['gradient_check']=dict(tensors=grads,tensor_count=len(grads),nonfinite_names=bad,nan=sum(v['nan'] for v in grads.values()),posinf=sum(v['posinf'] for v in grads.values()),neginf=sum(v['neginf'] for v in grads.values()),global_norm=float(torch.sqrt(sum(p.grad.detach().double().square().sum() for p in model.parameters() if p.grad is not None))) if not bad else None,without_gradient=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None])
  report['backward']=True
  assert len(grads)==275 and not bad and all(v['finite'] for rec in report['native_backward'] for v in rec['returned'].values())
 raise Done()
EpochRunner.run_iter=run_iter
orig_pre=GsSimulatorEmbodied._preprocess
def preprocess(self,*a,**kw):
 global graph_ids,row_id
 out=orig_pre(self,*a,**kw);gs=out[0];graph_ids={id(g):i for i,g in enumerate(gs)}
 cp0,cp1=a[3],a[4];pos=a[1]
 spatial=dict(position=xyz(pos),previous_equals_current=bool(torch.equal(a[0],a[1])),controller_displacement=dist(np.linalg.norm(array(cp1-cp0),axis=1)),controller_world=xyz(cp0),gravity=stats(a[9]),external_values=array(a[9]).tolist())
 from scipy.spatial import cKDTree
 pa=array(pos);ca=array(cp0);tree=cKDTree(pa);spatial['Gaussian_nearest_neighbor_distance']=dist(tree.query(pa,k=2)[0][:,1]);spatial['controller_nearest_object_distance']=dist(tree.query(ca)[0])
 spatial['graphs']=[dict(level=i,nodes=g.num_nodes(),edges=g.num_edges(),fields=graphfields(g),edge_displacement=stats(g.ndata['cur_state'][g.edges()[1]]-g.ndata['cur_state'][g.edges()[0]]),edge_length=dist(np.linalg.norm(array(g.ndata['cur_state'][g.edges()[1]]-g.ndata['cur_state'][g.edges()[0]]),axis=1))) for i,g in enumerate(gs)]
 report.setdefault('preprocess',[]).append(spatial)
 # p2c arrays already use combined controller/object IDs, as original preprocess.
 maps=kw.get('p2c_mapping',a[10] if len(a)>10 else None)
 rid=3648+len(cp0);ids=[rid]
 for mapping in maps:
  merged=torch.cat(mapping);rid=int(merged[rid]);ids.append(rid)
 row_id=ids;report['row_ancestor_ids']=ids
 return out
GsSimulatorEmbodied._preprocess=preprocess
orig_encode=GsSimulatorEmbodied.encode_decode
def encode(self,*a,**kw):
 global current,graphs,initial,cov64
 graphs=a[0];initial=graphs[0].ndata['cur_state'].detach().clone();cov64=None
 current=dict(number=len(report['passes'])+1,normalizers_before=norms(self),hierarchy=[],covariance=[],row_stages=[],initial_object=xyz(initial[30:]))
 report['passes'].append(current)
 out=orig_encode(self,*a,**kw)
 p=out[0][30:];c=out[1][30:];delta=array(p-initial[30:]).astype(np.float64);dn=np.linalg.norm(delta,axis=1);bbox=current['initial_object']['bbox_diagonal']
 current.update(predicted_object=xyz(p),displacement=dist(dn),displacement_over_bbox=dist(dn/bbox),displacement_exceed_counts={str(v):int((dn>v).sum()) for v in [.1,.5,1,5]},row3648=dict(displacement=float(dn[3648]),percentile_le=float(np.mean(dn<=dn[3648])*100),initial=array(initial[30+3648]).tolist(),predicted=array(p[3648]).tolist()),predicted_covariance=covstat(full(array(c))),normalizers_after=norms(self),render=[stats(x) for x in out[4]])
 if args.mode=='d360':
  from PIL import Image,ImageDraw
  for i,im in enumerate(out[4]):
   pred=(np.clip(array(im).transpose(1,2,0),0,1)*255).astype(np.uint8)
   gt=kw.get('gt_label')
   if gt is not None:
    target=(np.clip(array(gt[i]).transpose(1,2,0),0,1)*255).astype(np.uint8)
    canvas=np.concatenate([target,pred,((target.astype(float)+pred)/2).astype(np.uint8)],axis=1)
   else:canvas=pred
   image=Image.fromarray(canvas);draw=ImageDraw.Draw(image);draw.text((8,8),'target 123 | prediction | overlay',fill='white');image.save(args.out/f'camera_{i}_review.png')
 return out
GsSimulatorEmbodied.encode_decode=encode
orig_extract=GsSimulatorEmbodied.extract_feat
def extract(self,bb,g,cg):
 global level
 level=graph_ids[id(g)]
 # Leaf after higher-level broadcasts, before this level is computed.
 current['row_stages'].append(dict(event='before_level',level=level,row=rowfields(graphs[0],3678)))
 d=g.ndata;dt=bb.dt
 acc=-(d['anchor_next_state']-2*d['anchor_cur_state']+d['anchor_prev_state'])/(dt**2)+d['external']
 vel=((d['cur_state']-d['anchor_next_state'])-(d['prev_state']-d['anchor_prev_state']))/dt
 current['hierarchy'].append(dict(level=level,fields_before=graphfields(g),row_ancestor_before=rowfields(g,row_id[level]),actual_anchor_acceleration=stats(acc),actual_relative_velocity=stats(vel),dt=dt))
 return orig_extract(self,bb,g,cg)
GsSimulatorEmbodied.extract_feat=extract
orig_pred=AccDecoder.pre_predict
def predict(self,g,**kw):
 i=row_id[level];before=rowfields(g,i);out=orig_pred(self,g,**kw)
 current['hierarchy'][-1].update(predicted_position=stats(out[0]),F=fstat(out[2]),deformation_components=stats(out[1]),normalized_scales=dist(array(out[1][:,4:7])),row_ancestor_after=rowfields(g,i))
 current['row_stages'].append(dict(event='decoder_ancestor',level=level,before=before,after=rowfields(g,i)))
 return out
AccDecoder.pre_predict=predict
orig_cov=trans.apply_cov_rotation
def cov(c,f):
 global cov64
 if cov64 is None:
  cov64=array(c).astype(np.float64);current['covariance_initial']=covstat(cov64[30:])
 out=orig_cov(c,f)
 fc=array(f).astype(np.float64);cov64=fc@(cov64@fc.transpose(0,2,1))
 current['covariance'].append(dict(application_index=len(current['covariance']),hierarchy_level=2-len(current['covariance']),F=fstat(f[30:]),actual_float32=covstat(array(out)[30:]),float64_recompute=covstat(cov64[30:]),row_F=fc[3678].tolist(),row_cov32=array(out[3678]).tolist(),row_cov64=cov64[3678].tolist()))
 current['row_stages'].append(dict(event='after_all_position_broadcasts',row=rowfields(graphs[0],3678)))
 return out
trans.apply_cov_rotation=cov
# Reuse the audited CPU projection formula definitions only, never execute
# the T26.2 ablation program. Check all Gaussians with positive native radius.
import ast
projection_source=repo/'tools/deform360_adapter/audit_renderer_degeneracy.py'
parsed=ast.parse(projection_source.read_text());defs=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in ['fullcov','projection']]
assert len(defs)==2
exec(compile(ast.Module(body=defs,type_ignores=[]),str(projection_source),'exec'),globals())
report['projection_formula_source_sha256']=sha(projection_source)
report['normalizer_source_sha256']=sha(repo/'mmgs/models/utils/normalization.py')
def nonfinite(t):
 t=t.detach();good=t[torch.isfinite(t)]
 return dict(shape=list(t.shape),dtype=str(t.dtype),finite=bool(torch.isfinite(t).all()),nan=int(torch.isnan(t).sum()),posinf=int(torch.isposinf(t).sum()),neginf=int(torch.isneginf(t).sum()),finite_absmax=float(good.abs().max()) if good.numel() else None)
orig_native=raster._C.rasterize_gaussians
view_ids={}
def nf(*a):
 out=orig_native(*a);rr=array(out[2]);ci=len(current.get('rasterizer',[]));view_ids[digest(a[8])]=ci
 rec=dict(camera_index=ci,positive_radius=int((rr>0).sum()),radius=dist(rr),positive_radius_dist=dist(rr[rr>0]),rendered_tile_instances=out[0],render=stats(out[1]))
 if args.mode=='d360':
  p,c,v,pr=map(array,[a[1],a[7],a[8],a[9]])
  invalid=[];dets=[];pixels=[]
  for row in np.where(rr>0)[0]:
   z=projection(p[row],c[row],v,pr,a[10],a[11],a[13],a[12],np.float32)
   dets.append(float(z['det']));pixels.append(z['pixel'])
   if not np.isfinite(z['det']) or z['det']<=0 or z['eigenvalues'][0]<=0 or not np.isfinite(z['conic']).all():invalid.append(int(row))
  pixels=np.array(pixels)
  rec['projected_covariance']=dict(checked_positive_radius_rows=len(dets),invalid_rows=invalid,regularized_det=dist(dets),interpretation='CPU fp32 recomputation of installed formula; not a native intermediate readback')
  rec['projected_centers']=dict(in_image=int(((pixels[:,0]>=0)&(pixels[:,0]<a[13])&(pixels[:,1]>=0)&(pixels[:,1]<a[12])).sum()),x=dist(pixels[:,0]),y=dist(pixels[:,1]))
 current.setdefault('rasterizer',[]).append(rec);return out
raster._C.rasterize_gaussians=nf
orig_native_backward=raster._C.rasterize_gaussians_backward
original_tensor_backward=torch.Tensor.backward
report['native_backward']=[]
def nb(*a):
 out=orig_native_backward(*a)
 report['native_backward'].append(dict(camera_index=view_ids[digest(a[9])],incoming_color=nonfinite(a[13]),incoming_depth=nonfinite(a[14]),returned={n:nonfinite(t) for n,t in zip(['means2D','colors','opacity','means3D','covariance','SH','scales','rotations'],out)}))
 return out
raster._C.rasterize_gaussians_backward=nb
# Global diagnostic tripwire; never executes autograd backward, in any mode.
def forbidden_backward(*a,**kw):report['stopped_at_backward_request']=True;raise Done()
torch.Tensor.backward=forbidden_backward
OptimizerHook.after_train_iter=lambda *a,**kw: (_ for _ in ()).throw(Done())
start=time.monotonic()
try:
 torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();report['gpu']=torch.cuda.get_device_name()
 if args.mode=='t22':
  path=repo/'tools/deform360_adapter/smoke_dynamics_batch.py';sys.argv=[str(path),'--workspace',str(root),'--report',str(args.out/'historical_harness_report.json')]
 else:
  config=repo/'configs/SoMA/deform360_v0_stage1.py' if args.mode=='d360' else root/'environment/soma-stage1-isotropic-20260923/cloth_lift_stage1_scale3.py'
  report['config_path']=str(config);report['config_hash']=sha(config)
  path=repo/'tools/train.py';sys.argv=[str(path),str(config),'--seed','5','--gpus','1','--launcher','none','--work_dir',str(args.out/'runner')]
 runpy.run_path(str(path),run_name='__main__')
except Done:report['status']='BACKWARD_FINITE_NO_STEP' if report['backward'] else 'OBSERVED_NO_BACKWARD'
except BaseException:report['status']='ERROR';report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
finally:
 report['seconds']=time.monotonic()-start;report['peak_allocated']=torch.cuda.max_memory_allocated()
 if model is not None:report['final_normalizers']=norms(model)
 (args.out/'report.json').write_text(json.dumps(sanitize(report),indent=2)+'\n');print('AUDIT_RESULT',args.mode,report['status'],flush=True)
if report['status']=='ERROR':sys.exit(2)
