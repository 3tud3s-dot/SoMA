#!/usr/bin/env python3
"""T26.2 diagnostic only: formal seed5 first-step renderer causal ablations.
No optimizer/clip; modifications are per-render temporary tensors. Source untouched.
Projected covariance recomputation is CPU NumPy float32/64, not a kernel replacement.
"""
import argparse, hashlib, json, math, os, runpy, sys, time, traceback
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--probe', choices=['original','exclude','mean','cov','both'], required=True)
ap.add_argument('--camera', type=int, choices=[0,1], default=0)
args=ap.parse_args()
assert os.environ.get('SLURM_JOB_ID')
assert not args.out.exists(); args.out.mkdir(parents=True)
root=Path('/data1/userdata/tcweng/projects/tcgs'); repo=root/'SoMA'
os.chdir(repo); sys.path[:0]=[str(repo),str(repo/'tools')]
import numpy as np
import torch
import diff_gaussian_rasterization as raster
from mmcv.runner import OptimizerHook
from mmgs.core.runner.epoch_runner import EpochRunner
from mmgs.models.heads.acc_decoder import AccDecoder
from mmgs.models.losses.l2_loss import L2Loss
ROW=3648
CAMS=['brics-odroid-023_cam0','brics-odroid-009_cam1']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(t):return t.detach().cpu().contiguous().numpy()
def digest(t):return hashlib.sha256(arr(t).tobytes()).hexdigest()
def stat(t):
 t=t.detach(); f=torch.isfinite(t); v=t[f].double()
 return dict(shape=list(t.shape),dtype=str(t.dtype),finite=bool(f.all()),nan=int(torch.isnan(t).sum()),posinf=int(torch.isposinf(t).sum()),neginf=int(torch.isneginf(t).sum()),finite_abs_max=float(v.abs().max()) if v.numel() else None)
def nested(x):
 if torch.is_tensor(x):return dict(hash=digest(x),shape=list(x.shape),dtype=str(x.dtype))
 if isinstance(x,np.ndarray):return dict(hash=hashlib.sha256(x.tobytes()).hexdigest(),shape=list(x.shape),dtype=str(x.dtype))
 if isinstance(x,dict):return {k:nested(v) for k,v in x.items() if k!='cam'}
 if isinstance(x,(tuple,list)):return [nested(v) for v in x]
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 return str(type(x))
def safe(x):
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,np.generic):return safe(x.item())
 if isinstance(x,float) and not math.isfinite(x):return str(x)
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [safe(v) for v in x]
 return x
def fullcov(v):
 c=np.empty((*v.shape[:-1],3,3),dtype=v.dtype)
 c[...,0,0]=v[...,0];c[...,0,1]=c[...,1,0]=v[...,1];c[...,0,2]=c[...,2,0]=v[...,2]
 c[...,1,1]=v[...,3];c[...,1,2]=c[...,2,1]=v[...,4];c[...,2,2]=v[...,5]
 return c
def distribution(v,value):
 v=np.asarray(v);f=v[np.isfinite(v)]
 return dict(count=len(v),finite_count=len(f),quantiles=np.quantile(f,[0,.01,.05,.5,.95,.99,1]) if len(f) else [],row_value=value,rank_ascending_1based=int(np.sum(f<value))+1,percentile_le=100*float(np.mean(f<=value)) if len(f) else None)
def geometry(p,c):
 m=fullcov(c.astype(np.float64));e=np.linalg.eigvalsh(m);d=np.linalg.det(m);condition=np.max(np.abs(e),axis=1)/np.min(np.abs(e),axis=1)
 return dict(row_mean=p[ROW],row_packed_cov=c[ROW],row_matrix=m[ROW],eigenvalues=e[ROW],determinant=d[ROW],trace=np.trace(m[ROW]),condition_2=condition[ROW],PSD=bool(e[ROW,0]>=0),PD=bool(e[ROW,0]>0),symmetry_exact=bool(np.array_equal(m[ROW],m[ROW].T)),all_PD_count=int(np.sum(e[:,0]>0)),population={
 'position_norm':distribution(np.linalg.norm(p.astype(np.float64),axis=1),np.linalg.norm(p[ROW].astype(np.float64))),
 'eigen_min':distribution(e[:,0],e[ROW,0]),'eigen_max':distribution(e[:,2],e[ROW,2]),'determinant':distribution(d,d[ROW]),'condition':distribution(condition,condition[ROW])})
def projection(p,c,view,proj,tanx,tany,width,height,dtype):
 """GLM W/J interpreted as columns; row-vector view matrix stored transposed.
 Matmul association follows transpose(T)*transpose(Vrk)*T, scalar rounding
 differs from nvcc/FMA. These are diagnostic recomputations, not native internals.
 """
 D=dtype;p=p.astype(D);c=c.astype(D);view=view.astype(D);proj=proj.astype(D)
 hom=np.r_[p,D(1)].astype(D);t=(hom@view)[:3];clip=hom@proj
 ndc=clip[:3]/(clip[3]+D(1e-7));pixel=((ndc[:2]+D(1))*np.array([width,height],dtype=D)-D(1))/D(2)
 unclamped=t.copy();tx=t[0]/t[2];ty=t[1]/t[2];limx=D(1.3)*D(tanx);limy=D(1.3)*D(tany)
 t[0]=np.clip(tx,-limx,limx)*t[2];t[1]=np.clip(ty,-limy,limy)*t[2]
 fx=D(width)/(D(2)*D(tanx));fy=D(height)/(D(2)*D(tany))
 J=np.array([[fx/t[2],0,-(fx*t[0])/(t[2]*t[2])],[0,fy/t[2],-(fy*t[1])/(t[2]*t[2])],[0,0,0]],dtype=D).T
 W=view[:3,:3];T=W@J;cov=T.T@fullcov(c).T@T
 a,b,cc=cov[0,0],cov[1,0],cov[1,1];raw=np.array([[a,b],[b,cc]],dtype=D)
 detraw=a*cc-b*b;a=a+D(.3);cc=cc+D(.3);reg=np.array([[a,b],[b,cc]],dtype=D);det=a*cc-b*b
 with np.errstate(all='ignore'):
  ratio=detraw/det;scale=np.sqrt(np.maximum(D(.000025),ratio));conic=np.array([cc,-b,a],dtype=D)/det
  mid=D(.5)*(a+cc);rad_eig=mid+np.sqrt(np.maximum(D(.1),mid*mid-det));radius=np.ceil(D(3)*np.sqrt(rad_eig))
  aa_term=D(.3)*D(.3)+D(.3)*(a+cc)+a*cc-b*b
  aa_denom_sq=aa_term*aa_term
  denom_zero_upstream=D(0)/aa_denom_sq
 e=np.linalg.eigvalsh(reg.astype(np.float64))
 return dict(dtype=str(np.dtype(D)),view_xyz=unclamped,x_over_z=tx,y_over_z=ty,clamped_view_xyz=t,clip=clip,pixel=pixel,near_threshold=.2,near_pass=bool(unclamped[2]>.2),J=J,T=T,raw_2D=raw,raw_det=detraw,raw_eigen=np.linalg.eigvalsh(raw.astype(np.float64)),regularized_2D=reg,det=det,eigenvalues=e,condition=float(np.max(abs(e))/np.min(abs(e))),conic=conic,radius_eigenvalue=rad_eig,radius=radius,AA_ratio=ratio,AA_scale=scale,AA_backward_denominator_base=aa_term,AA_backward_denominator_square=aa_denom_sq,AA_backward_zero_upstream_division=denom_zero_upstream)

def native_geom(buffer,n):
 """Read-only prefix layout from installed build rasterizer_impl.cu:155-165.
 Avoid uninitialized cov3D/rgb when precomputed inputs supplied.
 """
 b=arr(buffer).tobytes();offset=0;result={}
 for name,typ,k in [('depths','<f4',1),('clamped','?',3),('internal_radii','<i4',1),('means2D','<f4',2),('cov3D','<f4',6),('conic_opacity','<f4',4),('rgb','<f4',3),('tiles_touched','<u4',1)]:
  offset=(offset+127)&~127;size=np.dtype(typ).itemsize*n*k
  if name in ['depths','means2D','conic_opacity','tiles_touched']:result[name]=np.frombuffer(b,dtype=typ,count=n*k,offset=offset).reshape(n,k)
  offset+=size
 return result

report=dict(task='T26.2',probe=args.probe,backward_camera=CAMS[args.camera],job_id=os.environ['SLURM_JOB_ID'],seed=5,row=ROW,rollout=1,source=113,target=123,optimizer_step=False,clipping=False,config_sha256=sha(repo/'configs/SoMA/deform360_v0_stage1.py'),tool_sha256=sha(__file__),native_forward=[],native_backward=[])
model=None;initial_p=initial_c=None;active_camera=None;objective=None
start=time.monotonic()
class Done(BaseException):pass
original_run=EpochRunner.run_iter
def run_iter(self,data_batch,train_mode,**kw):
 global model,initial_p,initial_c
 assert self.epoch==0 and self.iter==0 and train_mode
 model=self.model.module;assert len(self.data_loader)==50
 report['initial_state']={k:nested(v) for k,v in model.state_dict().items()};report['input_hashes']=nested(data_batch)
 assert len(model.gs_scene_dict)==1;g=next(iter(model.gs_scene_dict.values()))
 initial_p=g.get_xyz.detach().clone();initial_c=g.get_covariance().detach().clone()
 report['initial_Gaussian']={k:nested(getattr(g,k)) for k in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']}
 report['initial_geometry']=geometry(arr(initial_p),arr(initial_c))
 model.train_cfg.step_initial=1
 return original_run(self,data_batch,train_mode,**kw)
EpochRunner.run_iter=run_iter
original_render=AccDecoder.pre_render
def render(self,cam,*a,**kw):
 global active_camera
 active_camera=int(Path(cam.img_path).parent.name)
 assert active_camera in [0,1]
 expected_views=['2f3e3e39fec96734fc4de0cc7a20d163623dc77e458df3741754efcce9600b6f','16215af054e053511b8c26b7d3a2b092e3c0230f82ae006da4c86a09b46bc202']
 assert digest(cam.world_view_transform)==expected_views[active_camera]
 report.setdefault('cameras',{})[CAMS[active_camera]]=dict(img_path=str(cam.img_path),znear=cam.znear,zfar=cam.zfar,view=arr(cam.world_view_transform),projection=arr(cam.full_proj_transform),matrix_hashes=[digest(cam.world_view_transform),digest(cam.full_proj_transform)])
 return original_render(self,cam,*a,**kw)
AccDecoder.pre_render=render
original_raster=raster.GaussianRasterizer.forward
def raster_forward(self,*a,**kw):
 assert not a;assert kw['means3D'].shape==(12861,3)
 p=kw['means3D'];c=kw['cov3D_precomp']
 report.setdefault('pre_intervention_hashes',{})[CAMS[active_camera]]={k:digest(v) for k,v in kw.items() if torch.is_tensor(v)}
 if active_camera==0:
  report['predicted_geometry']=geometry(arr(p),arr(c))
  report['row_initial_delta']=dict(position=arr(p[ROW]-initial_p[ROW]),displacement=float(torch.linalg.norm(p[ROW]-initial_p[ROW])),covariance=arr(c[ROW]-initial_c[ROW]))
 if active_camera==0 and args.probe!='original':
  if args.probe=='exclude':
   keep=torch.arange(12861,device=p.device)!=ROW
   kw={k:(v[keep] if torch.is_tensor(v) and v.ndim>0 and v.shape[0]==12861 else v) for k,v in kw.items()}
  else:
   if args.probe in ['mean','both']:
    changed=p.clone();changed[ROW]=initial_p[ROW];kw['means3D']=changed
   if args.probe in ['cov','both']:
    changed=c.clone();changed[ROW]=initial_c[ROW];kw['cov3D_precomp']=changed
 return original_raster(self,**kw)
raster.GaussianRasterizer.forward=raster_forward
original_native_f=raster._C.rasterize_gaussians
original_native_b=raster._C.rasterize_gaussians_backward
view_to_camera={}
def native_forward(*a):
 out=original_native_f(*a);view_to_camera[digest(a[8])]=active_camera
 n=len(a[1]);p=arr(a[1]);c=arr(a[7]);v=arr(a[8]);pr=arr(a[9]);r=arr(out[2]);g=native_geom(out[3],n)
 report['cameras'][CAMS[active_camera]]['native_projection_parameters']=dict(tanfovx=a[10],tanfovy=a[11],height=a[12],width=a[13],antialiasing=a[18])
 rec=dict(camera=CAMS[active_camera],count=n,num_rendered_tile_instances=out[0],positive_radii_count=int((r>0).sum()),image=stat(out[1]),input_hashes={k:digest(a[i]) for k,i in [('position',1),('colors',2),('opacity',3),('covariance',7)]})
 z=(np.c_[p.astype(np.float64),np.ones(n)]@v.astype(np.float64))[:,2]
 if n==12861:
  rec['row']=dict(opacity=arr(a[3][ROW]),RGB=arr(a[2][ROW]),SH_DC=arr(next(iter(model.gs_scene_dict.values()))._features_dc[ROW]),radius=int(r[ROW]),native_geometry={k:val[ROW] for k,val in g.items()} if r[ROW]>0 else 'culled; geometry entries not initialized',fp32=projection(p[ROW],c[ROW],v,pr,a[10],a[11],a[13],a[12],np.float32),fp64=projection(p[ROW],c[ROW],v,pr,a[10],a[11],a[13],a[12],np.float64),initial_fp64=projection(arr(initial_p[ROW]),arr(initial_c[ROW]),v,pr,a[10],a[11],a[13],a[12],np.float64),z_distribution_positive_radius=distribution(z[r>0],z[ROW]),z_distribution_near_pass=distribution(z[z>.2],z[ROW]))
  # Record zero-denominator diagnostic candidates among actual visible rows, not all near-pass rows.
  suspicious=[]
  for row in np.where(r>0)[0]:
   q=projection(p[row],c[row],v,pr,a[10],a[11],a[13],a[12],np.float32)
   if q['AA_backward_denominator_square']==0 or q['det']<=0 or q['eigenvalues'][0]<=0:
    suspicious.append(dict(row=int(row),det=q['det'],eigenvalues=q['eigenvalues'],AA_backward_denominator_base=q['AA_backward_denominator_base']))
  rec['fp32_projection_suspicious_visible_rows']=suspicious
 report['native_forward'].append(rec)
 return out
def native_backward(*a):
 out=original_native_b(*a)
 rec=dict(camera=CAMS[view_to_camera[digest(a[9])]],incoming_color=stat(a[13]),incoming_depth=stat(a[14]),returned={},bad_rows={})
 for k,v in zip(['means2D','colors','opacity','means3D','covariance','SH','scales','rotations'],out):
  rec['returned'][k]=stat(v)
  if v.numel() and not torch.isfinite(v).all():rec['bad_rows'][k]=torch.nonzero(~torch.isfinite(v).reshape(v.shape[0],-1).all(-1)).flatten().cpu().tolist()
 report['native_backward'].append(rec);return out
raster._C.rasterize_gaussians=native_forward;raster._C.rasterize_gaussians_backward=native_backward
original_l2=L2Loss.forward
def l2(self,cls_score,label,mask_weights=None,*a,**kw):
 global objective
 result=original_l2(self,cls_score,label,mask_weights,*a,**kw)
 if self.loss_name=='loss_l2_render':
  assert len(cls_score)==2 and objective is None
  idx=args.camera; mw=None if mask_weights is None else [mask_weights[idx]]
  objective=original_l2(self,[cls_score[idx]],[label[idx]],mw,*a,**kw)
  report['objective']=dict(name=self.loss_name,weight=self.loss_weight,reduction=self.reduction,camera=CAMS[idx],single_camera_not_camera_mean=True,value=float(objective.detach()),formal_two_camera_loss=float(result.detach()))
 return result
L2Loss.forward=l2

def backward(self,runner):
 assert objective is not None and torch.isfinite(objective)
 assert all(p.grad is None for p in model.parameters())
 report['formal_losses']=dict(runner.outputs['log_vars']);objective.backward()
 gs={n:stat(p.grad) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
 bad=[n for n,s in gs.items() if not s['finite']]
 report['gradients']=dict(finite_tensor_count=len(gs)-len(bad),nonfinite_tensor_count=len(bad),first_bad_parameter=bad[0] if bad else None,**{k:sum(s[k] for s in gs.values()) for k in ['nan','posinf','neginf']})
 if not bad:report['gradients']['global_norm']=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
 report['result']='NONFINITE' if bad else 'FINITE';raise Done()
OptimizerHook.after_train_iter=backward
try:
 torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();report['GPU']=torch.cuda.get_device_name()
 report['native_source']=dict(python_path=raster.__file__,python_hash=sha(raster.__file__),so_path=raster._C.__file__,so_hash=sha(raster._C.__file__))
 sys.argv=[str(repo/'tools/train.py'),'configs/SoMA/deform360_v0_stage1.py','--seed','5','--gpus','1','--launcher','none','--work_dir',str(args.out/'runner')]
 runpy.run_path(sys.argv[0],run_name='__main__')
except Done:pass
except BaseException:
 report['result']='EXCEPTION';report['traceback']=traceback.format_exc();print(report['traceback'])
finally:
 report['wall_seconds']=time.monotonic()-start;report['peak_allocated']=torch.cuda.max_memory_allocated();report['peak_reserved']=torch.cuda.max_memory_reserved()
 if model is not None:
  g=next(iter(model.gs_scene_dict.values()))
  report['Gaussian_unchanged']=report['initial_Gaussian']=={k:nested(getattr(g,k)) for k in report['initial_Gaussian']}
  report['model_finite']=all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
 (args.out/'report.json').write_text(json.dumps(safe(report),indent=2)+'\n')
 print('RESULT',args.probe,args.camera,report['result'],report.get('gradients'),flush=True)
if report['result']=='EXCEPTION':sys.exit(2)
