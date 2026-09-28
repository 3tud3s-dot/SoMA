import json,hashlib,pathlib,torch,subprocess
from mmcv import Config
root=pathlib.Path('/data1/userdata/tcweng/projects/tcgs');repo=root/'SoMA';control=root/'outputs/deform360/t30-resume24h-control-20260928'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
cfg=Config.fromfile(str(repo/'configs/SoMA/deform360_v0_stage2.py'))
assert subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()==''
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert head=='9cd6fe0706f1b1ad4f4f1b0430bbaafb5dccd16e'
assert sha(repo/'configs/SoMA/deform360_v0_stage2.py')=='c4042fa409bdb4bc2746629a390f9763aa7f277a8ec54d2e43e314b30566e16a'
prior=json.loads((root/'outputs/deform360/t30-entry-fixed-control-20260927/preflight.json').read_text())
assets=prior['asset_hashes'];assert all(sha(p)==h for p,h in assets.items())
p=root/'outputs/deform360/stage2/config_a_seed5/epoch_22.pth';h=sha(p)
assert h=='31223b8ca972db83a78412040ddee9dcf7e5836f63fb11cae54b5f91aec9380c'
ck=torch.load(p,map_location='cpu',weights_only=False)
def finite(x):
 if torch.is_tensor(x):return bool(torch.isfinite(x).all())
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
 return True
assert finite(ck['state_dict']) and finite(ck['optimizer'])
assert ck['meta']['epoch']==22 and ck['meta']['iter']==13200
st=ck['optimizer']['state'];assert len(st)==275 and all(float(v['step'])==13200 for v in st.values())
norm={k:v for k,v in ck['state_dict'].items() if 'normalizer.' in k};assert len(norm)==12 and all(v.dtype==torch.float64 for v in norm.values())
assert cfg.runner.max_epochs==61 and cfg.seed==5 and cfg.resume_from is None
assert not cfg.model.flag_update_gaussian and not cfg.model.flag_update_gaussian_train
assert cfg.model.test_rollout_mode=='segmented'
assert not torch.cuda.is_initialized()
r={'status':'PASS','head':head,'config_sha256':sha(repo/'configs/SoMA/deform360_v0_stage2.py'),'asset_hashes':assets,'checkpoint':{'path':str(p),'bytes':p.stat().st_size,'sha256':h,'epoch':22,'iter':13200},'model_finite':True,'optimizer_finite':True,'Adam_states':275,'Adam_steps':[13200],'normalizer_fp64_count':12,'cuda_initialized':False,'allocation_hours':24,'target_epochs':61,'target_steps':36600,'first_resumed_human_epoch':23,'first_resumed_human_iteration':13201,'expected_first_LR':4e-4*(.5**(23//2)+.01),'expected_requested_rollout':69,'effective_rollout':[12,9]}
(control/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='asset_hashes'}))
