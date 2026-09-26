import json,hashlib,pathlib,torch,subprocess
from mmcv import Config
root=pathlib.Path('/data1/userdata/tcweng/projects/tcgs');repo=root/'SoMA';control=root/'outputs/deform360/t30-stage2-control-20260927'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
cfg=Config.fromfile(str(repo/'configs/SoMA/deform360_v0_stage2.py'))
assert not pathlib.Path(cfg.work_dir).exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='1d54006ed2a7a8d3f1c247232b484186f9a16363'
contract=json.loads((repo/'docs/deform360/contracts/008-pink-cloth/episode_0/stage2_protocol_contract.json').read_text())
assert sha(repo/'configs/SoMA/deform360_v0_stage2.py')==contract['config_sha256']
cache=json.loads((repo/'docs/deform360/contracts/008-pink-cloth/episode_0/stage1_cache_contract.json').read_text())['canonical']
assets={}
for row in cache['frames']:
 assert sha(row['path'])==row['sha256'];assets[row['path']]=row['sha256']
 d=torch.load(row['path'],map_location='cpu',weights_only=False)
 assert d['pred_pos'].shape==(12861,3) and d['pred_cov'].shape==(12861,6)
 assert all(torch.isfinite(v).all() for v in d.values())
assert sha(cfg.load_from)==contract['initialization']['sha256'];assets[cfg.load_from]=sha(cfg.load_from)
ck=torch.load(cfg.load_from,map_location='cpu',weights_only=False)
norm={n:v for n,v in ck['state_dict'].items() if 'normalizer.' in n}
assert len(norm)==12 and all(v.dtype==torch.float64 and torch.isfinite(v).all() for v in norm.values())
prior=json.loads((repo/'docs/deform360/validation/t26-stage1-20260926/resumed_preflight.json').read_text())
for p,h in prior['asset_hashes'].items():
 assert sha(p)==h;assets[p]=h
assert cfg.resume_from is None and cfg.runner.max_epochs==61 and cfg.seed==5
assert not cfg.model.flag_update_gaussian and not cfg.model.flag_update_gaussian_train
assert cfg.model.test_rollout_mode=='segmented'
assert not torch.cuda.is_initialized()
r=dict(status='PASS',head='1d54006ed2a7a8d3f1c247232b484186f9a16363',config_sha256=contract['config_sha256'],asset_hashes=assets,cache_files=16,normalizer_fp64_count=12,epochs=61,total_steps=36600,allocation_hours=12,automatic_resume_after_12h=False,cuda_initialized=False)
(control/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='asset_hashes'}))
