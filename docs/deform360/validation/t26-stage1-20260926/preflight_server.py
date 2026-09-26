"""CPU-only T26 preflight: verify frozen config and existing assets; generate nothing."""
import hashlib, json, pickle, subprocess, sys
from pathlib import Path
import numpy as np
from mmcv import Config
root=Path('/data1/userdata/tcweng/projects/tcgs'); repo=root/'SoMA'
cp=repo/'docs/deform360/contracts/008-pink-cloth/episode_0'
control=Path(sys.argv[1])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def doc(name):return json.loads((cp/(name+'.json')).read_text())
head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert head=='ff02eaf9fceaf6f9bf1f53ebd615eef545ddd922'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
protocol=doc('stage1_protocol_contract');config=repo/protocol['config']['path']
assert sha(config)==protocol['config']['sha256']
for name,h in protocol['provenance']['input_contract_hashes'].items():assert sha(cp/name)==h,name
for name,info in protocol['provenance']['source_files'].items():assert sha(repo/name)==info['sha256'],name
cfg=Config.fromfile(str(config))
assert cfg.seed==5 and cfg.runner.max_epochs==46 and cfg.data.train.times==50
assert cfg.model.train_cfg.step_initial==3 and cfg.model.train_cfg.max_rollout_step==1000
assert cfg.optimizer.lr==4e-4 and cfg.optimizer_config.grad_clip.max_norm==1
assert cfg.load_from is None and cfg.resume_from is None
assert not Path(cfg.work_dir).exists(), 'Existing output path: stop, no overwrite'
package=doc('scene_package_contract');cameras=protocol['canonical_setup']['camera_ids']
scene=root/package['configurations']['A']['scene_directory']
for phase in ['train','val','test']:
 env=cfg.data.train.dataset.env_cfg if phase=='train' else cfg.data[phase].env_cfg
 assert env.scene_list==['config_a_2cam'] and env.split_list=={'config_a_2cam':[[0,155]]}
 assert env.frame_gap==10 and env.resolution==[360,640]
 assert env.controller_cfg['config_a_2cam']==[{'num_cluster':10},{'downsample_rate':.5}]
 assert env.dt==1/30 and env.real_dt['config_a_2cam']==1/15
metadata=json.loads((scene/'metadata.json').read_text())
assert metadata['serial_numbers']==cameras and metadata['WH']==[640,360]
assert metadata['frame_num']==194
checked={}
def check(p,h=None):
 p=Path(p);actual=sha(p)
 if h is not None:assert actual==h,str(p)
 checked[str(p)]=actual
for rel,info in package['generated_files'].items():
 check(root/rel,info['sha256'])
for file in scene.rglob('*'):
 if file.is_file() and 'cluster_mask' not in file.parts and 'controller_mask' not in file.parts:
  check(file)
traj_doc=doc('controller_trajectory_contract');p=root/traj_doc['output']['path_relative_to_workspace']
check(p,traj_doc['output']['sha256']);traj=np.load(p,allow_pickle=False)
assert traj.shape==(194,30,3) and traj.dtype==np.float32 and np.isfinite(traj).all()
with (scene/'track_process_data.pkl').open('rb') as f:tracked=pickle.load(f)
assert np.array_equal(tracked['controller_points'],traj)
with (scene/'calibrate.pkl').open('rb') as f:cal=pickle.load(f)
for name in ['rgb_export_contract','mask_export_contract']:
 d=doc(name);assert d['configurations']['A']['camera_ids']==cameras
 for camera in cameras:
  m=d['cameras'][camera]
  for row in m['manifest']:
   p=Path(m['output_directory'])/(str(row['local_frame'])+'.png');check(p,row['sha256'])
for row in doc('gaussian_sequence_sh0_contract')['frames']:
 p=root/row['output_path'];assert p.is_file()
 # All sequence files exist; only canonical initial is an input to this run.
 if row['source_frame']==113:check(p,row['output_sha256'])
initial=Path(cfg.model.gs_scene[0].model_path);check(initial)
assert sha(initial)==doc('gaussian_sequence_sh0_contract')['frames'][0]['output_sha256']
expanded=control/'expanded_config.py';assert not expanded.exists();cfg.dump(str(expanded))
report={'status':'PASS','head':head,'config_sha256':sha(config),'expanded_config_sha256':sha(expanded),'camera_ids':cameras,'sample_local_frames':list(range(0,155,10)),'sample_source_frames':list(range(113,268,10)),'test_supervision':False,'controller_shape':list(traj.shape),'checked_asset_count':len(checked),'asset_hashes':checked,'new_output_directory':cfg.work_dir,'CUDA_initialized':False,'assets_regenerated':False}
(control/'resumed_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='asset_hashes'},indent=2))
