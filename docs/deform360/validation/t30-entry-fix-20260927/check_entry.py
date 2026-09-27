"""CPU execution of actual train.py functions to a mocked model-build boundary.
No production edits, environment probing, CUDA seeding, logging writes or model construction.
"""
import ast,argparse,copy,json,os,sys,time,types,hashlib,pathlib
import torch
from mmcv import Config,DictAction
repo=pathlib.Path('/data1/userdata/tcweng/projects/tcgs/SoMA')
p=repo/'configs/SoMA/deform360_v0_stage2.py'
c=Config.fromfile(str(p));before=c.to_dict()
old=Config.fromstring(__import__('subprocess').check_output(['git','show','HEAD:configs/SoMA/deform360_v0_stage2.py'],cwd=repo,text=True).replace("'../_base_/","'"+str(repo/'configs/_base_')+'/'),'.py').to_dict()
assert {k:v for k,v in before.items() if k!='max_seq'}==old and c.max_seq==1
source=(repo/'tools/train.py').read_text();tree=ast.parse(source)
fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['parse_args','main']]
reads=sorted({n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='cfg' and isinstance(n.ctx,ast.Load)})
class Boundary(Exception):pass
captured={}
def stop(model_cfg):
 captured.update(sys._getframe(1).f_locals['cfg'].to_dict());raise Boundary()
ns=dict(argparse=argparse,copy=copy,os=os,osp=os.path,sys=sys,time=time,Config=Config,DictAction=DictAction,torch=torch,mmcv=types.SimpleNamespace(mkdir_or_exist=lambda p:None),get_root_logger=lambda **kw:types.SimpleNamespace(info=lambda *a:None),collect_env=lambda:{'preflight':'CPU mocked environment probe'},set_random_seed=lambda *a,**kw:None,build_simulator=stop)
exec(compile(ast.Module(body=fns,type_ignores=[]),str(repo/'tools/train.py'),'exec'),ns)
sys.argv=['tools/train.py',str(p),'--seed','5','--gpus','1','--launcher','none','--work_dir',c.work_dir]
try:ns['main']()
except Boundary:pass
else:raise AssertionError('Did not reach model boundary')
assert captured['data']==before['data'] and captured['seed']==5 and list(captured['gpu_ids'])==[0]
for k,t in {'max_seq':int,'seed':int,'work_dir':str,'load_from':str,'workflow':list,'model':dict,'data':dict,'optimizer':dict,'optimizer_config':dict,'lr_config':dict,'runner':dict,'checkpoint_config':dict,'evaluation':dict,'log_config':dict,'log_level':str,'dist_params':dict}.items():assert isinstance(captured[k],t),(k,type(captured[k]))
assert captured['resume_from'] is None
assert c.runner.max_epochs==61 and c.data.train.times==40 and c.data.samples_per_gpu==1
assert c.model.train_cfg.step_initial==3 and c.model.train_cfg.step_increase_interval==1 and c.lr_config.step_start==-1
assert not c.model.flag_update_gaussian and not c.model.flag_update_gaussian_train
windows=c.data.train.dataset.env_cfg.split_list.config_a_2cam
assert windows==[[i,min(i+13,150)] for i in range(0,150,10)]
assert c.model.frame_gap==10 and c.data.train.dataset.env_cfg.frame_gap==1
assert len(windows)*40*61==36600
ref=json.loads((repo/'docs/deform360/validation/t30_1-protocol-freeze-20260927/static_validation.json').read_text())
for row in ref['schedule']:
 k=row['relative_epoch'];assert min(3*k+3,1000)==row['requested_rollout'];assert .0004*(.5**((k+1)//2)+.01)==row['lr']
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),only_config_change='top-level max_seq=1',direct_cfg_reads=reads,created_fields={'gpu_ids':'set by actual CLI branch [0]','seed':'CLI --seed5'},boundary='actual main reaches build_simulator; injected sentinel stops before model/CUDA',mocked_side_effects=['mkdir','logger','collect_env','set_random_seed'],full_main_functions_from_actual_AST=True,CUDA_initialized=False,windows=15,steps_per_epoch=600,total_steps=36600,all61schedule_rows_unchanged=True,first_rollout=3,first_lr=.000404),indent=2))
