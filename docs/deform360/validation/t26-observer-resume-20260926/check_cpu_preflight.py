"""CPU-only: real observer JSON paths, checkpoint and source-derived schedule.
Extract pure methods using AST; do not import mmgs/mmcv or initialize CUDA.
"""
import ast,json,math,time,hashlib,tempfile,types,subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
root=Path('/data1/userdata/tcweng/projects/tcgs');repo=root/'SoMA';out=root/'outputs/deform360/t26-resume-control-20260926'
def extract(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
ns={'np':np,'math':math,'json':json,'time':time};extract(out/'run_stage1_observed.py',['clean','save','event'],ns)
old={**ns};extract(repo/'docs/deform360/validation/t26-normalizer-fix-20260926/run_stage1_observed.py',['clean'],old)
ens={'np':np,'defaultdict':defaultdict};extract(repo/'mmgs/datasets/embodied_dataset.py',['evaluate_frame'],ens)
# Same shape/type evidence through the ACTUAL evaluate_frame implementation.
# The failed process's exact metrics were not serialized; do not claim recovery.
metric_input=[{'acc':{'decode.loss_mse_momentum_step15':np.float32(.5335409641265869),'acc_l2_render.acc_step15':np.float32(.125)}}]
metrics={'per_frame':ens['evaluate_frame'](None,metric_input)}
try:json.dumps(old['clean'](metrics),allow_nan=False)
except TypeError: old_reproduced=True
else:raise AssertionError('Old sanitizer did not reproduce failure')
payload={'python':[1.5,2,True,None,'text'],'numpy':[np.float16(1.5),np.float32(.25),np.float64(.75),np.int32(4),np.int64(5),np.uint64(6),np.bool_(True)],'array':np.arange(6,dtype=np.float32).reshape(2,3),'nested':({'metrics':metrics},[np.array([np.nan,np.inf,-np.inf],dtype=np.float32)]),'nonfinite':[float('nan'),float('inf'),float('-inf')]}
clean=ns['clean'];c=clean(payload)
assert c['array']==[[0.,1.,2.],[3.,4.,5.]] and c['numpy']==[1.5,.25,.75,4,5,6,True]
assert c['nonfinite']==['nan','inf','-inf'] and c['nested'][1][0]==['nan','inf','-inf']
with tempfile.TemporaryDirectory() as td:
 path=Path(td);ns.update(CONTROL=path,report=payload,ctx={'phase':'test'},started=time.monotonic(),events=(path/'events.jsonl').open('w'))
 ns['event']('evaluation_complete',metrics=metrics)
 propagated=False
 try:
  try:raise RuntimeError('original training failure sentinel')
  finally:
   ns['report']={'status':'FAIL','original_exception':'original training failure sentinel','metrics':metrics,'context':payload}
   ns['save']()
 except RuntimeError as e:propagated=str(e)=='original training failure sentinel'
 ns['events'].close()
 loaded=json.loads((path/'training_report.json').read_text());event=json.loads((path/'events.jsonl').read_text())
 assert loaded['context']==c and event['metrics']==clean(metrics) and propagated
try:json.dumps(clean({'unsupported':object()}),allow_nan=False)
except TypeError:unsupported_explicit=True
else:raise AssertionError('Unknown object silently swallowed')
assert not torch.cuda.is_initialized()
checkpoint=root/'outputs/deform360/stage1/config_a_seed5_normalizer_fix/epoch_1.pth'
sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert sha=='ac600808287c27277edf39dba59d0bf21a7d90204345cf479a06c8d00bf173ae'
ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
def finite(x):
 if torch.is_tensor(x):return bool(torch.isfinite(x).all())
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,(tuple,list)):return all(finite(v) for v in x)
 return True
assert ck['meta']['epoch']==1 and ck['meta']['iter']==50 and finite(ck['state_dict']) and finite(ck['optimizer'])
stats={k:str(v.dtype) for k,v in ck['state_dict'].items() if 'normalizer' in k};assert len(stats)==12 and set(stats.values())=={'torch.float64'}
assert len(ck['optimizer']['state'])==275 and all(float(v['step'])==50 for v in ck['optimizer']['state'].values())
assert all(g['initial_lr']==.0004 for g in ck['optimizer']['param_groups'])
config=repo/'configs/SoMA/deform360_v0_stage1.py';cfg={}
for n in ast.parse(config.read_text()).body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):cfg[n.targets[0].id]=ast.literal_eval(n.value)
fns={};extract(repo/'mmgs/models/simulators/gs_simulator_embodied.py',['_rollout_steps'],fns)
assert fns['_rollout_steps'](None,1,50,cfg['model']['train_cfg'])[0]==6
lr_path=repo/'mmgs/core/lr_updater/hooks.py'
lr_class=next(n for n in ast.parse(lr_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HoodLrUpdaterHook')
lr_methods=[n for n in lr_class.body if isinstance(n,ast.FunctionDef) and n.name in ['_sched_fun','get_lr']]
assert len(lr_methods)==2
exec(compile(ast.Module(body=lr_methods,type_ignores=[]),str(lr_path),'exec'),fns)
obj=types.SimpleNamespace(**cfg['lr_config'],decay_min=0.0);obj._sched_fun=types.MethodType(fns['_sched_fun'],obj)
LR=fns['get_lr'](obj,types.SimpleNamespace(epoch=1,iter=50),ck['optimizer']['param_groups'][0]['initial_lr']);assert LR==.000404
assert cfg['data']['train']['times']==50 and cfg['runner']['max_epochs']==46
assert not torch.cuda.is_initialized()
r={'status':'PASS','test_harness_note':'Initial AST extraction matched get_lr in multiple classes; narrowed to actual HoodLrUpdaterHook before successful CPU rerun. No CUDA/training was launched.','CPU_only':True,'CUDA_initialized':False,'serialization':{'old_failure_reproduced':old_reproduced,'real_evaluate_frame_used':True,'fixture_provenance':'same-structure/type fixture; exact interrupted metric payload unavailable','nested_types_roundtrip':True,'event_and_finally_save':True,'original_exception_propagated':True,'nonfinite_policy':'Explicit strings nan/inf/-inf, existing semantics; not replaced by finite numbers. Training finite guards unchanged.','unsupported_object_raises':unsupported_explicit,'torch_dump_support_added':False,'torch_reason':'Actual report paths convert tensors explicitly; evaluation path returns NumPy. No arbitrary tensor JSON conversion.'},'checkpoint':{'path':str(checkpoint),'sha256':sha,'bytes':checkpoint.stat().st_size,'epoch':1,'iter':50,'all_finite':True,'normalizer_dtypes':stats,'Adam_steps':[50],'Adam_states':275,'initial_lr':.0004},'resume_semantics':{'restored_epoch':1,'restored_iter':50,'next_human_epoch':2,'next_optimizer_iteration':51,'next_rollout':6,'next_LR':LR,'dataset_repeat':50,'remaining_steps':2250,'target_total_steps':2300,'source_refs':['tools/train.py:94-95','mmgs/apis/train.py:141-143','MMCV base_runner.py:359-407','mmgs/core/runner/epoch_runner.py:19','mmgs/models/simulators/gs_simulator_embodied.py:584-609,638-640','mmgs/core/lr_updater/hooks.py:164-177','MMCV hooks/lr_updater.py:111-139','MMCV epoch_based_runner.py:44-60'],'actual_GPU_resume_exact_state_and_first_batch_gate':'required in submitted observer before continuation'},'observer_sha256':hashlib.sha256((out/'run_stage1_observed.py').read_bytes()).hexdigest()}
(out/'cpu_preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
