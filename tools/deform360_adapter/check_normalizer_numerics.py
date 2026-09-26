#!/usr/bin/env python3
"""T26.4 CPU-only numerical and dtype-migration tests; outputs server-only state.
Run save and reload as independent processes. No model training or CUDA calls.
"""
import argparse, hashlib, importlib.util, json, os
from pathlib import Path
import torch
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--reload',action='store_true');args=ap.parse_args()
root=Path('/data1/userdata/tcweng/projects/tcgs');repo=root/'SoMA'
p=repo/'mmgs/models/utils/normalization.py';spec=importlib.util.spec_from_file_location('normalizer_only',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);Normalizer=mod.Normalizer
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'pid':os.getpid(),'patch_sha256':sha(p),'CUDA':False}
if args.reload:
 payload=torch.load(args.out/'new_statistics.pth',map_location='cpu',weights_only=False)
 matches=[]
 for name,s in payload.items():
  n=Normalizer(s['_acc_sum'].shape[-1]);r=n.load_state_dict(s,strict=True)
  assert not r.missing_keys and not r.unexpected_keys
  assert all(v.dtype==torch.float64 and torch.equal(v,s[k]) for k,v in n.state_dict().items())
  matches.append(name)
 report.update(status='PASS',new_process_exact=True,normalizers=matches,state_sha256=sha(args.out/'new_statistics.pth'))
 (args.out/'reload.json').write_text(json.dumps(report,indent=2)+'\n');print(report);raise SystemExit
assert not args.out.exists();args.out.mkdir(parents=True)
evidence=repo/'docs/deform360/validation/t26_3-scale-audit-20260926/d360-fine-v2.json'
x=torch.tensor(json.loads(evidence.read_text())['normalizers'][0]['variance_audit']['small_coarse_input'],dtype=torch.float32)
report['input_evidence_sha256']=sha(evidence)
new={};cases=[]
for name,data in [('real_9x3',x),('synthetic',torch.tensor([[-39.2],[-39.2074]],dtype=torch.float32)),('constant',torch.full((8,3),-39.2,dtype=torch.float32))]:
 n=Normalizer(data.shape[-1]);n.train();y=n(data)
 xd=data.double();mean=xd.mean(0,keepdim=True);var=((xd-mean)**2).mean(0,keepdim=True);std=torch.maximum(var.sqrt(),n._std_epsilon.double())
 actual_var=n._acc_sum_squared/n._acc_count-n._mean().square()
 torch.testing.assert_close(n._mean(),mean,atol=1e-12,rtol=1e-12)
 torch.testing.assert_close(actual_var,var,atol=2e-12,rtol=1e-7)
 torch.testing.assert_close(n._std_with_epsilon(),std,atol=1e-9,rtol=1e-6)
 assert torch.isfinite(y).all() and y.dtype==data.dtype and y.abs().max()<5
 if name=='real_9x3':assert actual_var[0,2]>0 and .0030<float(n._std_with_epsilon()[0,2])<.0032
 if name=='constant':assert torch.equal(actual_var,torch.zeros_like(actual_var)) and torch.equal(n._std_with_epsilon(),torch.full_like(std,float(n._std_epsilon)))
 split=Normalizer(data.shape[-1]);split.train()
 for chunk in data.tensor_split(3):split(chunk)
 for key in ['_acc_count','_acc_sum','_acc_sum_squared']:
  torch.testing.assert_close(getattr(split,key),getattr(n,key),atol=2e-12,rtol=1e-12)
 torch.testing.assert_close(split._mean(),n._mean(),atol=1e-12,rtol=1e-12)
 torch.testing.assert_close(split._std_with_epsilon(),n._std_with_epsilon(),atol=1e-9,rtol=1e-6)
 before={k:v.clone() for k,v in n.state_dict().items()};n.eval();n(data)
 assert all(torch.equal(v,before[k]) for k,v in n.state_dict().items())
 torch.testing.assert_close(n.inverse(y),data,atol=1e-6,rtol=1e-7)
 assert n.inverse(y).dtype==data.dtype
 n.train()
 with torch.no_grad():n(data)
 assert n._num_accumulations.item()==2
 new[name]={k:v.clone() for k,v in n.state_dict().items()}
 cases.append(dict(name=name,mean=mean.tolist(),variance=actual_var.tolist(),reference_variance=var.tolist(),std=std.tolist(),output_absmax=float(y.abs().max()),output_dtype=str(y.dtype),split_count_equal=True,split_update_count=[1,3],eval_frozen=True,no_grad_still_updates=True))
oldpath=root/'outputs/deform360/t24-finite-recovery-v3-20260926/epoch_1.pth'
old=torch.load(oldpath,map_location='cpu',weights_only=False);matches=[]
for prefix in ['backbone.anchor_normalizer.','backbone.node_normalizer.','backbone.edge_normalizer.']:
 s={k[len(prefix):]:v for k,v in old['state_dict'].items() if k.startswith(prefix)}
 assert len(s)==4 and all(v.dtype==torch.float32 for v in s.values())
 n=Normalizer(s['_acc_sum'].shape[-1]);r=n.load_state_dict(s,strict=True)
 assert not r.missing_keys and not r.unexpected_keys
 assert all(v.shape==s[k].shape and v.dtype==torch.float64 and torch.equal(v,s[k].double()) for k,v in n.state_dict().items())
 new[prefix]={k:v.clone() for k,v in n.state_dict().items()};matches.append(prefix)
torch.save(new,args.out/'new_statistics.pth')
report.update(status='PASS',cases=cases,old_checkpoint=str(oldpath),old_checkpoint_sha256=sha(oldpath),old_to_new={'keys':12,'exact_upcast':True,'prefixes':matches,'lost_precision_recovered':False},new_state_sha256=sha(args.out/'new_statistics.pth'))
(args.out/'save.json').write_text(json.dumps(report,indent=2)+'\n');print('NORMALIZER_UNIT PASS')
