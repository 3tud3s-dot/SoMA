#!/usr/bin/env python3
"""Read-only comparison of old R, post-save S, fresh A/B and targeted runtime snapshots."""
import argparse, hashlib, json
from pathlib import Path
import torch

def compare(a,b,path='',rows=None):
    if rows is None:rows={}
    if torch.is_tensor(a):
        assert a.shape==b.shape and a.dtype==b.dtype
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        d=(a.double()-b.double()).abs()
        rows[path]={'max_abs_diff':float(d.max()) if d.numel() else 0,'mean_abs_diff':float(d.mean()) if d.numel() else 0,
                    'bitwise_equal':a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes(),
                    'within_original_tolerance':torch.allclose(a,b,atol=1e-5,rtol=1e-5)}
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k],path+'/'+str(k),rows)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),rows)
    else:assert a==b
    return rows

def diff_json(a,b,path='',rows=None):
    if rows is None:rows=[]
    if path.startswith('/rng/'):
        if a!=b:rows.append({'path':path,'equal':False,'detail':'RNG state differs'})
    elif isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:rows.append({'path':path+'/'+k,'missing':True})
            else:diff_json(a[k],b[k],path+'/'+k,rows)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):rows.append({'path':path,'length_mismatch':True})
        else:
            for i,(x,y) in enumerate(zip(a,b)):diff_json(x,y,path+'/'+str(i),rows)
    elif a!=b:rows.append({'path':path,'left':a,'right':b})
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--old-directory',type=Path,required=True);a=ap.parse_args()
    p=a.directory;old=a.old_directory;outfile=p/'comparison.json';assert not outfile.exists()
    o={k:torch.load(p/f'{k}_output.pt',map_location='cpu',weights_only=False) for k in ['S','A','B']}
    o['R']=torch.load(old/'reference_output.pt',map_location='cpu',weights_only=False)
    out={'comparisons':{x+'_vs_'+y:compare(o[x],o[y]) for x,y in [('R','S'),('S','A'),('S','B'),('A','B')]},'state_differences':{},'tolerance':{'atol':1e-5,'rtol':1e-5}}
    states={k:json.loads((p/f'{k}_before_forward_state.json').read_text()) for k in ['S','A','B']}
    for x,y in [('S','A'),('S','B'),('A','B')]:out['state_differences'][x+'_vs_'+y]=diff_json(states[x],states[y])
    out['during_forward_state_differences']={k:diff_json(states[k],json.loads((p/f'{k}_after_forward_state.json').read_text())) for k in ['S','A','B']}
    oldref=torch.load(old/'reference_state.pt',map_location='cpu',weights_only=False);newref=torch.load(p/'reference_state.pt',map_location='cpu',weights_only=False)
    out['old_R_vs_new_S_checkpoint_model']=compare(oldref['model'],newref['model'])
    out['old_R_vs_new_S_checkpoint_model_all_exact']=all(v['bitwise_equal'] for v in out['old_R_vs_new_S_checkpoint_model'].values())
    out['hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [old/'reference_output.pt',old/'epoch_1.pth',p/'epoch_1.pth']}
    outfile.write_text(json.dumps(out,indent=2)+'\n')
    for key,rows in out['comparisons'].items():print(key,json.dumps(rows))
    print('runtime',json.dumps(out['state_differences']));print('R/S same checkpoint model',out['old_R_vs_new_S_checkpoint_model_all_exact'])

if __name__=='__main__':main()
