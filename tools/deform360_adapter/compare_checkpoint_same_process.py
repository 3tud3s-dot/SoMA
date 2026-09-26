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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args()
    p=a.directory;outfile=p/'comparison.json';assert not outfile.exists()
    labels=['S0','S0_repeat','S1','S1_repeat','A']
    o={k:torch.load(p/f'{k}_output.pt',map_location='cpu',weights_only=False) for k in labels}
    pairs=[('S0','S0_repeat'),('S1','S1_repeat'),('S0','S1'),('S1','A'),('S0','A')]
    out={'comparisons':{x+'_vs_'+y:compare(o[x],o[y]) for x,y in pairs},'state_differences':{},'tolerance':{'atol':1e-5,'rtol':1e-5}}
    states={k:json.loads((p/f'{k}_before_forward_state.json').read_text()) for k in labels}
    for x,y in pairs:out['state_differences'][x+'_vs_'+y]=diff_json(states[x],states[y])
    out['during_forward_state_differences']={k:diff_json(states[k],json.loads((p/f'{k}_after_forward_state.json').read_text())) for k in labels}
    required=['forward_arguments','module_modes','parameters','registered_buffers','scene_dictionaries','render_pipeline','input','backend']
    out['required_state_groups_equal']={f'{x}_vs_{y}':{k:states[x][k]==states[y][k] for k in required} for x,y in pairs}
    assert all(all(v.values()) for v in out['required_state_groups_equal'].values()),'Required setup differs'
    summaries={k:{'all_bitwise_equal':all(v['bitwise_equal'] for v in rows.values()),'all_original_tolerance_pass':all(v['within_original_tolerance'] for v in rows.values())} for k,rows in out['comparisons'].items()}
    out['summaries']=summaries
    b=lambda k:summaries[k]['all_bitwise_equal']
    if b('S1_vs_A') and not b('S0_vs_S1'):case=1
    elif b('S0_vs_S1') and not b('S1_vs_A'):case=2
    elif not b('S0_vs_S1') and not b('S1_vs_A'):case=3
    elif b('S0_vs_S1') and b('S1_vs_A'):case=4
    else:raise AssertionError('Unclassified')
    out['case']=case;out['classification_basis']='all outputs bitwise equality; unchanged atol/rtol gates also reported'
    outfile.write_text(json.dumps(out,indent=2)+'\n')
    print('CASE',case,'summary',json.dumps(summaries))
    for k,rows in out['comparisons'].items():print(k,json.dumps(rows))

if __name__=='__main__':main()
