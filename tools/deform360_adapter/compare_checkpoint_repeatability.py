#!/usr/bin/env python3
"""Read-only CPU comparison of T24 independent-reload outputs; no tolerance changes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--presave',type=Path,required=True)
    args=parser.parse_args();d=args.directory
    report_path=d/'comparison.json';assert not report_path.exists()
    reports={k:json.loads((d/f'{k}_report.json').read_text()) for k in ['A','B']}
    match_keys=['seed','tool_sha256','checkpoint_sha256','config_sha256','fixed_input_sha256','model_sha256','optimizer_sha256','initial_gaussian_sha256','backend','counters','runtime']
    for k in match_keys:assert reports['A'][k]==reports['B'][k],k
    assert reports['A']['pid']!=reports['B']['pid'] and reports['A']['job_id']!=reports['B']['job_id']
    for r in reports.values():
        assert r['status']=='PASS' and r['forward_count']==1 and r['optimizer_steps']==0
        for k in ['model_exact','optimizer_exact','normalizer_exact','counters_exact','all_output_finite','input_checkpoint_source_unchanged']:assert r[k],k
    outputs={k:torch.load(d/f'{k}_output.pt',map_location='cpu',weights_only=False) for k in ['A','B']}
    outputs['presave']=torch.load(args.presave,map_location='cpu',weights_only=False)
    result={'matched_provenance_keys':match_keys,'processes':{k:{'pid':r['pid'],'job_id':r['job_id']} for k,r in reports.items()},'comparisons':{},'original_tolerance':{'atol':1e-5,'rtol':1e-5}}
    def compare(a,b,path,rows):
        if torch.is_tensor(a):
            assert a.dtype==b.dtype and a.shape==b.shape,path
            assert torch.isfinite(a).all() and torch.isfinite(b).all(),path
            delta=(a.double()-b.double()).abs()
            rows[path]={'shape':list(a.shape),'dtype':str(a.dtype),'max_abs_diff':float(delta.max()) if delta.numel() else 0.,
                        'mean_abs_diff':float(delta.mean()) if delta.numel() else 0.,
                        'bitwise_equal':a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes(),
                        'original_tolerance_pass':torch.allclose(a,b,atol=1e-5,rtol=1e-5)}
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k],path+'/'+str(k),rows)
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),rows)
        else:assert a==b
    for left,right in [('A','B'),('presave','A'),('presave','B')]:
        rows={};compare(outputs[left],outputs[right],'output',rows);result['comparisons'][left+'_vs_'+right]=rows
    result['artifact_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [d/'A_output.pt',d/'B_output.pt',args.presave]}
    report_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['comparisons']['A_vs_B'],indent=2))

if __name__=='__main__':main()
