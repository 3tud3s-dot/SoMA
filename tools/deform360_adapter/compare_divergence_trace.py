#!/usr/bin/env python3
"""CPU comparison of chronological coarse/fine observation tensors, without changing tolerance."""
import argparse,json
from pathlib import Path
import torch

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--left',required=True);ap.add_argument('--right',required=True);args=ap.parse_args()
    root=args.directory;out=root/(args.left+'_vs_'+args.right+'.json');assert not out.exists()
    l=torch.load(root/(args.left+'_trace.pt'),map_location='cpu',weights_only=False)
    r=torch.load(root/(args.right+'_trace.pt'),map_location='cpu',weights_only=False)
    meta=json.loads((root/(args.left+'_trace.json')).read_text());assert l.keys()==r.keys()
    rows={};stages={s:{'bitwise_equal':True,'non_equal_tensors':[]} for s in meta['stages']}
    for k,a in l.items():
        b=r[k];assert a.shape==b.shape and a.dtype==b.dtype
        same=a.contiguous().numpy().tobytes()==b.contiguous().numpy().tobytes()
        delta=(a.double()-b.double()).abs()
        rows[k]={'bitwise_equal':same,'max_abs_diff':float(delta.max()) if delta.numel() else 0.,'mean_abs_diff':float(delta.mean()) if delta.numel() else 0.,
                 'shape':list(a.shape),'dtype':str(a.dtype),'original_tolerance_pass':torch.allclose(a,b,atol=1e-5,rtol=1e-5)}
        stage=k.split('/')[0]
        if not same:stages[stage]['bitwise_equal']=False;stages[stage]['non_equal_tensors'].append(k)
    first=next((s for s in meta['stages'] if not stages[s]['bitwise_equal']),None)
    previous=meta['stages'][meta['stages'].index(first)-1] if first and meta['stages'].index(first)>0 else None
    report={'left':args.left,'right':args.right,'first_non_equal_stage':first,'last_exact_stage_before_first':previous,
            'stages':stages,'tensors':rows,'non_equal_tensor_count':sum(not v['bitwise_equal'] for v in rows.values())}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['stages','tensors']}))
    if first:print(json.dumps({k:rows[k] for k in stages[first]['non_equal_tensors']},indent=2))

if __name__=='__main__':main()
