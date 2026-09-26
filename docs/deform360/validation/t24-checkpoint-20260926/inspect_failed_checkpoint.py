import torch,json,hashlib
from pathlib import Path
p=Path('/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t24-recovery-20260926/epoch_1.pth')
c=torch.load(p,map_location='cpu',weights_only=False)
r={'path':str(p),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'meta':c['meta'],'model_key_count':len(c['state_dict']),'optimizer_state_count':len(c['optimizer']['state']),'nonfinite_model':[],'nonfinite_optimizer':[]}
for k,v in c['state_dict'].items():
 if torch.is_tensor(v) and not torch.isfinite(v).all():r['nonfinite_model'].append({'key':k,'count':int((~torch.isfinite(v)).sum())})
for k,d in c['optimizer']['state'].items():
 for n,v in d.items():
  if torch.is_tensor(v) and not torch.isfinite(v).all():r['nonfinite_optimizer'].append({'id':k,'field':n,'count':int((~torch.isfinite(v)).sum())})
Path('/tmp/tcgs_t24_checkpoint_inspection.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if not k.startswith('nonfinite')}));print('nonfinite model',len(r['nonfinite_model']),'optimizer',len(r['nonfinite_optimizer']))
