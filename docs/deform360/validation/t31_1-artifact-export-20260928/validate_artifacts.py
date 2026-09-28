"""CPU-only readback/hash validation of T31.1 artifacts; no metric recomputation."""
import csv,hashlib,json,pathlib,sys
import numpy as np
from PIL import Image
root=pathlib.Path(sys.argv[1]);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text());assert sha(root/'manifest.json')==(root/'manifest.sha256').read_text().split()[0]
assert m['sequence']==dict(start_source=113,end_source=306,num_steps=193,total_frames=194,initial_frame=0,predicted_local_frames=[1,193])
assert len(m['frames'])==194
for local,row in enumerate(m['frames']):
 s=row['state'];assert s['local_frame']==local and s['source_frame']==113+local
 p=root/s['path'];assert sha(p)==s['sha256']
 with np.load(p,allow_pickle=False) as d:
  assert set(d.files)=={'pred_pos','pred_cov'}
  for k,shape,h in [('pred_pos',(12861,3),s['pos_hash']),('pred_cov',(12861,6),s['cov_hash'])]:
   a=d[k];assert a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all()
   assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==h
 assert [c['camera_id'] for c in row['cameras']]==m['artifact']['camera_order']
 for c in row['cameras']:
  assert c['local_frame']==local and c['source_frame']==local+113
  p=root/c['render'];assert sha(p)==c['sha256']
  with Image.open(p) as im:
   im.load();a=np.asarray(im);assert im.mode=='RGB' and a.dtype==np.uint8 and a.shape==(360,640,3)
  assert sha(c['rgb_source'])==c['rgb_sha256'] and sha(c['mask_source'])==c['mask_sha256']
assert len(list((root/'renders').rglob('*.png')))==388
assert len(list((root/'states').glob('*.npz')))==194
for path,h in m['table_hashes'].items():assert sha(root/path)==h
pro=list(csv.DictReader((root/'provenance.csv').open()));metrics=list(csv.DictReader((root/'metrics.csv').open()));report=json.loads((root/'report.json').read_text())
assert len(pro)==194 and len(metrics)==193 and len(report['steps'])==193
for i,row in enumerate(pro):
 assert int(row['local_frame'])==i and int(row['source_frame'])==113+i
 assert row['current_state_hash']==m['frames'][i]['state']['pos_hash']
 if i:
  assert row['state_origin']=='model prediction'
  assert row['previous_state_hash']==m['frames'][i-1]['state']['pos_hash']
for r,s in zip(metrics,report['steps']):
 assert int(r['local_frame'])==s['step'] and int(r['source_frame'])==s['target_source']
 for k,v in s['metrics'].items():assert float(r[k])==float(v)
 for k,v in s['displacement'].items():assert float(r['displacement_'+k])==float(v)
 assert int(r['covariance_PD_count'])==s['PD_count']
assert not m['future_PLY_read'] and not m['future_cache_injection'] and m['registered_state_unchanged']
result=dict(task='T31.1',status='PASS',artifact_directory=str(root),checkpoint=m['checkpoint'],job_id=m['job_id'],config=m['config'],sequence=m['sequence'],counts=m['artifact'],manifest_sha256=sha(root/'manifest.json'),report_sha256=sha(root/'report.json'),all_artifact_hashes_verified=True,all_PNG_decoded=True,all_states_finite_float32=True,metrics_exactly_transcribed=True,position_chain_exact=True,future_PLY_read=False,future_cache_injection=False,known_previous_state_reassignments=len(report['online_updates']),limitations=m['limitations'],total_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
(root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
