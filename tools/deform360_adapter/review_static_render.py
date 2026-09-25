#!/usr/bin/env python3
"""T18: initial-frame render review only. Requires Slurm GPU allocation."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm GPU allocation'
    assert not a.output.exists(), 'Refuse overwrite'
    root = a.workspace.resolve()
    repo = root/'SoMA'
    cp = repo/'docs/deform360/contracts/008-pink-cloth/episode_0'
    package = json.loads((cp/'scene_package_contract.json').read_text())
    intr = json.loads((cp/'camera_intrinsics_contract.json').read_text())
    ext = json.loads((cp/'camera_extrinsics_contract.json').read_text())
    for path, info in package['generated_files'].items():
        assert sha(root/path) == info['sha256'], path
    for path, target in package['symlink_mapping'].items():
        assert (root/path).resolve() == (root/target).resolve(), path
    smi = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True)
    import torch
    assert torch.cuda.is_available()
    sys.path.insert(0,str(repo/'gaussian-splatting'))
    from scene.gaussian_model import GaussianModel
    cams = module('t18_cameras',repo/'mmgs/datasets/utils/cameras.py')
    renderer = module('t18_renderer',repo/'mmgs/models/utils/render.py')
    code = ['gaussian-splatting/scene/gaussian_model.py','mmgs/datasets/utils/cameras.py','mmgs/models/utils/render.py']
    hashes = {f:sha(repo/f) for f in code}
    report = {'task':'T18','source_frame':113,'local_frame':0,
              'slurm_job_id':os.environ['SLURM_JOB_ID'],'gpu':smi.strip(),'torch':torch.__version__,
              'scope':'Static initial render; no dynamics model, prediction, backward or training.',
              'source_sha256':hashes,'package_contract_sha256':sha(cp/'scene_package_contract.json'),
              'tool_sha256':sha(Path(__file__)),'configurations':{},'cameras':{},
              'metrics_note':'Alpha is rendered using white override on black; IoU thresholds are diagnostics, not training loss or fitted alignment corrections. Projection tolerance is T9 1 pixel.'}
    selected = {}
    for key in ['A','B']:
        entry=package['configurations'][key];scene=root/entry['scene_directory']
        meta=json.loads((scene/'metadata.json').read_text());poses=joblib.load(scene/'calibrate.pkl')
        assert meta['serial_numbers']==entry['camera_ids']==intr['configurations'][key]['camera_ids']==ext['configurations'][key]['camera_ids']
        report['configurations'][key]={'camera_ids':entry['camera_ids'],'scene':str(scene),'order_verified':True}
        ply=scene/'pi3/gs/point_cloud/iteration_10000/point_cloud.ply'
        assert sha(ply)==package['initial_gaussian']['output_sha256']
        for i,camid in enumerate(entry['camera_ids']):
            k=np.array(meta['intrinsics'][i]);pose=poses[i]
            assert np.array_equal(k,np.array(intr['configurations'][key]['K'][i]))
            assert np.array_equal(pose,np.array(ext['configurations'][key]['c2w'][i]))
            rgb=scene/'color'/str(i)/'0.png';mask=scene/'mask'/str(i)/'1/0.png'
            item=(k,pose,ply,rgb,mask)
            if camid in selected:
                old=selected[camid]
                assert np.array_equal(k,old[0]) and np.array_equal(pose,old[1])
                assert all(sha(item[j])==sha(old[j]) for j in [2,3,4])
            else:selected[camid]=item
    a.output.mkdir(parents=True)
    panels=[]
    with torch.no_grad():
        for camid,(k,c2w,ply,rgb_path,mask_path) in selected.items():
            pc=GaussianModel(0);pc.load_ply(str(ply))
            xyz=pc.get_xyz.detach().cpu().numpy().astype(np.float64)
            w2c=np.linalg.inv(c2w)
            cam=cams.Camera(R=w2c[:3,:3].T,T=w2c[:3,3],
                FoVx=2*math.atan(640/(2*k[0,0])),FoVy=2*math.atan(360/(2*k[1,1])),
                img_path=str(rgb_path),static_img_path=str(rgb_path),img_hw=(360,640))
            cam.to_device('cuda')
            pipe=renderer.RenderPipe();bg=torch.zeros(3,device='cuda')
            r=renderer.render_gaussian_physdreamer(cam,pc,pipe,bg)['render']
            alpha=renderer.render_gaussian_physdreamer(cam,pc,pipe,bg,
                override_color=torch.ones_like(pc.get_xyz))['render'][0]
            assert torch.isfinite(r).all() and torch.isfinite(alpha).all()
            image=r.permute(1,2,0).cpu().numpy();al=alpha.cpu().numpy()
            rgb=np.asarray(Image.open(rgb_path).convert('RGB'));mask=np.asarray(Image.open(mask_path))>0
            # Independent float64 pinhole, versus actual SoMA float32 camera matrix.
            homogeneous=np.c_[xyz,np.ones(len(xyz))]
            camera=homogeneous@w2c.T
            proj=camera[:,:3]@k.T;uv=proj[:,:2]/proj[:,2:3]
            clip=(torch.tensor(homogeneous,dtype=torch.float32,device='cuda')@cam.full_proj_transform).cpu().numpy()
            ndc=clip[:,:2]/(clip[:,3:4]+1e-7)
            soma_uv=((ndc+1)*np.array([640,360])-1)/2
            err=np.linalg.norm(uv-soma_uv,axis=1)
            inside=(camera[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<360)
            assert err.max()<1.,'T9 projection tolerance exceeded'
            metrics={}
            for threshold in [0.1,0.5,0.9]:
                fg=al>=threshold
                metrics[str(threshold)]={'IoU':float((fg&mask).sum()/(fg|mask).sum()),
                    'precision':float((fg&mask).sum()/max(fg.sum(),1)),
                    'recall':float((fg&mask).sum()/mask.sum()),'foreground_pixels':int(fg.sum())}
            fg=al>=0.5
            center=lambda m:np.argwhere(m).mean(axis=0)[::-1]
            mirrored={}
            for label,m in [('horizontal_flip',fg[:,::-1]),('vertical_flip',fg[::-1,:])]:
                mirrored[label]=float((m&mask).sum()/(m|mask).sum())
            report['cameras'][camid]={'input_sha256':{'PLY':sha(ply),'RGB':sha(rgb_path),'mask':sha(mask_path)},
                'gaussian_count':len(xyz),'positive_depth_count':int((camera[:,2]>0).sum()),
                'in_image_count':int(inside.sum()),'projection_max_error_pixels':float(err.max()),
                'projection_mean_error_pixels':float(err.mean()),'uv_min':uv.min(axis=0).tolist(),'uv_max':uv.max(axis=0).tolist(),
                'alpha_mask_metrics':metrics,'mask_centroid_xy':center(mask).tolist(),'render_centroid_xy':center(fg).tolist(),
                'centroid_delta_xy_pixels':(center(fg)-center(mask)).tolist(),'flipped_render_IoU':mirrored,
                'render_min_max':[float(image.min()),float(image.max())],'finite':True}
            rendered=np.uint8(np.clip(image,0,1)*255)
            overlay=np.uint8(0.5*rgb+0.5*rendered)
            contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(overlay,contours,-1,(0,255,0),1)
            contours2=cv2.findContours(fg.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(overlay,contours2,-1,(255,0,255),1)
            mask_rgb=np.repeat((mask*255).astype(np.uint8)[...,None],3,axis=2)
            panel=Image.new('RGB',(1280,768),'#202020');draw=ImageDraw.Draw(panel)
            for j,(label,img) in enumerate([('RGB source 113',rgb),('SoMA SH0 render',rendered),('Object mask',mask_rgb),('Overlay: green=mask magenta=alpha0.5',overlay)]):
                x=(j%2)*640;y=(j//2)*384
                panel.paste(Image.fromarray(img),(x,y+24));draw.text((x+8,y+5),label,fill='white')
            panel.save(a.output/f'{camid}_comparison.jpg',quality=90)
            Image.fromarray(rendered).save(a.output/f'{camid}_render.png')
            projection=rgb.copy()
            for u,v in uv[inside][::40]:cv2.circle(projection,(int(round(u)),int(round(v))),2,(0,255,255),-1)
            Image.fromarray(projection).save(a.output/f'{camid}_projection.jpg',quality=90)
            panels.append((camid,panel))
            del pc,cam,r,alpha
            torch.cuda.empty_cache()
    grid=Image.new('RGB',(960,606*3),'#202020');draw=ImageDraw.Draw(grid)
    for i,(camid,panel) in enumerate(panels):
        grid.paste(panel.resize((960,576)),(0,i*606+30));draw.text((8,i*606+8),camid,fill='white')
    grid.save(a.output/'comparison_grid.jpg',quality=90)
    assert all(sha(repo/f)==h for f,h in hashes.items())
    report['artifacts']={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in a.output.iterdir() if f.is_file()}
    report['numeric_projection_status']='PASS'
    report['visual_review_status']='pending human/assistant image inspection; numerical metrics alone are not final geometry PASS'
    (a.output/'projection_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['cameras'],indent=2))
    print('SLURM',report['slurm_job_id'],'OUTPUT',a.output)


if __name__=='__main__':main()
