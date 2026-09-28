"""CPU-only comparison video from T31.1 files. No SoMA/torch/model imports."""
import argparse,hashlib,json,pathlib,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('artifact_dir',type=pathlib.Path);p.add_argument('--fps',type=int,default=10);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
 root=a.artifact_dir;manifest=root/'manifest.json'
 assert sha(manifest)==(root/'manifest.sha256').read_text().split()[0]
 m=json.loads(manifest.read_text());out=a.output or root/'rgb_gt_prediction_error.mp4'
 assert not out.exists() and 1<=a.fps<=60
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 w,h=1280,490
 cmd=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(out)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 try:
  for frame in m['frames']:
   local=frame['state']['local_frame'];source=frame['state']['source_frame'];canvas=Image.new('RGB',(w,h),'#f8fafc');draw=ImageDraw.Draw(canvas)
   label='initial (not prediction)' if local==0 else 'prediction'
   draw.text((12,7),f"Exploratory | epoch {m['checkpoint']['epoch']} | local {local:03d} / source {source} | {label}",font=font,fill='black')
   for i,c in enumerate(frame['cameras']):
    for kind,path in [('rgb',pathlib.Path(c['rgb_source'])),('mask',pathlib.Path(c['mask_source'])),('render',root/c['render'])]:
     expected=c[kind+'_sha256'] if kind!='render' else c['sha256'];assert sha(path)==expected
    with Image.open(c['rgb_source']) as im:rgb=np.asarray(im.convert('RGB'))
    with Image.open(c['mask_source']) as im:mask=np.asarray(im.convert('L'))
    with Image.open(root/c['render']) as im:pred=np.asarray(im.convert('RGB'))
    assert rgb.shape==pred.shape==(360,640,3) and mask.shape==(360,640)
    gt=np.where(mask[...,None]>0,rgb,0).astype(np.uint8)
    # Display error only, never used to replace float-domain T31 metrics.
    error=np.abs(pred.astype(np.int16)-gt.astype(np.int16)).astype(np.uint8)
    y=32+i*225
    draw.text((10,y),c['camera_id'],font=font,fill='black')
    for j,(title,img) in enumerate([('RGB',rgb),('GT = RGB x object mask',gt),('Existing render',pred),('Abs error (uint8 display)',error)]):
     draw.text((j*320+6,y+19),title,font=font,fill='black')
     canvas.paste(Image.fromarray(img).resize((320,180),Image.Resampling.LANCZOS),(j*320,y+39))
   proc.stdin.write(canvas.tobytes())
 finally:
  proc.stdin.close()
 code=proc.wait();assert code==0
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height','-of','json',str(out)],text=True))['streams'][0]
 assert int(probe['nb_read_frames'])==194 and probe['width']==w and probe['height']==h
 record={'path':str(out),'sha256':sha(out),'bytes':out.stat().st_size,'frames':194,'fps':a.fps,'manifest_sha256':sha(manifest),'layout':'2 cameras x RGB / masked GT / existing render / uint8 absolute error','no_model_execution':True,'error_is_visualization_not_rescored_metric':True,'playback_note':'10fps default is display pacing, not data timestamp/FPS'}
 sidecar=out.with_suffix('.json');sidecar.write_text(json.dumps(record,indent=2)+'\n');out.with_suffix('.sha256').write_text(sha(out)+'  '+out.name+'\n');print(json.dumps(record))
if __name__=='__main__':main()
