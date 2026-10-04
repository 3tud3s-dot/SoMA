"""仅从 report_data.json 生成报告图表；不导入 SoMA，不访问数据集或 CUDA。
依赖已有 CPU matplotlib/numpy/ffmpeg；运行目录为本报告根目录。
"""
import json,pathlib,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,animation
from matplotlib.patches import FancyBboxPatch
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
D=json.loads((ROOT/'assets/report_data.json').read_text())
for path in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/System/Library/Fonts/STHeiti Light.ttc']:
 if pathlib.Path(path).exists():
  font_manager.fontManager.addfont(path);plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name();break
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'white','axes.titleweight':'bold'})
BLUE='#226491';GREEN='#188675';RED='#b74e48';GOLD='#be8a2d'
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=155,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def diagram(name,title,rows):
 fig,ax=plt.subplots(figsize=(12,len(rows)*1.05+1.2));ax.set(xlim=(0,12),ylim=(0,len(rows)*1.1+.8));ax.axis('off');ax.set_title(title,pad=20,fontsize=17)
 for i,(left,right,color) in enumerate(rows):
  y=(len(rows)-i-1)*1.1+.35
  ax.add_patch(FancyBboxPatch((.3,y),3.8,.72,boxstyle='round,pad=.09',fc=color,ec='none'))
  ax.text(2.2,y+.36,left,ha='center',va='center',color='white',fontsize=12)
  ax.text(4.5,y+.36,right,va='center',fontsize=11)
  if i<len(rows)-1:ax.annotate('',(2.2,y-.31),(2.2,y-.05),arrowprops=dict(arrowstyle='->',color='#777'))
 save(fig,name)
diagram('01_pipeline','从 Deform360 到连续 Gaussian 动力学推演',[
 ('官方 processed 数据','承接 raw 的官方处理结果；本项目未重新执行全部 raw 预处理',BLUE),
 ('相机、控制器、Gaussian','T7–T18：几何、时间、监督语义与场景打包',BLUE),
 ('真实 dataset / graph / 动力学','T19–T23：加载、图构建、forward / backward / 短 rollout',BLUE),
 ('Stage-1 粗时间动力学','T24–T26：恢复、数值修复、46 epochs / 2300 steps',GREEN),
 ('Stage-1 预测 cache','T27：local 0,10,…,150；后续帧来自模型预测',GREEN),
 ('Stage-2 稠密动力学','T28–T30：dense gap=1；22/61 epochs，预算未完成',GOLD),
 ('连续 rollout 探索性验证','T31：source 113→306；epoch22 中间 checkpoint',GOLD)])
diagram('02_milestones','T1–T32 里程碑（依赖顺序示意，非按耗时比例）',[
 ('T1–T6｜表示与时间','PLY/SH0、属性语义、frame/split、时间倍率',BLUE),('T7–T18｜场景契约','相机、controller、gravity、RGB/mask、场景与静态渲染',BLUE),('T19–T24.5｜可执行性','真实 sample、graph、优化器、恢复与非阻塞重复性审计',BLUE),('T25–T26.5｜粗动力学','冻结协议 → backward NaN 归因 → FP64 修复 → 正式训练完成',GREEN),('T27–T29｜稠密训练接口','修正 cache 命名 → 子窗口与真实优化器 smoke',GREEN),('T30/T30.1｜部分训练','协议冻结；入口修正；12h TIMEOUT，22/61 epochs',GOLD),('T31–T32｜探索与交付','193步探索性推演；本报告，不宣称最终 benchmark',GOLD)])
diagram('03_normalizer_causal_chain','最重要的故障链：从统计量消减到 renderer backward NaN',[
 ('FP32 累计二阶矩','z≈−39.2、真实 std≈0.003073；E[x²]−E[x]² 舍入为0',RED),('std 落到 1e−8 floor','非零残差 / 1e−8 → anchor 绝对最大值574874.875',RED),('层级网络尺度失控','位移中位数50.638m；第一级 F 最大奇异值785.772',RED),('covariance 病态','有限精度下正定性恶化 → native renderer backward 非有限',RED),('只修 Normalizer 统计精度','先转FP64再平方/求和；不改数据、重力、loss或renderer',GREEN),('分层回归通过','anchor 1.871；位移中位数0.0431m；F 1.081；全部covariance PD',GREEN)])
diagram('04_cache_provenance','训练 cache 与连续推演的状态来源必须分开',[
 ('T27 cache：0,10,…,150','frame_0 = canonical initial；其余 = Stage-1 prediction',BLUE),('Stage-2 segmented training','按窗口读对应T27 key；fresh Adam；两update flags=false',BLUE),('T31 initial：local0 / source113','canonical SH0，与T27 frame_0位置/covariance逐值一致',GREEN),('dense position：t → t+1','每一步消费前一步prediction；不读future PLY或future cache',GREEN),('每10帧在线模板更新','模板来自prediction，但prev_state重设为同一边界位置',GOLD),('跨 train/test：267 → 268','无外部位置reset；存在周期性速度历史重设的已知限制',GOLD)])
for key,name,title in [('stage1','05_stage1_training','Stage-1：完整46 epochs'),('stage2','06_stage2_partial_training','Stage-2：只完成22/61 epochs（TIMEOUT）')]:
 e=D[key];x=[z['epoch'] for z in e];fig,axs=plt.subplots(2,2,figsize=(12,7));fig.suptitle(title,fontsize=17)
 axs[0,0].plot(x,[z['loss'] for z in e],color=BLUE);axs[0,0].set(title='每 epoch 的实际总 loss 均值',ylabel='原训练 reduction / 权重')
 for suffix,label,color in [('loss_l2_render_config_a_2cam','L2',BLUE),('loss_ssim_render_config_a_2cam','SSIM',GREEN),('loss_mse_momentum_config_a_2cam','动量',GOLD)]:axs[0,1].plot(x,[z['loss_components'].get('decode.'+suffix,np.nan) for z in e],label=label,color=color)
 axs[0,1].set_yscale('log');axs[0,1].set_title('加权 loss 分量（对数轴）');axs[0,1].legend()
 axs[1,0].semilogy(x,[z['lr'] for z in e],color=GREEN);axs[1,0].set_title('实际学习率 LR')
 for k,label,color in [('peak_allocated','分配峰值',BLUE),('peak_reserved','保留峰值',GOLD)]:axs[1,1].plot(x,[z['memory'].get(k,np.nan)/2**30 for z in e],label=label,color=color)
 axs[1,1].set(title='每 epoch 记录的显存峰值',ylabel='GiB');axs[1,1].legend()
 for ax in axs.flat:ax.set_xlabel('epoch（从1计数）');ax.grid(alpha=.18)
 fig.tight_layout(rect=(0,0,1,.95));save(fig,name)
r=D['rollout'];x=np.array([z['target_source'] for z in r])
fig,axs=plt.subplots(2,2,figsize=(12,7));fig.suptitle('T31：epoch22 中间模型的193步探索性记录',fontsize=17)
for q,col in [('p50',BLUE),('p99',GREEN),('max',GOLD)]:
 axs[0,0].plot(x,[z['step_displacement_'+q+'_m'] for z in r],label=q,color=col)
 axs[0,1].plot(x,[z['from_initial_'+q+'_m'] for z in r],label=q,color=col)
axs[0,0].set(title='单步 Gaussian 位移',ylabel='m');axs[0,1].set(title='相对初始位置的位移（不是GT误差）',ylabel='m')
axs[1,0].plot(x,[z['l2'] for z in r],color=RED);axs[1,0].set(title='逐步图像 L2 指标（两相机聚合）',ylabel='既有路径输出')
axs[1,1].plot(x,[z['psnr'] for z in r],color=BLUE);axs[1,1].set(title='逐步 PSNR（两相机聚合）',ylabel='dB')
for ax in axs.flat:
 ax.axvline(267.5,ls='--',color='black',lw=1);ax.set_xlabel('目标 source frame；虚线=训练/测试边界');ax.grid(alpha=.18)
axs[0,0].legend();axs[0,1].legend();fig.tight_layout(rect=(0,0,1,.94));save(fig,'07_continuous_rollout_metrics')
fig,axs=plt.subplots(1,2,figsize=(12,4.5));fig.suptitle('相机对照：仅为已记录的渲染数值统计，不是分相机质量评分')
for cam,col in [('023_cam0',BLUE),('009_cam1',GREEN)]:
 axs[0].plot(x,[z[cam+'_render_mean'] for z in r],label=cam,color=col);axs[1].plot(x,[z[cam+'_render_max'] for z in r],label=cam,color=col)
for ax,title in zip(axs,['渲染张量均值','渲染张量最大值']):ax.set(title=title,xlabel='目标 source frame');ax.legend();ax.grid(alpha=.18);ax.axvline(267.5,color='black',ls='--',lw=1)
fig.tight_layout(rect=(0,0,1,.92));save(fig,'08_camera_render_statistics')
# A visualization of measured numbers only. It intentionally contains no synthetic predicted images.
fig,axs=plt.subplots(2,2,figsize=(11,6.2));title=fig.suptitle('',fontsize=14)
series=[('单步位移 max（m）','step_displacement_max_m',BLUE),('累计位移 p99（m），非GT误差','from_initial_p99_m',GREEN),('两相机聚合 L2 图像指标','l2',RED),('两相机聚合 PSNR（dB）','psnr',BLUE)]
lines=[]
for ax,(label,k,col) in zip(axs.flat,series):
 y=np.array([z[k] for z in r]);ax.plot(x,y,color='#ddd',lw=1);line,=ax.plot([],[],color=col,lw=2);lines.append((line,y));ax.set(title=label,xlim=(114,306),xlabel='目标 source frame');ax.set_ylim(min(0,float(y.min())*.95),float(y.max())*1.12);ax.axvline(267.5,ls='--',color='#555');ax.grid(alpha=.18)
fig.text(.5,.015,'仅回放已有数值记录，不是 RGB / GT / 预测渲染视频；每10帧 prev_state 重设限制仍存在。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.035,1,.93))
writer=animation.FFMpegWriter(fps=10,metadata={'title':'T31 数值回放：非预测渲染视频'},codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','25'])
with writer.saving(fig,str(ROOT/'assets/rollout_metrics_replay.mp4'),dpi=100):
 for i,z in enumerate(r):
  title.set_text(f"探索性数值回放 | epoch22 | source {z['current_source']}→{z['target_source']} | 步 {i+1}/193 | PD {z['PD_count']}/{z['N']}")
  for line,y in lines:line.set_data(x[:i+1],y[:i+1])
  writer.grab_frame()
plt.close(fig)
print('Generated 8 PNG/SVG figures and numeric-only MP4; no model execution.')
