import json
import os
# 1. loss; 2. build and freeze; 3. Iterbasedrunner
_base_ = [
    '../_base_/models/gs_simulator_embodied.py',
    '../_base_/datasets/gs_soma_dataloader.py',
    '../_base_/schedules/adam_hood.py',
    '../_base_/default_runtime.py'
]
data_dir='data_soma_sample/cloth_lift/' # PATH TO YOUR DATASET


n_gpu = 4
num_sample = 1
num_worker = 0
find_unused_parameters = True

# New customized option
max_cam_num=3

frame_gap=10
frame_gap_stage2=1
dt = 1/30 * frame_gap / 5

#schedule
lr_decay_steps=2
lr_step_start=16
max_epoch=16+30
dataset_times = 50
render_loss_weight = 1
by_epoch = True



#sequence length schedule
step_increase_interval = 1                         # every X epochs, increase the sequence length
step_increase_magnitude = 3                        # increase the sequence length by X
step_initial = 3                                   # initial sequence length
max_rollout_step = 1000                            # maximum sequence length
test_max_frame = 3 + max_rollout_step
test_start_frame = -1
assert test_start_frame < test_max_frame
start_frame = 0
max_seq = 1

#Model Parameters
num_encoder_layers = 16 # default 16
edge_theta=True
selfsup_loss=False
avg_loss=True

# For mov only
render_mov_only=True
sub_video_dir='color' # masked dynamic img
sub_img_dir='color'        # masked static img (initial state for gs)
tracking_pcd='track_process_data.pkl'





# Scene Parameters
scene_list = [
    "left_lift_1"
] #train

prompt_dict = {
    'object': ['cloth'],
    'obstacle': ['hand', 'robotic arm'],
}

aligned_scene = None

white_bg=False
default_rot = [0, 0, 0, 1] 
f_rot_est = lambda scene_name: json.load(open(os.path.join(data_dir, scene_name, 'scene_info.json'), 'r'))['gravity_rot_quat']
f_gs_path = lambda scene_name: f'{data_dir}{scene_name}/pi3/gs/point_cloud/iteration_10000/point_cloud.ply'
scene_attr_name = "rope"
split = [[0, 150]]
# Cluster Parameters
cluster_type='dis_split' # default ''
default_cluster_scheme = [
            dict(downsample_rate=0.02),
            dict(downsample_rate=0.2),]
default_controller_scheme = [
            dict(downsample_rate=0.5),
            dict(downsample_rate=0.5),]
default_volume_scalar = 512

use_random_background = {}
const_white_bg = {}
rot_est = {}
real_dt = {}
gs_scene = []
split_list = {'train': {}, 'test': {}}
cluster_cfg = {}
controller_cfg = {}
volume_scalar = {}

for scene_name in scene_list:
    use_random_background[scene_name] = white_bg
    const_white_bg[scene_name] = white_bg
    rot_est[scene_name] = f_rot_est(scene_name) #default_rot
    real_dt[scene_name] = dt
    gs_scene.append(dict(name=scene_name, model_path=f_gs_path(scene_name), sh_degree=0, num_seq=max_seq, attr_name=scene_attr_name))
    split_list['train'][scene_name] = split
    split_list['test'][scene_name] = split
    cluster_cfg[scene_name] = default_cluster_scheme
    controller_cfg[scene_name] = default_controller_scheme
    volume_scalar[scene_name] = default_volume_scalar




fix_bug = True
use_rotation = False
pred_vel = True
llffhold=0 # For evaluation

# Custom model
model = dict(
    pred_vel=pred_vel,
    use_rotation=use_rotation,
    fix_bug=fix_bug,
    dt=dt,
    cluster_cfg=cluster_cfg,
    controller_cfg=controller_cfg,
    checkpoint_rollout=20,
    gs_scene=gs_scene,
    processor_cfg=[
        dict(
            type='GsHieEmbodiedDGLProcessor',
            anchor_prefix='anchor_',
            radius=[
                0.03, 0.04*5,
                0.05*10
            ],
            group_cfg=dict(
                max_radius=None,
                min_radius=0.0,
                sample_num=16,
                use_xyz=True,
                normalize_xyz=False,
                return_grouped_xyz=False,
                return_grouped_idx=True,
                return_unique_cnt=False,),
        ),],
    opt_sim=True,
    opt_vel=True,
    static_loss=False,
    selfsup_loss=selfsup_loss,
    avg_loss=avg_loss,
    render_mov_only=render_mov_only,
    train_cfg=dict(
        by_epoch=by_epoch,
        step_increase_interval=step_increase_interval, # for epoch
        step_increase_magnitude=step_increase_magnitude,
        step_initial=step_initial,
        max_rollout_step=max_rollout_step,),
    test_cfg=dict(
        by_epoch=by_epoch,
        step_increase_interval=step_increase_interval, # for epoch
        step_increase_magnitude=step_increase_magnitude,
        step_initial=step_initial,
        max_rollout_step=max_rollout_step,),
    backbone=dict(
        fix_bug=fix_bug,
        num_encoder_layers=num_encoder_layers,
        edge_theta=edge_theta,
        num_fcs=3,
        ),
    decode_head=dict(
        loss_decode=[
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_momentum'),
            dict(type='SSIMLoss', reduction='sum', kernel_size=5, loss_weight=0.1 * render_loss_weight, loss_name='loss_ssim_render'),
            dict(type='L2Loss', reduction='sum', loss_weight=0.9 * render_loss_weight, loss_name='loss_l2_render'),
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_static'),
            dict(type='SpatialLoss', reduction='sum', loss_weight=1.0, loss_name='loss_spatial'),
            dict(type='L2Loss', reduction='sum', loss_weight=0.9, loss_name='loss_l2_bounding'),
            dict(type='SSIMLoss', reduction='sum', kernel_size=5, loss_weight=0.1, loss_name='loss_ssim_bounding'),
            
            ],
        ),
    )

# Custom dataset
data = dict(
    samples_per_gpu=num_sample,
    workers_per_gpu=num_worker,
    train=dict(
        times=dataset_times,
        dataset=dict(
        phase='all',
        env_cfg=dict(
            split_list=split_list['train'],
            prompt_dict=prompt_dict,
            aligned_scene=aligned_scene,
            frame_gap=frame_gap,
            tracking_pcd=tracking_pcd,
            volume_scalar=volume_scalar,
            llffhold=llffhold,
            real_dt=real_dt,
            rot_est=rot_est,
            use_random_background=use_random_background,
            const_white_bg=const_white_bg,
            cluster_type=cluster_type,
            cluster_cfg=cluster_cfg,
            controller_cfg=controller_cfg,
            scene_list=scene_list,
            data_dir=data_dir,
            sub_video_dir=sub_video_dir,
            sub_img_dir=sub_img_dir,
            max_frame=max_rollout_step+1,
            start_frame=start_frame,
            max_cam_num=max_cam_num,
            max_seq=max_seq,
            max_cam_total=6*max_seq,
        ),)),
    val=dict(
        phase='all',
        env_cfg=dict(
            split_list=split_list['train'],
            prompt_dict=prompt_dict,
            aligned_scene=aligned_scene,
            frame_gap=frame_gap,
            tracking_pcd=tracking_pcd,
            volume_scalar=volume_scalar,
            llffhold=llffhold,
            real_dt=real_dt,
            rot_est=rot_est,
            use_random_background=use_random_background,
            const_white_bg=const_white_bg,
            cluster_type=cluster_type,
            cluster_cfg=cluster_cfg,
            controller_cfg=controller_cfg,
            scene_list=scene_list,
            data_dir=data_dir,
            sub_video_dir=sub_video_dir,
            sub_img_dir=sub_img_dir,
            max_frame=test_max_frame,
            start_frame=start_frame,
            eval_start_frame=test_start_frame,
            max_cam_num=max_cam_num,
            max_seq=max_seq,
            max_cam_total=6*max_seq,
        ),),
    test=dict(
        phase='all',
        env_cfg=dict(
            split_list=split_list['test'],
            prompt_dict=prompt_dict,
            aligned_scene=aligned_scene,
            frame_gap=frame_gap,
            tracking_pcd=tracking_pcd,
            volume_scalar=volume_scalar,
            llffhold=llffhold,
            real_dt=real_dt,
            rot_est=rot_est,
            use_random_background=use_random_background,
            const_white_bg=const_white_bg,
            cluster_type=cluster_type,
            cluster_cfg=cluster_cfg,
            controller_cfg=controller_cfg,
            scene_list=scene_list,
            data_dir=data_dir,
            sub_video_dir=sub_video_dir,
            sub_img_dir=sub_img_dir,
            max_frame=test_max_frame, # 1 for ground truth
            start_frame=start_frame,
            eval_start_frame=test_start_frame,
            max_cam_num=max_cam_num, # one time only one camera image
            max_seq=max_seq,
            max_cam_total=6*max_seq, # Total number of view of cameras
        ),),
    )

optimizer = dict(type='Adam', lr=1e-4*num_sample*n_gpu, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)

assert by_epoch == True
lr_config = dict(by_epoch=by_epoch, policy='Hood', decay_rate=5e-1, decay_steps=int(step_increase_interval*lr_decay_steps), step_start=int(step_increase_interval*(lr_step_start-lr_decay_steps))) # ITER
runner = dict(type='EpochRunner', max_epochs=int(max_epoch*step_increase_interval), max_iters=None)
checkpoint_config = dict(by_epoch=by_epoch, interval=int(min(step_increase_interval, 32*6)), max_keep_ckpts=10000)
evaluation = dict(by_epoch=by_epoch, interval=int(step_increase_interval)) # For debug val