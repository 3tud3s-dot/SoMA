model = dict(
    type='GsSimulatorEmbodied',
    cluster_cfg=dict(
        config_a_2cam=[dict(downsample_rate=0.02),
                       dict(downsample_rate=0.2)]),
    processor_cfg=[
        dict(
            type='GsHieEmbodiedDGLProcessor',
            anchor_prefix='anchor_',
            radius=[0.03, 0.2, 0.5],
            group_cfg=dict(
                max_radius=None,
                min_radius=0.0,
                sample_num=16,
                use_xyz=True,
                normalize_xyz=False,
                return_grouped_xyz=False,
                return_grouped_idx=True,
                return_unique_cnt=False))
    ],
    accumulate_gradient=False,
    dt=0.06666666666666667,
    backbone=dict(
        type='MeshGraphNetHieEmbodied',
        attr_dim=5,
        state_dim=6,
        position_dim=3,
        num_frames=2,
        embed_dims=128,
        num_encoder_layers=16,
        dropout=0.0,
        eps=1e-07,
        num_fcs=3,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='LN'),
        pre_norm=True,
        norm_acc_steps=None,
        edge_mode='ratio',
        edge_theta=True,
        fix_bug=True),
    decode_head=dict(
        type='AccDecoder',
        in_channels=128,
        out_channels=11,
        add_residual=False,
        init_quant=1.0,
        loss_decode=[
            dict(
                type='MSELoss',
                reduction='sum',
                loss_weight=1.0,
                loss_name='loss_mse_momentum'),
            dict(
                type='SSIMLoss',
                reduction='sum',
                kernel_size=5,
                loss_weight=0.1,
                loss_name='loss_ssim_render'),
            dict(
                type='L2Loss',
                reduction='sum',
                loss_weight=0.9,
                loss_name='loss_l2_render'),
            dict(
                type='MSELoss',
                reduction='sum',
                loss_weight=1.0,
                loss_name='loss_mse_static'),
            dict(
                type='SpatialLoss',
                reduction='sum',
                loss_weight=1.0,
                loss_name='loss_spatial'),
            dict(
                type='L2Loss',
                reduction='sum',
                loss_weight=0.9,
                loss_name='loss_l2_bounding'),
            dict(
                type='SSIMLoss',
                reduction='sum',
                kernel_size=5,
                loss_weight=0.1,
                loss_name='loss_ssim_bounding')
        ],
        accuracy=[
            dict(
                type='L2Accuracy', reduction='mean', acc_name='acc_l2_render'),
            dict(
                type='L1Accuracy', reduction='mean', acc_name='acc_l1_render'),
            dict(
                type='SSIMAccuracy',
                reduction='mean',
                acc_name='acc_ssim_render'),
            dict(
                type='PSNRAccuracy',
                reduction='mean',
                acc_name='acc_psnr_render')
        ]),
    gs_scene=[
        dict(
            name='config_a_2cam',
            model_path=
            '/data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/t17_scene_packages/pink_cloth_episode_0/config_a_2cam/pi3/gs/point_cloud/iteration_10000/point_cloud.ply',
            sh_degree=0,
            num_seq=1,
            attr_name='rope')
    ],
    forward_last_layer=False,
    share_weight=True,
    opt_sim=True,
    opt_vel=True,
    checkpoint_rollout=20,
    selfsup_loss=False,
    pred_vel=True,
    use_rotation=False,
    fix_bug=True,
    controller_cfg=dict(
        config_a_2cam=[dict(
            num_cluster=10), dict(downsample_rate=0.5)]),
    static_loss=False,
    avg_loss=True,
    render_mov_only=True,
    train_cfg=dict(
        by_epoch=True,
        step_increase_interval=1,
        step_increase_magnitude=3,
        step_initial=3,
        max_rollout_step=1000),
    test_cfg=dict(
        by_epoch=True,
        step_increase_interval=1,
        step_increase_magnitude=3,
        step_initial=3,
        max_rollout_step=1000))
env_cfg = dict(
    data_dir='',
    sub_video_dir='video_images',
    sub_img_dir='images',
    cam_transform_fn=None,
    scene_list=['mothorchids'],
    use_random_background=dict(mothorchids_v2=False),
    const_white_bg=dict(mothorchids_v2=True),
    resolution=[960, 540],
    scale_x_angle=1.0,
    load_imgs=False,
    volume_scalar=dict(mothorchids_v2=512),
    gravity=[0, 0, -9.8],
    rot_est=dict(mothorchids=[-0.912, 0.007, 0.015, 0.411]),
    max_cam_num=6,
    pad_cam=False,
    max_cam_total=180,
    max_frame=-1,
    start_frame=0,
    cluster_cfg=dict(
        mothorchids=[dict(downsample_rate=0.01),
                     dict(downsample_rate=0.01)]),
    dt=0.03333333333333333,
    real_dt=dict(mothorchids=0.033366700033366704, duck=0.02),
    llffhold=0)
dataset_type = 'EmbodiedDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        times=50,
        type='RepeatDataset',
        dataset=dict(
            type='EmbodiedDataset',
            phase='all',
            env_cfg=dict(
                data_dir=
                '/data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/t17_scene_packages/pink_cloth_episode_0',
                scene_list=['config_a_2cam'],
                sub_img_dir='color',
                sub_video_dir='color',
                cam_transform_fn=None,
                resolution=[360, 640],
                load_imgs=False,
                scale_x_angle=1.0,
                max_cam_num=2,
                max_cam_total=2,
                max_seq=1,
                pad_cam=False,
                llffhold=0,
                tracking_pcd='track_process_data.pkl',
                prompt_dict=dict(object=['cloth'], obstacle=[]),
                bounding_box=None,
                use_random_background=dict(config_a_2cam=False),
                const_white_bg=dict(config_a_2cam=False),
                controller_cfg=dict(config_a_2cam=[
                    dict(num_cluster=10),
                    dict(downsample_rate=0.5)
                ]),
                cluster_type='dis_split',
                gravity=[0.0, 0.0, -9.8],
                rot_est=dict(config_a_2cam=[0.0, 0.0, 0.0, 1.0]),
                dt=0.03333333333333333,
                real_dt=dict(config_a_2cam=0.06666666666666667),
                split_list=dict(config_a_2cam=[[0, 155]]),
                frame_gap=10,
                cluster_cfg=dict(config_a_2cam=[
                    dict(downsample_rate=0.02),
                    dict(downsample_rate=0.2)
                ]),
                volume_scalar=dict(config_a_2cam=512),
                aligned_scene=None,
                max_frame=1001,
                start_frame=0))),
    val=dict(
        type='EmbodiedDataset',
        phase='all',
        env_cfg=dict(
            data_dir=
            '/data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/t17_scene_packages/pink_cloth_episode_0',
            scene_list=['config_a_2cam'],
            sub_img_dir='color',
            sub_video_dir='color',
            cam_transform_fn=None,
            resolution=[360, 640],
            load_imgs=False,
            scale_x_angle=1.0,
            max_cam_num=2,
            max_cam_total=2,
            max_seq=1,
            pad_cam=False,
            llffhold=0,
            tracking_pcd='track_process_data.pkl',
            prompt_dict=dict(object=['cloth'], obstacle=[]),
            bounding_box=None,
            use_random_background=dict(config_a_2cam=False),
            const_white_bg=dict(config_a_2cam=False),
            controller_cfg=dict(config_a_2cam=[
                dict(num_cluster=10),
                dict(downsample_rate=0.5)
            ]),
            cluster_type='dis_split',
            gravity=[0.0, 0.0, -9.8],
            rot_est=dict(config_a_2cam=[0.0, 0.0, 0.0, 1.0]),
            dt=0.03333333333333333,
            real_dt=dict(config_a_2cam=0.06666666666666667),
            split_list=dict(config_a_2cam=[[0, 155]]),
            frame_gap=10,
            cluster_cfg=dict(config_a_2cam=[
                dict(downsample_rate=0.02),
                dict(downsample_rate=0.2)
            ]),
            volume_scalar=dict(config_a_2cam=512),
            aligned_scene=None,
            max_frame=1003,
            start_frame=0,
            eval_start_frame=-1)),
    test=dict(
        type='EmbodiedDataset',
        phase='all',
        env_cfg=dict(
            data_dir=
            '/data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/t17_scene_packages/pink_cloth_episode_0',
            scene_list=['config_a_2cam'],
            sub_img_dir='color',
            sub_video_dir='color',
            cam_transform_fn=None,
            resolution=[360, 640],
            load_imgs=False,
            scale_x_angle=1.0,
            max_cam_num=2,
            max_cam_total=2,
            max_seq=1,
            pad_cam=False,
            llffhold=0,
            tracking_pcd='track_process_data.pkl',
            prompt_dict=dict(object=['cloth'], obstacle=[]),
            bounding_box=None,
            use_random_background=dict(config_a_2cam=False),
            const_white_bg=dict(config_a_2cam=False),
            controller_cfg=dict(config_a_2cam=[
                dict(num_cluster=10),
                dict(downsample_rate=0.5)
            ]),
            cluster_type='dis_split',
            gravity=[0.0, 0.0, -9.8],
            rot_est=dict(config_a_2cam=[0.0, 0.0, 0.0, 1.0]),
            dt=0.03333333333333333,
            real_dt=dict(config_a_2cam=0.06666666666666667),
            split_list=dict(config_a_2cam=[[0, 155]]),
            frame_gap=10,
            cluster_cfg=dict(config_a_2cam=[
                dict(downsample_rate=0.02),
                dict(downsample_rate=0.2)
            ]),
            volume_scalar=dict(config_a_2cam=512),
            aligned_scene=None,
            max_frame=1003,
            start_frame=0,
            eval_start_frame=-1)))
optimizer = dict(
    type='Adam', lr=0.0004, betas=(0.9, 0.999), weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
checkpoint_config = dict(interval=1, by_epoch=True, max_keep_ckpts=10000)
log_config = dict(interval=100, hooks=[dict(type='CusTextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
seed = 5
max_seq = 1
find_unused_parameters = True
lr_config = dict(
    by_epoch=True, policy='Hood', decay_rate=0.5, decay_steps=2, step_start=14)
runner = dict(type='EpochRunner', max_epochs=46, max_iters=None)
evaluation = dict(by_epoch=True, interval=1)
work_dir = '/data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage1/config_a_seed5/'
