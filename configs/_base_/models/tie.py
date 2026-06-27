# Only the decoder is changed: DDPMDecoder -> DDIMDecoder
model = dict(
    type='GsSimulatorHierarchy',
    cluster_cfg=dict(
        carnations=[
            dict(downsample_rate=0.01), # Init level
            dict(downsample_rate=0.01),],
    ),
    processor_cfg=[
        dict(
            type='GsHieDynamicDGLProcessor',
            anchor_prefix='anchor_',
            radius=[
                0.01, 0.01*10,
                0.01*100
            ], # TODO: DIFFERENT HERE, must match the cluster_cfg and the one in dataset
            group_cfg=dict(
                max_radius=None, # If use 0.6, ball_query will not garentee all possible nodes; only None will use KNN can search all possible nodes
                min_radius=0.0,
                sample_num=8, # TODO: DIFFERENT HERE
                use_xyz=True,
                normalize_xyz=False,
                return_grouped_xyz=False,
                return_grouped_idx=True,
                return_unique_cnt=False,),
        ),],
    accumulate_gradient=False,
    dt=1/30,
    backbone=dict(
        type='TIE',
        attr_dim=5,
        state_dim=6, # pos, vel
        position_dim=3,
        num_frames=2,
        embed_dims=128,
        num_heads=8,
        num_encoder_layers=16,
        dropout=0.0,
        eps=1e-7,
        num_fcs=2,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='LN'),
        pre_norm=True,
        # dt=1.0/30,
        norm_acc_steps=None,
        ),
    decode_head=dict(
        type='AccDecoder',
        # out_channels is position_dim
        in_channels=128, # particle, patch, human
        out_channels=4+3+4,
        add_residual=False,
        init_quant=1.0,
        # dt=1.0/30,
        loss_decode=[
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_momentum'),
            dict(type='SSIMLoss', reduction='sum', kernel_size=5, loss_weight=1.0, loss_name='loss_ssim_render'),
            # dict(type='L1Loss', reduction='sum', loss_weight=1.0, loss_name='loss_l1_render'),
            dict(type='L2Loss', reduction='sum', loss_weight=1.0, loss_name='loss_l2_render'),
            dict(type='MSELoss', reduction='sum', loss_weight=1.0, loss_name='loss_mse_static'),
            # dict(type='EnergyLoss', reduction='sum', loss_weight=0.5, loss_name='loss_energy_selfsup'),
            dict(type='EnergyLoss', reduction='sum', loss_weight=0.5, loss_name='loss_energy_inertia'),
            dict(type='EnergyLoss', reduction='sum', loss_weight=0.5, loss_name='loss_energy_gravity'),
            dict(type='EnergyLoss', reduction='sum', loss_weight=0.5, loss_name='loss_energy_potential'),
            ],
        accuracy=[
            dict(type='L2Accuracy', reduction='mean', acc_name='acc_l2_render'),
            dict(type='L1Accuracy', reduction='mean', acc_name='acc_l1_render'),
            ],
        ),
    gs_scene=[
        # # Must align with those in dataset. e.g., load_cluster_mask
        # dict(name='telephone', model_path='data/telephone/point_cloud.ply', mov_mask_path='data/telephone/pc_mask.pkl', sh_degree=3, num_seq=5),
                # Must align with those in dataset. e.g., load_cluster_mask
        dict(name='carnations', model_path='data_v2/carnations/point_cloud.ply', mov_mask_path='data_v2/carnations/pc_mask.pkl', cln_mask_path='data_v2/carnations/cln_pc_mask.pkl', sh_degree=3, num_seq=36),
    ],
    forward_last_layer=False,
    share_weight=True, # Better with edge_mode='ratio'.
    opt_sim=True,
    opt_vel=False,
    checkpoint_rollout=50, # For use of torch.utils.checkpoint.checkpoint
    selfsup_loss=False,
)