_base_ = [
    '../../../datasets/custom_nus-3d.py',
    '../../../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

ida_aug_conf = {
        "resize_lim": (1.0, 1.0),
        "final_dim": (640, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 50
bev_w_ = 50
bev_z_ = 16
voxel_lidar = [0.256, 0.256, 0.125]
voxel_det = [102.4 / bev_h_, 102.4 / bev_w_, 8 / bev_z_]
queue_length = 4 # each sequence contains `queue_length` frames.
key_frame = 3

early_supervision_cfg=dict(
    num_early_loss_layers=3,
    layer0_loss=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=10.0
    ),
    layer0_decoder=dict(
        type='SimpleMLPDecoder',
        num_classes=1, # only learn if voxel is occupied
        out_dim=_dim_,
    ),

    layer1_loss=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=5.0
    ),
    layer1_decoder=dict(
        type='SparseMLPDecoder',
        num_classes=1,
        out_dim=_dim_,
    ),

    layer2_loss=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=5.0
    ),
    layer2_decoder=dict(
        type='SparseMLPDecoder',
        num_classes=1,
        out_dim=_dim_,
    ),

    extra_layer0_conv=dict(
        num_extra_conv=2,
    ),
)

sparse_cfg=dict(
    in_channels=[_dim_, _dim_, _dim_, _dim_],
    out_channels=[_dim_, _dim_, _dim_, _dim_//8],
    strides=[(1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 2, 2)],
    paddings=[(0, 2, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
    kernel_sizes=[(1, 5, 5), (1, 2, 2), (2, 2, 2), (2, 2, 2)],
    subm_kernel_sizes=[3, 3, 3, (3, 3, 3)],
    act_type='relu',
    pruning_thresh=[0.02, 0.0, 0.0],
    max_keep_ratio=[0.5, 0.5, 0.6],
    norm_cfg=dict(type='naiveSyncBN1d'),
    num_attached_subm=[1, 1, 1, 0],
)

model = dict(
    type='PanoSegOccSparse',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp = True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='SparseOccupancyHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_z=bev_z_,
        voxel_lidar = voxel_lidar,
        voxel_det = voxel_det,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=False,
        as_two_stage=False,
        num_occ_classes=17,
        transformer=dict(
            type='SparseOccupancyTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            rotate_center=[bev_h_//2,bev_w_//2],
            cam_encoder=dict(
                type='OccupancyEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=bev_z_,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccupancyLayer',
                    attn_cfgs=[
                        dict(
                            type='OccTemporalAttention',
                            embed_dims=_dim_,
                            num_points= 8,
                            num_levels=1),
                        dict(
                            type='OccSpatialAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points= 8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            temporal_encoder=dict(
                    type="OccTemporalEncoder",
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    bev_z=bev_z_,
                    num_bev_queue=2, # concat:4, rnn:2
                    embed_dims=_dim_,
                    num_block=1,
                    block_type="self_attn",
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d', requires_grad=True)
                    ),
            voxel_encoder = dict( # decoder is a more reasonable name, keep unchanged for consistency
                type='SparseOccupancyDecoder',
                bev_h=bev_h_,
                bev_w=bev_w_,
                bev_z=bev_z_,
                embed_dim = _dim_,
                out_dim = _dim_//8,
                early_supervision_cfg=early_supervision_cfg,
                sparse_cfg=sparse_cfg,
                ),
            seg_decoder = dict(
                type='SparseMLPDecoder',
                num_classes = 17,
                out_dim = _dim_//8,
            ),
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='Learned3DPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            z_num_embed = bev_z_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_occupancy=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=10.0),
        loss_occupancy_aux = dict(
            type='Lovasz3DLoss',
            ignore=17,
            loss_weight=10.0),
        bg_weight = 0.0,
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        early_supervision_cfg=early_supervision_cfg,
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))
    )
)

dataset_type = 'LidarSegNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadPointsFromFile',load_dim=5,coord_type='LIDAR'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False,with_seg_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointsRangeFilter',point_cloud_range=point_cloud_range),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='RandomMultiScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d','pts_semantic_mask', 'points', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='LoadPointsFromFile',load_dim=5,coord_type='LIDAR'),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_attr_label=False,with_seg_3d=True),
    dict(type='PointsRangeFilter',point_cloud_range=point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 640),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img','points','pts_semantic_mask'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4, # 
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        key_frame = key_frame,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    step=[8, 11],
)

total_epochs = 12
evaluation = dict(interval=4, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=4)

# 12EP: 0.650