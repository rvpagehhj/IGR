# model settings
model = dict(
    type='QG_OS',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        #groups=32,
        #base_width=4,
        frozen_stages=1,   # frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=8,
        scales_per_octave=1,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)

# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
train_fm_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.00,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=1000)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.00,  # 0.05
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=100)

# dataset settings

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='PairWrapper',
        ann_file=None,  # for compatibility
        base_dataset='coco_train,got10k_train,lasot_train',   # 'fsod_train',
        base_transforms='extra_partial',
        sampling_prob=[1],
        max_size=40000,   # 30000
        max_instances=8,
        with_label=True))

# optimizer
optimizer = dict(type='SGD', lr=0.01/2, momentum=0.9, weight_decay=0.0001)  # lr=0.01
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16,22])  # 8,11
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1000, # 50
    hooks=[
        dict(type='TextLoggerHook'),
 ])
# yapf:enable
# runtime settings
total_epochs = 24   # 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/os'
load_from = None
resume_from = None
workflow = [('train', 1)]
