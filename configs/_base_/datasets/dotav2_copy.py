# dataset settings
dataset_type = 'DOTADataset'
data_1_5_root = '../dl_data/DOTA-v1.0-v1.5/'
data_root = '../dl_data/DOTA-v2.0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
dota_1_5_train = dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelTxt-v2.0/DOTA-v2.0_train/',
        img_prefix=data_1_5_root + 'train/images/',
        pipeline=train_pipeline)
dota_2_0_train = dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelTxt-v2.0/DOTA-v2.0_train/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline)
dota_1_5_val = dict(
        type=dataset_type,
        ann_file=data_root + 'val/labelTxt-v2.0/DOTA-v2.0_val/',
        img_prefix=data_1_5_root + 'val/images/',
        pipeline=test_pipeline)
dota_2_0_val = dict(
        type=dataset_type,
        ann_file=data_root + 'val/labelTxt-v2.0/DOTA-v2.0_val/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dota_2_0_train
    ],
    val=[
        dota_2_0_val
    ],
    test=[
        dota_2_0_val
    ]
)

