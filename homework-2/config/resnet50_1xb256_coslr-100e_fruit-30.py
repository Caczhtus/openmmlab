# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=30,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    train_cfg=dict(
        augments=[
            dict(type='Mixup', alpha=0.7),
            dict(type='CutMix', alpha=1.0)
        ],
        probs=[0.3, 0.7]
    ),
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='checkpoint/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
    )
)

# dataset settings
dataset_type = 'CustomDataset'
dataset_root = 'data/fruit-30'
data_preprocessor = dict(
    num_classes=30,
    # RGB format normalization parameters
    mean=[0.43706022, 0.56941104, 0.65829831],
    std=[0.26386994, 0.23573034, 0.22187396],
    # convert image from BGR to RGB
    to_rgb=True,
)

view_pipeline = [
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8
    ),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)
    ),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5
    ),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root + '/train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root + '/val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root + '/val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)


# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(
        type='CheckpointHook',
        interval=-1,
        save_best='accuracy/top1',
        by_epoch=True),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=3407, deterministic=False)

work_dir = 'work_dir/resnet50/resnet50_1xb256_coslr-100e_fruit-30'