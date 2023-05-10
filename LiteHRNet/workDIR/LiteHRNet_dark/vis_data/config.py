default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='PoseVisualizationHook',
        enable=True,
        show=True,
        wait_time=1,
        out_dir='showDir',
        interval=1000))
custom_hooks = [dict(type='SyncBuffersHook')]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=6)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=5,
        milestones=[2, 4],
        gamma=0.1,
        by_epoch=True)
]
auto_scale_lr = dict(base_batch_size=512, enable=True)
codec = dict(
    type='MSRAHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
    unbiased=True)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='LiteHRNet',
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=((40, 80), (40, 80, 160), (40, 80, 160, 320))),
            with_head=True)),
    head=dict(
        type='HeatmapHead',
        in_channels=40,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2,
            unbiased=True)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        rotate_factor=60,
        scale_factor=(0.75, 1.25)),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='MSRAHeatmap',
            input_size=(192, 256),
            heatmap_size=(48, 64),
            sigma=2,
            unbiased=True)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='../../../../../../train2017/'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(
                type='RandomBBoxTransform',
                rotate_factor=60,
                scale_factor=(0.75, 1.25)),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='MSRAHeatmap',
                    input_size=(192, 256),
                    heatmap_size=(48, 64),
                    sigma=2,
                    unbiased=True)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=
        'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='../../../../../../val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=
        'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='../../../../../../val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(192, 256)),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/person_keypoints_val2017.json')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/person_keypoints_val2017.json')
launcher = 'none'
work_dir = 'workDIR'
