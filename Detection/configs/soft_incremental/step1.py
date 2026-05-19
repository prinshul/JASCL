_base_ = ['../_base_/models/faster-rcnn_r50_fpn.py',
          '../_base_/schedules/schedule_1x.py',
          '../_base_/default_runtime.py']
# '../_base_/datasets/semi_coco_detection.py'

data_root = '<Path_to_Your_Root_data>/Detection_data/ood_coco/'
split_root = '<Path_to_FoSSIL_Code_Root>/Detection/ood_coco_splits/'

dataset_type = 'CocoDataset'
classes = ( 'bicycle',  'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

new_classes = ('person', 'bench', 'car')
all_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

work_dir = './work_dirs/ssl_coco_o_step1'

## change base model path
load_from = '<Path_to_Your_Base_Model>/best_coco_bbox_mAP_epoch_100.pth'

resume = False
find_unused_parameters = True

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    bgr_to_rgb=False,
    pad_size_divisor=32)
detector.backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=False),
    norm_eval=True,
    style='caffe',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=load_from,
        prefix='backbone' 
    )
)

detector.roi_head=dict(
    bbox_head=dict(
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        fc_out_channels=1024,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        num_classes=len(all_classes),
        reg_class_agnostic=False,
        roi_feat_size=7,
        type='Shared2FCBBoxHead'),
    bbox_roi_extractor=dict(
        featmap_strides=[
            4,
            8,
            16,
            32,
        ],
        out_channels=256,
        roi_layer=dict(
            output_size=7, sampling_ratio=0, type='RoIAlign'),
        type='SingleRoIExtractor'),
    type='StandardRoIHead')

model = dict(
    _delete_=True,
    type='SoftTeacher',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.7,
        cls_pseudo_thr=0.7,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseudo_bbox_wh=(1e-2, 1e-2)
    ),
    semi_test_cfg=dict(predict_on='teacher')
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),  # can be omitted for pure inference
    dict(type='PackDetInputs')
]

scale = [(1333, 800), (1333, 800)]

branch_field = ['sup', 'unsup_teacher', 'unsup_student']

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]



sup_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='MultiBranch',
        branch_field=branch_field,
        sup=dict(type='PackDetInputs'))
]


weak_pipeline = [
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]


strong_pipeline = [
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomOrder',
        transforms=[
            dict(type='RandAugment', aug_space=color_space, aug_num=1),
            dict(type='RandAugment', aug_space=geometric, aug_num=1),
        ]),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]


unsup_pipeline = [
    dict(type='LoadImageFromFile',),
    dict(type='LoadEmptyAnnotations'),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    )
]


labeled_dataset = dict(
    type='CocoDataset',
    ann_file=split_root + 'step_1/weather_train.json',
    data_root=data_root + 'weather/val2017/',
    data_prefix=dict(img=''),
    metainfo=dict(classes=new_classes),
    pipeline=sup_pipeline,
    filter_cfg=dict(filter_empty_gt=True, min_size=32)
)

unlabeled_dataset = dict(
    type='CocoDataset',
    ann_file=split_root + 'step_1/weather_unlabeled.json',
    data_root=data_root + 'weather/val2017/',
    data_prefix=dict(img=''),
    metainfo=dict(classes=new_classes),
    pipeline=unsup_pipeline,
    filter_cfg=dict(filter_empty_gt=False),
)

## change source_ratio for label/unlabeled ratio 
batch_size = 16 
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='GroupMultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[1, 4]),
    dataset=dict(type='ConcatDataset',
        datasets=[labeled_dataset, unlabeled_dataset])
)



val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        ann_file=split_root + 'step_1/weather_val.json',
        data_root=data_root + 'weather/val2017/',
        data_prefix=dict(img=''),
        metainfo=dict(classes=new_classes),
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=split_root + 'step_1/weather_val.json',
    metric=['bbox'],
    classwise=True
)


test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        ann_file=split_root + 'step_1/weather_test.json',
        data_root=data_root + 'weather/val2017/',
        data_prefix=dict(img=''),
        metainfo=dict(classes=new_classes),
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=split_root + 'step_1/weather_test.json',
    metric=['bbox'],
    classwise=True
)


'''
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                ann_file=split_root + 'base/cartoon_test.json',
                data_root=data_root + 'cartoon/val2017/',
                data_prefix=dict(img=''),
                metainfo=dict(classes=classes),
                pipeline=test_pipeline
            ),
            dict(
                type=dataset_type,
                ann_file=split_root + 'step_1/sketch_test.json',
                data_root=data_root + 'sketch/val2017/',
                data_prefix=dict(img=''),
                metainfo=dict(classes=classes),
                pipeline=test_pipeline
            )
        ]
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)


test_evaluator = dict(
    type='Evaluators',
    evaluators=[
        dict(
            type='CocoMetric',
            ann_file=split_root + 'base/cartoon_test.json',
            metric=['bbox']
        ),
        dict(
            type='CocoMetric',
            ann_file=split_root + 'step_1/sketch_test.json',
            metric=['bbox']
        )
    ]
)
'''

## Change the max_iters and val checkpoint saving interval
train_cfg = dict(type='IterBasedTrainLoop', max_iters=1000, val_interval=100)
## Change the max epochs and val checkpoint saving interval
#train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=2, val_interval=1)

val_cfg = dict(type='TeacherStudentValLoop')

custom_hooks = [dict(type='MeanTeacherHook', momentum=0.999, interval=1, skip_buffer=True)]


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=0,              
        save_best='teacher/coco/bbox_mAP', # best model on val set is saved
        rule='greater'  
    ),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)
