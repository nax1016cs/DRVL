# The new config inherits a base config to highlight the necessary modification
_base_ = 'cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)),
        )

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nuclei',)
runner = dict(type='EpochBasedRunner', max_epochs=200)
test_pipelines = [
   dict(type='LoadImageFromFile'),
   dict(
       type='MultiScaleFlipAug',
       img_scale=(1333, 800),
       flip=True,
       transforms=[
           dict(type='Resize', keep_ratio=True),
           dict(type='RandomFlip'),
           dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
           dict(type='Pad', size_divisor=32),
           dict(type='DefaultFormatBundle'),
           dict(type='Collect', keys=['img']),
       ])
   ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='/eva_data/ming/DRVL/HW3/detection_data/full_train/',
        classes=classes,
        ann_file='/eva_data/ming/DRVL/HW3/nuclei.json'),
    val=dict(
        img_prefix='/eva_data/ming/DRVL/HW3/detection_data/full_train/',
        classes=classes,
        ann_file='/eva_data/ming/DRVL/HW3/nuclei.json'),
    test=dict(
        img_prefix='/eva_data/ming/DRVL/HW3/detection_data/test/',
        classes=classes,
        ann_file='/eva_data/ming/DRVL/HW3/detection_data/annotations/instance_test.json'),
        # pipelines = [
        # dict(type='LoadImageFromFile'),
        # dict(
        #     type='MultiScaleFlipAug',
        #     img_scale=(1333, 800),
        #     flip=False,
        #     transforms=[
        #         dict(type='Resize', keep_ratio=True),
        #         dict(type='RandomFlip'),
        #         dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1]),
        #         dict(type='Pad', size_divisor=32),
        #         dict(type='DefaultFormatBundle'),
        #         dict(type='Collect', keys=['img']),
        #     ])
        # ]
        )

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/eva_data/ming/DRVL/HW3/mmdetection/ckpt/cascade_mask_rcnn_x101_32x4d_fpn_mstrain_3x_coco_20210706_225234-40773067.pth'