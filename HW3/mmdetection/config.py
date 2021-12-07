# The new config inherits a base config to highlight the necessary modification
_base_ = 'configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head = dict(num_classes=1),
        mask_head = dict(num_classes=1)
    ),
)
data_root = '/eva_data/ming/DRVL/HW3/'
# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('nuclei',)
runner = dict(type='EpochBasedRunner', max_epochs=100)

data = dict(
    samples_per_gpu = 1,
    workers_per_gpu = 1,
    train = dict(
        img_prefix = data_root + 'detection_data/full_train/',
        classes = classes,
        ann_file = data_root + 'nuclei.json'),
    val = dict(
        img_prefix = data_root + 'detection_data/full_train/',
        classes = classes,
        ann_file = data_root + 'nuclei.json'),
    test = dict(
        img_prefix = data_root +'detection_data/test/',
        classes = classes,
        ann_file = data_root + 'detection_data/annotations/instance_test.json'),
    )

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'mmdetection/ckpt/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth'