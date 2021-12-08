_base_ = 'configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    ),
)
data_root = '/eva_data/ming/test/DRVL/HW3/'
dataset_type = 'COCODataset'
classes = ('nuclei',)
runner = dict(type='EpochBasedRunner', max_epochs=100)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix=data_root + 'dataset/full_train/',
        classes=classes,
        ann_file=data_root + 'nuclei.json'
    ),
    val=dict(
        img_prefix=data_root + 'dataset/full_train/',
        classes=classes,
        ann_file=data_root + 'nuclei.json'
    ),
    test=dict(
        img_prefix=data_root + 'dataset/test/',
        classes=classes,
        ann_file=data_root + 'test_img_ids.json'
    ),
)
load_from = 'ckpt/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422_\
_segm_mAP-0.378_20200506_004702-faef898c.pth'
