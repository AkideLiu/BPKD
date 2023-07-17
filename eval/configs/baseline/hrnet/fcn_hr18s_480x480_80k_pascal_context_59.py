_base_ = [
    '../../_base_/models/fcn_hr18.py',
    '../../_base_/datasets/mmseg/pascal_context_59.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py',
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    decode_head=dict(num_classes=59),
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2,)),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(320, 320))
)

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict()
