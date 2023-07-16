_base_ = [
    '../../_base_/models/fcn_hr18.py',
    '../../_base_/datasets/mmseg/ade20k.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py',
]

model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    decode_head=dict(num_classes=150),
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2))))
)