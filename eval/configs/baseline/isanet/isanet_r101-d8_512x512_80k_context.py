_base_ = [
    '../../_base_/models/isanet_r50-d8.py',
    '../../_base_/datasets/mmseg/pascal_context_59_512x512.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py',
]

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        num_classes=59,
    ),
    auxiliary_head=dict(
        num_classes=59,
    )
)

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)