_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/mmseg/pascal_context_59.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py',
]

model = dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6)),
    decode_head=dict(in_channels=320, c1_in_channels=24,num_classes=59, channels=512),
    auxiliary_head=dict(in_channels=96, num_classes=59,channels=256)
)

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict()