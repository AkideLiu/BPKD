_base_ = [
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/mmseg/cityscapes.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(in_channels=320,
                     c1_in_channels=24,
                     channels=128,
                     num_classes=19,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                     ),
    auxiliary_head=dict(
        in_channels=96,
        channels=64,
        num_classes=19
    ))
