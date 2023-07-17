_base_ = [
    '../../_base_/models/pspnet_r50-d8.py',
    '../../_base_/datasets/mmseg/pascal_context_59.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py',
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        num_classes=59,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(num_classes=59,in_channels=256, channels=64),
    # change the test time crop size here
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
