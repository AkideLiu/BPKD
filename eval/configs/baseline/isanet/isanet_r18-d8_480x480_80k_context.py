_base_ = [
    '../../_base_/models/isanet_r50-d8.py',
    '../../_base_/datasets/mmseg/pascal_context_59.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_40k.py',
]

model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=512, isa_channels=256, channels=128, num_classes=59),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=59))

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
