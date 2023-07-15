import torch

_base_ = [
    '../../_base_/models/isanet_r50-d8.py',
    '../../_base_/datasets/mmseg/pascal_context_59_512x512.py',
    '../../_base_/mmseg_runtime.py',
    '../../_base_/schedules/mmseg/schedule_80k.py',
    '../../_base_/wandb_logger.py',
    '../../_base_/upload_cloudreve_config.py',
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

if torch.cuda.device_count() > 1:
    # start customized evaluation
    val_interval = 4000
    evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=False, save_best='mIoU', gpu_collect=True)
    data = dict(samples_per_gpu=4, workers_per_gpu=2)
    log_config = {{_base_.customized_log_config}}
else:
    val_interval = 500
    evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=False, save_best='mIoU')
    # debugging
    data = dict(samples_per_gpu=2, workers_per_gpu=0, persistent_workers=False)

workflow = [('train', val_interval), ('val', 1)]
checkpoint_config = dict(_delete_=True)
