set -x
checkpoint_name="bpkd_80k_iasnet_r18_cityscapes.pth"
checkpoint_url=""
config_name="configs/baseline/isanet/isanet_r18-d8_512x1024_80k_cityscapes.py"



IFS='.' read -ra parts <<< "${checkpoint_name}"
checkpoint_name_root="${parts[0]}"
checkpoint_name_student="${checkpoint_name_root}_student.pth"

DIR=$(pwd)

cd checkpoints
aria2c -x8 $checkpoint_url -o $checkpoint_name
cd $DIR
python tools/model_converters.py checkpoints/$checkpoint_name --out-path checkpoints/student
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python tools/test.py $config_name checkpoints/student/$checkpoint_name_student --eval mIoU

echo $checkpoint_name
ls checkpoints/student/

