#!/bin/bash

if [ "$1" = "" ]; then
    echo "Usage: $0 <model> <gpu1> <gpu2> ..."
    exit 1
fi

model="$1" # "ctranspath" "swin" "retccl" "resnet50" "owkin" "vit"

aug=""

# dataset="/data/shiprec/tcga-brca"
# output="/data/histaug/features/tcga_brca"
# dataset="/data/shiprec/cptac-brca"
# output="/data/histaug/features/cptac_brca"
# dataset="/data/shiprec/camelyon17"
# output="/data/histaug/features/camelyon17"
# dataset="/data/shiprec/tcga-crc"
# output="/data/histaug/features/tcga_crc"
# dataset="/data/shiprec/cptac-crc"
# output="/data/histaug/features/cptac_crc"

dataset="/data/shiprec/tcga-brca_mpp0.5"
output="/data/histaug/features/tcga_brca_mpp0.5"
# dataset="/data/shiprec/cptac-brca_mpp0.5"
# output="/data/histaug/features/cptac_brca_mpp0.5"
# dataset="/data/shiprec/tcga-crc_mpp0.5"
# output="/data/histaug/features/tcga_crc_mpp0.5"
# dataset="/data/shiprec/cptac-crc_mpp0.5"
# output="/data/histaug/features/cptac_crc_mpp0.5"
# dataset="/data/shiprec/camelyon17_mpp0.5"
# output="/data/histaug/features/camelyon17_mpp0.5"
aug="--aug Macenko"

GPUS=("${@:2}")
CMDS=(
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug"
    "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug --start 0 --end 100"
    "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug --start 100 --end 200"
    "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug --start 200 --end 300"
    "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug --start 300"
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug --start 300 --end 600"
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug --start 600 --end 900"
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model $aug --start 900"
)

session="extract"

# Check if session exists
tmux has-session -t $session 2>/dev/null
session_exists=$?

# If session does not exist, create it
if [ "$session_exists" != "0" ]; then
    tmux new-session -d -s $session
fi

for i in "${!CMDS[@]}"; do
    cmd=${CMDS[$i]}
    gpu=${GPUS[$i]}
    
    if [ "$session_exists" == "0" ] || [ "$i" != "0" ]; then
        tmux split-window -v -t $session
        tmux select-layout -t $session tiled
    fi  

    tmux send-keys -t $session "CUDA_VISIBLE_DEVICES=$gpu $cmd" ENTER
done

tmux select-layout -t $session even-vertical
tmux attach-session -t $session
