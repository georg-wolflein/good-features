#!/bin/bash

if [ "$1" = "" ]; then
    echo "Usage: $0 <model> <gpu1> <gpu2> ..."
    exit 1
fi

model="$1" # "ctranspath" "swin" "retccl" "resnet50" "owkin" "vit"

# dataset="/data/shiprec/camelyon16/training"
# output="/data/histaug/features/camelyon16"
# dataset="/data/shiprec/TCGA-BRCA"
# output="/data/histaug/features/TCGA-BRCA"
# dataset="/data/shiprec/CPTAC-BRCA"
# output="/data/histaug/features/CPTAC-BRCA"
# dataset="/raid/shiprec/CAMELYON17" # RAID!!!
# output="/data/histaug/features/CAMELYON17"
# dataset="/data/shiprec/TCGA-CRC"
# output="/data/histaug/features/TCGA-CRC"
# dataset="/data/shiprec/CPTAC-CRC"
# output="/data/histaug/features/CPTAC-CRC"
dataset="/data/shiprec/TCGA-BRCA_MPP0.5"
output="/data/histaug/features/TCGA-BRCA_MPP0.5"

GPUS=("${@:2}")
CMDS=(
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model"
    "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model --aug Macenko"
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model --aug Macenko --start 0 --end 300"
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model --aug Macenko --start 300 --end 600"
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model --aug Macenko --start 600 --end 900"
    # "env/bin/python -m histaug.extract_features.slide_dataset --dataset $dataset --output $output --model $model --aug Macenko --start 900"
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