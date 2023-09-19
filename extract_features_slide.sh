#!/bin/bash

if [ "$1" = "" ]; then
    echo "Usage: $0 <model> <gpu1> <gpu2> ..."
    exit 1
fi

model="$1" # "ctranspath" "swin" "retccl" "resnet50" "owkin" "vit"

GPUS=("${@:2}")
CMDS=(
    "env/bin/python -m histaug.extract_features.slide_dataset --model $model --start 0 --end 400"
    "env/bin/python -m histaug.extract_features.slide_dataset --model $model --start 400 --end 800"
    "env/bin/python -m histaug.extract_features.slide_dataset --model $model --start 800"
)

session="extract-$model"
tmux new-session -d -s $session

for i in "${!CMDS[@]}"; do
    cmd=${CMDS[$i]}
    gpu=${GPUS[$i]}

    if [ "$i" != "0" ]; then
        tmux split-window -v -t $session
        tmux select-layout -t $session tiled
    fi

    tmux send-keys -t $session "CUDA_VISIBLE_DEVICES=$gpu $cmd" ENTER
done

tmux select-layout -t $session even-vertical
tmux attach-session -t $session