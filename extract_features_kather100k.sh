#!/bin/bash

# MODELS=(ctranspath swin retccl resnet50 owkin vit)
# GPUS=(2 3 4 5 6 7)
# MODELS=(ctranspath swin)
# MODELS=(owkin vit)
MODELS=(retccl resnet50)
GPUS=(1 1)
session="extract"

tmux new-session -d -s $session

for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    gpu=${GPUS[$i]}

    if [ "$i" != "0" ]; then
        tmux split-window -v -t $session
        tmux select-layout -t $session tiled
    fi

    cmd="CUDA_VISIBLE_DEVICES=$gpu env/bin/python -m histaug.extract_features.kather100k --model $model"
    tmux send-keys -t $session "$cmd" ENTER
done

tmux select-layout -t $session even-vertical
tmux attach-session -t $session