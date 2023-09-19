#!/bin/bash

model="swin"

CMDS=(
    "env/bin/python -m histaug.extract_features.slide_dataset --model $model --start 0 --end 400"
    "env/bin/python -m histaug.extract_features.slide_dataset --model $model --start 400 --end 800"
    "env/bin/python -m histaug.extract_features.slide_dataset --model $model --start 800"
)

GPUS=(2 3 4)

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