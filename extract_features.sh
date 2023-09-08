#!/bin/bash

MODELS=(ctranspath retccl resnet50)
GPUS=(3 4 5)
OUT_DIR="/app/results"

mkdir -p $OUT_DIR

session="extract"
tmux new-session -d -s $session

for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    gpu=${GPUS[$i]}

    if [ "$i" != "0" ]; then
        tmux split-window -v -t $session
        tmux select-layout -t $session tiled
    fi

    cmd="CUDA_VISIBLE_DEVICES=$gpu env/bin/python -m histaug.extract_features --model $model --output $OUT_DIR/kather100k_$model.h5"
    tmux send-keys -t $session "$cmd" ENTER
done

tmux select-layout -t $session even-vertical
tmux attach-session -t $session