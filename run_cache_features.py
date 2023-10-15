from pathlib import Path
from loguru import logger
import json
import shutil
import libtmux
import hashlib


def run():
    print("#!/bin/bash")
    print("set -e")
    print()
    num_workers = 16
    # for experiment in ("brca_subtype",):
    for dataset in ("_tcga_brca",):
        for feature_extractor in (
            "ctranspath",
            "owkin",
            "swin",
            "vit",
            "retccl",
            "resnet50",
            "bt",
            "swav",
            "dino_p16",
        ):
            for augmentations in ("simple_rotate", "all"):  # only do the expensive ones
                cmd = f"env/bin/python -m histaug.train.cache dataset={dataset} augmentations@dataset.augmentations={augmentations} settings.feature_extractor={feature_extractor} dataset.num_workers={num_workers}"
                print("echo ====================")
                print(f"echo RUNNING: {cmd}")
                print("echo ====================")
                print(cmd)


if __name__ == "__main__":
    run()
