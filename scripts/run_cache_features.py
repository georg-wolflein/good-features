def run():
    print("#!/bin/bash")
    print("set -e")
    print()
    num_workers = 16
    # for dataset in ("_tcga_brca",):
    # for dataset in ("_tcga_crc",):
    for dataset in ("_camelyon17",):
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
            # for augmentations in ("simple_rotate", "all"):  # only do the expensive ones
            for augmentations in ("all",):  # only do the expensive ones
                # for augmentations in ("simple_rotate",):  # only do the expensive ones
                cmd = f"env/bin/python -m histaug.train.cache dataset={dataset} augmentations@dataset.augmentations={augmentations} settings.feature_extractor={feature_extractor} dataset.num_workers={num_workers}"
                print("echo ====================")
                print(f"echo RUNNING: {cmd}")
                print("echo ====================")
                print(cmd)
                print()


if __name__ == "__main__":
    run()
