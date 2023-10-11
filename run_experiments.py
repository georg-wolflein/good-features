configs = []
experiment = "brca_subtype"

for model in ("attmil", "map"):
    for feature_extractor in ("ctranspath", "owkin", "swin", "vit", "retccl", "resnet50", "bt", "swav", "dino_p16"):
        for augmentations in ("none", "macenko_patchwise", "simple_rotate"):
            for seed in range(5):
                config = {
                    "+experiment": experiment,
                    "+feature_extractor": feature_extractor,
                    "augmentations@dataset.augmentations": augmentations,
                    "seed": seed,
                }
                configs.append(config)

for config in configs:
    cmd = f"CUDA_VISIBLE_DEVICES=1 python -m histaug.train.oneval {' '.join(f'{k}={v}' for k, v in config.items())}"
    print(cmd)
