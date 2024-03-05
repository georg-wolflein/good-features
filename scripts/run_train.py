from pathlib import Path
from loguru import logger
import json
import shutil
import libtmux
import hashlib
from tqdm import tqdm

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
GPUS = [5, 6, 7]
GPUS = [7]

IGNORE_CONFIG_KEYS = ["early_stopping.metric", "early_stopping.goal", "dataset.num_workers"]
RAM_BOMB = True  # whether to run the RAM bomb to clear disk cache before changing feature extractors (only makes sense when using 1 GPU)


def run(dry_run: bool = False, check_wandb: bool = True):
    log_dir = Path("experiment_logs")
    success_dir = log_dir / "success"
    fail_dir = log_dir / "fail"
    pending_dir = log_dir / "pending"
    logs_dir = log_dir / "logs"

    if not dry_run:
        shutil.rmtree(pending_dir, ignore_errors=True)

    for dir in (log_dir, success_dir, fail_dir, pending_dir, logs_dir):
        dir.mkdir(exist_ok=True, parents=True)

    # Generate configs
    configs = []

    for magnification in ("low", "high"):
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
            "vits",
            "owkin_teacher",
            "mocov2",
        ):
            for experiment in (
                (
                    "brca_subtype",
                    "brca_CDH1",
                    "brca_TP53",
                    "brca_PIK3CA",
                    "camelyon17_lymph",
                    "crc_MSI",
                    "crc_KRAS",
                    "crc_BRAF",
                    "crc_SMAD4",
                )
                if magnification == "low"
                else (
                    "brca_subtype",
                    "brca_CDH1",
                    "brca_TP53",
                    "brca_PIK3CA",
                    "crc_MSI",
                    "crc_KRAS",
                    "crc_BRAF",
                    "crc_SMAD4",
                )
            ):
                for augmentations in (
                    ("none", "macenko_patchwise", "macenko_slidewise", "simple_rotate", "all")
                    if magnification == "low"
                    else ("none", "macenko_patchwise")
                ):
                    for model in ("attmil", "map", "transformer") if magnification == "low" else ("attmil",):
                        for seed in range(5):
                            config = {
                                "+experiment": experiment,
                                "+feature_extractor": feature_extractor,
                                "augmentations@dataset.augmentations": augmentations,
                                "model": model,
                                "seed": seed,
                            }
                            if "camelyon" in experiment:
                                config["settings.camelyon17_fold"] = seed
                                config["early_stopping.metric"] = "'val/${dataset.targets[0].column}/auroc'"
                                config["early_stopping.goal"] = "max"
                            if magnification == "high":
                                config["+magnification"] = "high"
                                config["dataset.num_workers"] = 20
                            configs.append(config)

    logger.info(f"Generated {len(configs)} configs")

    def get_hash_for_config(config):
        return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

    with (log_dir / "tasks.txt").open("w") as f:
        for config in configs:
            f.write(f"{get_hash_for_config(config)} {' '.join(f'{k}={v}' for k, v in config.items())}\n")
    logger.info(f"Generated task list")

    if check_wandb:
        import wandb

        api = wandb.Api()

        runs = (run for run in api.runs("histaug", order="+created_at", per_page=1000) if run.state == "finished")
        for run in tqdm(runs, desc="Checking wandb"):
            run_overrides = {k: v for (k, v) in [x.split("=") for x in run.config.get("overrides", "").split(" ")]}
            configs = [
                config
                for config in configs
                if not all(
                    run_overrides.get(k, None) == str(v)
                    for k, v in config.items()
                    if k not in IGNORE_CONFIG_KEYS and v is not None
                )
            ]
        logger.info(f"Removed configs that were already completed on wandb; {len(configs)} remaining")
    else:
        # Remove already completed tasks
        for path in success_dir.glob("*.json"):
            with path.open() as f:
                config = json.load(f)
            if config in configs:
                configs.remove(config)
        logger.info(f"Removed configs that were already completed; {len(configs)} remaining")

    if dry_run:
        for config in configs:
            print(" ".join(f"{k}={v}" for k, v in config.items()))
        logger.info(f"Total: {len(configs)} configs")
    else:
        # Generate task files
        configs_and_paths = []
        for config in configs:
            hashed = get_hash_for_config(config)
            path = pending_dir / f"{hashed}.json"
            with path.open("w") as f:
                json.dump(config, f)
            configs_and_paths.append((config, path))
        logger.info(f"Generated task files")

        # Split tasks across workers
        configs_per_worker = [[] for _ in range(len(GPUS))]
        for i, (config, path) in enumerate(configs_and_paths):
            configs_per_worker[i % len(configs_per_worker)].append((config, path))

        for worker, (gpu, configs) in enumerate(zip(GPUS, configs_per_worker)):
            script = log_dir / f"run_{worker}.sh"
            with script.open("w") as f:
                f.write("#!/bin/bash\n")
                previous_feature_extractor = None
                for i, (config, path) in enumerate(configs, 1):
                    if RAM_BOMB and config["+feature_extractor"] != previous_feature_extractor:
                        f.write("echo\n")
                        f.write("echo ========================================\n")
                        f.write(f"echo Running RAM bomb to clear disk cache\n")
                        f.write("echo ========================================\n")
                        f.write("echo\n")
                        f.write("python scripts/rambomb.py\n")
                        f.write("\n")
                        previous_feature_extractor = config["+feature_extractor"]
                    cmd = f"CUDA_VISIBLE_DEVICES={gpu} python -m histaug.train.oneval {' '.join(f'{k}={v}' for k, v in config.items())}"
                    f.write("echo\n")
                    f.write("echo ========================================\n")
                    f.write(f"echo Running [{i}/{len(configs)}] {path.stem}: {cmd}\n")
                    f.write("echo ========================================\n")
                    f.write("echo\n")
                    f.write(f"{cmd} 2>&1 | tee -a {logs_dir / path.with_suffix('.txt').name}\n")
                    f.write("status=${PIPESTATUS}\n")
                    f.write(f"if [ $status -eq 0 ]; then\n")
                    f.write(f'    mv "{path}" {success_dir}\n')
                    f.write("else\n")
                    f.write(f'    mv "{path}" {fail_dir}\n')
                    f.write("fi\n")
                    f.write("\n")
            script.chmod(0o755)
        logger.info(f"Generated scripts")

        # Run scripts using tmux
        server = libtmux.Server()
        session = server.new_session("train")
        for worker, gpu in enumerate(GPUS):
            window = session.new_window(window_name=f"worker{worker}-gpu{gpu}")
            # send keys to window
            window.select_pane(0).send_keys(f"source env/bin/activate")
            window.select_pane(0).send_keys(f"sleep {worker*2} && source {log_dir / f'run_{worker}.sh'}")

        logger.info(f"Started tmux session")

        # Attach to session
        session.attach_session()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-check-wandb", action="store_true")
    args = parser.parse_args()

    run(dry_run=args.dry_run, check_wandb=not args.no_check_wandb)
