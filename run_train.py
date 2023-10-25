from pathlib import Path
from loguru import logger
import json
import shutil
import libtmux
import hashlib

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]


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
    for augmentations in ("none", "macenko_patchwise", "macenko_slidewise", "simple_rotate", "all"):
        for experiment in ("brca_subtype", "brca_CDH1", "brca_TP53", "brca_PIK3CA"):
            for model in ("attmil", "map", "transformer"):
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
                    if augmentations == "all" and feature_extractor in ("swav", "dino_p16"):
                        continue  # those are not yet cached
                    for seed in range(5):
                        config = {
                            "+experiment": experiment,
                            "+feature_extractor": feature_extractor,
                            "augmentations@dataset.augmentations": augmentations,
                            "model": model,
                            "seed": seed,
                        }
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
        runs = [run for run in api.runs("histaug") if run.state == "finished"]

        def run_exists_for_config(config):
            overrides = set(f"{k}={v}" for k, v in config.items())
            for run in runs:
                if set(run.config.get("overrides", "").split(" ")) == overrides:
                    logger.debug(f"Config {' '.join(f'{k}={v}' for k, v in config.items())} already exists on wandb")
                    return True
            return False

        configs = [config for config in configs if not run_exists_for_config(config)]
        logger.info(f"Removed configs that were already completed on wandb; {len(configs)} remaining")

    else:
        # Remove already completed tasks
        for path in success_dir.glob("*.json"):
            with path.open() as f:
                config = json.load(f)
            if config in configs:
                configs.remove(config)
        logger.info(f"Removed configs that were already completed; {len(configs)} remaining")

    if not dry_run:
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
                for i, (config, path) in enumerate(configs, 1):
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
        session = server.new_session("histaug-experiments")
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
