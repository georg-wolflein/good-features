import wandb
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import math
import torch
from torchmetrics.classification import MulticlassAUROC

from histaug.utils import cached_df

TRAIN_DIR = Path("/data/histaug/train")
BOOTSTRAPS_DIR = Path("/data/histaug/bootstraps")
N_BOOTSTRAPS = 1000


def filter_runs(runs, filters: dict):
    return [run for run in runs if all(getattr(run, key, None) == value for key, value in filters.items())]


def get_test_preds(run) -> pd.DataFrame:
    return pd.read_csv(next(iter(TRAIN_DIR.glob(f"*/{run.id}"))) / "test-patient-preds.csv").set_index("PATIENT")


def load_bootstraps(run, n_bootstraps: int = 1000):
    # Load test-patient-preds.csv for a run and generate bootstraps for it
    dataset = run.config["test"]["dataset"]["name"]
    column = run.config["test"]["dataset"]["targets"][0]["column"]
    classes = run.config["test"]["dataset"]["targets"][0]["classes"]

    bootstrap_csv = BOOTSTRAPS_DIR / f"{dataset}_{column}_{n_bootstraps}.csv"
    if not bootstrap_csv.exists():
        logger.debug(f"Caching bootstraps for {dataset} {column} at {bootstrap_csv}")
        df = get_test_preds(run)
        patients = df.index.unique()
        bootstraps = np.random.choice(patients, size=(n_bootstraps, len(patients)), replace=True)
        # Ensure that there is at least one instance from each class in each bootstrap
        for i, bootstrap in enumerate(bootstraps):
            while not all(df.loc[bootstrap, column].astype(str).value_counts().reindex(classes).fillna(0) > 0):
                bootstraps[i] = np.random.choice(patients, size=len(patients), replace=True)
        bootstraps_df = pd.DataFrame(bootstraps)
        BOOTSTRAPS_DIR.mkdir(parents=True, exist_ok=True)
        bootstraps_df.to_csv(bootstrap_csv, index=False, header=False)

    return pd.read_csv(bootstrap_csv, header=None).values


def compute_auroc(df, column, classes):
    auroc = MulticlassAUROC(num_classes=len(classes))

    logits = df[[f"{column}_{c}" for c in classes]].values
    y_true = df[column].map(lambda x: classes.index(str(x))).values
    auroc.update(torch.from_numpy(logits), torch.from_numpy(y_true))
    return auroc.compute().numpy()


def compute_auroc_diffs(runs_a, runs_b, n_bootstraps: int, column, classes):
    for run_a, run_b in zip(runs_a, runs_b):
        df_a = get_test_preds(run_a)
        df_b = get_test_preds(run_b)
        # bootstraps is an array of shape (n_bootstraps, n_patients) containing the (potentially duplicated) patients in each bootstrap
        bootstraps_a = load_bootstraps(run_a, n_bootstraps=n_bootstraps)
        bootstraps_b = load_bootstraps(run_b, n_bootstraps=n_bootstraps)
        assert (bootstraps_a == bootstraps_b).all()
        for bootstrap_a, bootstrap_b in zip(bootstraps_a, bootstraps_b):
            df_bootstrap_a = df_a.loc[bootstrap_a]
            df_bootstrap_b = df_b.loc[bootstrap_b]
            yield compute_auroc(df_bootstrap_b, column, classes) - compute_auroc(df_bootstrap_a, column, classes)


@cached_df(
    lambda augmentation_a, augmentation_b, *args, **kwargs: f"bootstrapped_augmentations_{augmentation_a}_vs_{augmentation_b}"
)
def compare_bootstraps(augmentation_a, augmentation_b):
    logger.info("Loading runs...")
    api = wandb.Api()
    runs = list(api.runs("histaug", order="+created_at", per_page=1000))
    runs = filter_runs(runs, {"state": "finished"})
    targets = ["BRAF", "CDH1", "KRAS", "MSI", "PIK3CA", "SMAD4", "TP53", "lymph", "subtype"]
    feature_extractors = ["ctranspath", "swin", "owkin", "vit", "resnet50", "retccl", "bt", "swav", "dino_p16"]
    models = ["Transformer", "MeanAveragePooling", "AttentionMIL"]

    configs = [(model, feature_extractor) for model in models for feature_extractor in feature_extractors]

    results = []
    for model, feature_extractor in (pbar := tqdm(configs)):
        pbar.set_description(f"{model} {feature_extractor}")
        filtered_runs = [
            run
            for run in runs
            if run.config["dataset"]["targets"][0]["column"] in targets
            and run.config["settings"]["feature_extractor"] == feature_extractor
            and run.config["model"]["_target_"] == f"histaug.train.models.{model}"
            and run.config["dataset"]["augmentations"]["name"] in [augmentation_a, augmentation_b]
        ]

        seeds = set({run.config["seed"] for run in filtered_runs})
        targets = set({run.config["dataset"]["targets"][0]["column"] for run in filtered_runs})

        # n_bootstraps = int(math.ceil(N_BOOTSTRAPS / len(seeds) / len(targets)))
        n_bootstraps = 25

        for target in targets:
            comparison_runs = {
                seed: (
                    next(
                        iter(
                            run
                            for run in filtered_runs
                            if run.config["seed"] == seed
                            and run.config["dataset"]["targets"][0]["column"] == target
                            and run.config["dataset"]["augmentations"]["name"] == augmentation_a
                        )
                    ),
                    next(
                        iter(
                            run
                            for run in filtered_runs
                            if run.config["seed"] == seed
                            and run.config["dataset"]["targets"][0]["column"] == target
                            and run.config["dataset"]["augmentations"]["name"] == augmentation_b
                        )
                    ),
                )
                for seed in seeds
            }  # dict of seed -> (run_a, run_b)
            runs_a, runs_b = zip(*comparison_runs.values())

            # Get info from a sample run
            sample_run = runs_a[0]
            classes = sample_run.config["dataset"]["targets"][0]["classes"]
            results.extend(
                {
                    "feature_extractor": feature_extractor,
                    "model": model,
                    "target": target,
                    "auroc_diff": diff,
                }
                for diff in compute_auroc_diffs(
                    runs_a, runs_b, n_bootstraps=n_bootstraps, column=target, classes=classes
                )
            )

    return pd.DataFrame(results).set_index(["feature_extractor", "model"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="none", help="Augmentation A (default: none)")
    parser.add_argument(
        "--b", type=str, default="Macenko_slidewise", help="Augmentation B (default: Macenko_slidewise)"
    )
    args = parser.parse_args()
    compare_bootstraps(args.a, args.b)
