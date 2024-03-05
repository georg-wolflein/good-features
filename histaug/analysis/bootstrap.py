import wandb
import pandas as pd
from loguru import logger
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from torchmetrics.classification import MulticlassAUROC
from typing import Sequence
from tqdm.contrib.concurrent import process_map
from functools import partial

from histaug.utils import cached_df
from histaug.utils.display import RENAME_FEATURE_EXTRACTORS, RENAME_MODELS, RENAME_TARGETS
from histaug.analysis.collect_results import load_results, INDEX_COLS

TRAIN_DIR = Path("/data/histaug/train")
BOOTSTRAPS_DIR = Path("/data/histaug/bootstraps")
N_BOOTSTRAPS = 1000

api = wandb.Api()


def get_test_preds(run_id: str) -> pd.DataFrame:
    try:
        return pd.read_csv(next(iter(TRAIN_DIR.glob(f"*/{run_id}/test-patient-preds.csv")))).set_index("PATIENT")
    except StopIteration as e:
        raise ValueError(f"Could not find test-patient-preds.csv for {run_id}") from e


def get_classes_from_test_preds(test_preds: pd.DataFrame, target: str) -> Sequence[str]:
    return [
        c[len(target) + 1 :]  # Remove column name and underscore
        for c in test_preds.columns
        if c.startswith(f"{target}_") and c != f"{target}_pred"
    ]


def load_bootstraps(run, n_bootstraps: int = 1000):
    # Load test-patient-preds.csv for a run and generate bootstraps for it
    dataset = run.test_dataset
    column = run.target

    bootstrap_csv = BOOTSTRAPS_DIR / f"{dataset}_{column}_{n_bootstraps}.csv"
    if not bootstrap_csv.exists():
        logger.debug(f"Caching bootstraps for {dataset} {column} at {bootstrap_csv}")
        df = get_test_preds(run.wandb_id)
        patients = df.index.unique()
        classes = get_classes_from_test_preds(df, column)
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


def compute_auroc_diffs(runs_a: pd.DataFrame, runs_b: pd.DataFrame, target: str, n_bootstraps: int):
    for (_, run_a), (_, run_b) in zip(runs_a.iterrows(), runs_b.iterrows()):
        df_a = get_test_preds(run_a.wandb_id)
        df_b = get_test_preds(run_b.wandb_id)
        if df_a is None:
            logger.warning(f"Could not find test-patient-preds.csv for {run_a.wandb_id}")
            continue
        classes = get_classes_from_test_preds(df_a, target)
        # bootstraps is an array of shape (n_bootstraps, n_patients) containing the (potentially duplicated) patients in each bootstrap
        bootstraps_a = load_bootstraps(run_a, n_bootstraps=n_bootstraps)
        bootstraps_b = load_bootstraps(run_b, n_bootstraps=n_bootstraps)
        assert (bootstraps_a == bootstraps_b).all()
        for bootstrap_a, bootstrap_b in zip(bootstraps_a, bootstraps_b):
            df_bootstrap_a = df_a.loc[bootstrap_a]
            df_bootstrap_b = df_b.loc[bootstrap_b]
            yield compute_auroc(df_bootstrap_b, target, classes) - compute_auroc(df_bootstrap_a, target, classes)


def hash_df(df):
    import hashlib

    return hashlib.md5(df.to_csv().encode()).hexdigest()


def worker(df, config, comparison_column, value_a, value_b, n_bootstraps_per_config):
    filtered_runs = df
    for key, value in config.items():
        filtered_runs = filtered_runs[filtered_runs[key] == value]

    runs_a = filtered_runs[filtered_runs[comparison_column] == value_a]
    runs_b = filtered_runs[filtered_runs[comparison_column] == value_b]

    if len(runs_a) == 0 or len(runs_b) == 0:
        return None

    assert len(runs_a) == len(runs_b) == 5
    runs_a = runs_a.sort_values("seed")
    runs_b = runs_b.sort_values("seed")
    assert (runs_a.seed.values == runs_b.seed.values).all()
    return [
        {
            **{k: runs_a.iloc[0][k] for k in INDEX_COLS if k != "seed" and k != comparison_column},
            "auroc_diff": diff,
        }
        for diff in compute_auroc_diffs(
            runs_a,
            runs_b,
            target=runs_a.iloc[0].target,
            n_bootstraps=n_bootstraps_per_config,  # n_bootstraps=int(math.ceil(N_BOOTSTRAPS / len(seeds) / len(targets)))
        )
    ]


@cached_df(
    lambda runs, comparison_column, value_a, value_b, *args, **kwargs: f"bootstrapped_{comparison_column}_{value_a}_vs_{value_b}_{kwargs.get('n_bootstraps_per_config', 25)}_{hash_df(runs)}"
)
def compare_bootstraps(
    runs: pd.DataFrame, comparison_column, value_a, value_b, *, n_bootstraps_per_config: int = 25, n_workers: int = 16
):
    # runs is a dataframe of runs to compare, obtained from load_results()

    keep_fixed = ["magnification", "augmentations", "feature_extractor", "model", "target"]
    assert comparison_column in keep_fixed and comparison_column != "target"
    keep_fixed.remove(comparison_column)

    df = runs.reset_index()
    df = df[df[comparison_column].isin([value_a, value_b]) & df[comparison_column].notna()]

    configs = df[keep_fixed].drop_duplicates().to_dict(orient="records")

    fn = partial(
        worker,
        df,
        comparison_column=comparison_column,
        value_a=value_a,
        value_b=value_b,
        n_bootstraps_per_config=n_bootstraps_per_config,
    )

    if not n_workers or n_workers == 1:
        results = []
        for config in tqdm(configs, desc="Computing results"):
            results.append(fn(config))
    else:
        results = process_map(fn, configs, max_workers=n_workers, tqdm_class=tqdm, desc="Computing results")

    # Flatten results
    results = [i for r in results if r is not None for i in r]

    return pd.DataFrame(results).set_index(keep_fixed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, default="none", help="Augmentation A (default: none)")
    parser.add_argument(
        "--b", type=str, default="Macenko_slidewise", help="Augmentation B (default: Macenko_slidewise)"
    )
    parser.add_argument(
        "--magnification", type=str, default="low", choices=["low", "high"], help="Magnification (default: low)"
    )
    args = parser.parse_args()
    df = load_results()
    compare_bootstraps(
        runs=df,
        comparison_column="augmentations",
        value_a=args.a,
        value_b=args.b,
    )
