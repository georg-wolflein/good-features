from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule, Trainer
import re
import numpy as np
import pandas as pd
import torch
from textwrap import indent
from omegaconf import ListConfig

from loguru import logger

__all__ = [
    "make_dataset_df",
    "read_table",
    "make_preds_df",
    "flatten_batched_dicts",
    "summarize_dataset",
    "pathlist",
    "add_zero_vector_on_dims",
    "DummyBiggestBatchFirstCallback",
]

FEATURES_EXT = "zarr"  # "h5"


def make_dataset_df(
    *,
    clini_tables: Iterable[Path] = [],
    slide_tables: Iterable[Path] = [],
    feature_dirs: Iterable[Path],
    patient_col: str = "patient",
    filename_col: str = "filename",
    group_by: Optional[str] = None,
    target_labels: Sequence[str],
) -> pd.DataFrame:
    if slide_tables:
        slide_dfs = []
        for slide_table in slide_tables:
            slide_df = read_table(slide_table)
            slide_df = slide_df.loc[:, slide_df.columns.isin([patient_col, filename_col])]  # type: ignore

            assert filename_col in slide_df, f"{filename_col} not in {slide_table}"
            slide_df["path"] = slide_df[filename_col].map(
                lambda fn: next(
                    (
                        path
                        for f in feature_dirs
                        if (path := f / fn).exists() or (path := f / f"{fn}.{FEATURES_EXT}").exists()
                    ),
                    None,
                )
            )

            if (na_idxs := slide_df.path.isna()).any():
                logger.warning(
                    f"some slides from {slide_table} have no features: {list(slide_df.loc[na_idxs, filename_col])}"
                )
            slide_df = slide_df[~na_idxs]
            slide_dfs.append(slide_df)
            assert not slide_df.empty, f"no features for slide table {slide_table} found in {feature_dirs}"

        df = pd.concat(slide_dfs)
    else:
        # Create a table mapping slide names to their paths
        h5s = {h5 for d in feature_dirs for h5 in d.glob("*.{FEATURES_EXT}")}
        assert h5s, f"no features found in {feature_dirs}!"
        df = pd.DataFrame(list(h5s), columns=["path"])
        df[filename_col] = df.path.map(lambda p: p.name)

    # df is now a DataFrame containing at least a column "path", possibly a patient and filename column

    if clini_tables:
        assert patient_col in df.columns, f"patient column {patient_col} not found in the slide table(s) {slide_tables}"

        clini_df = pd.concat([read_table(clini_table) for clini_table in clini_tables])
        # select all the relevant available ground truths,
        # make sure there's no conflicting patient info
        clini_df = (
            # select all important columns
            clini_df.loc[:, clini_df.columns.isin([patient_col, group_by, *target_labels])]  # type: ignore
            .drop_duplicates()
            .set_index(patient_col, verify_integrity=True)
        )
        # TODO assert patient_col in clini_df, f"no column named {patient_col} in {clini_df}"
        df = df.merge(clini_df.reset_index(), on=patient_col)
        assert not df.empty, "no match between slides and clini table"

    # At this point we have a dataframe containing
    # - h5 paths
    # - the corresponding slide names
    # - the patient id (if a slide table was given)
    # - the ground truths for the target labels present in the clini table

    group_by = group_by or patient_col if patient_col in df else filename_col

    # Group paths and metadata by the specified column
    grouped_paths_df = df.groupby(group_by)[["path"]].aggregate(list)
    grouped_metadata_df = df.groupby(group_by).first().drop(columns=["path", filename_col], errors="ignore")
    df = grouped_metadata_df.join(grouped_paths_df)

    return df


def read_table(table: Union[Path, pd.DataFrame], dtype=str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table

    if table.suffix == ".csv":
        return pd.read_csv(table, dtype=dtype, low_memory=False)  # type: ignore
    else:
        return pd.read_excel(table, dtype=dtype)  # type: ignore


def make_preds_df(
    predictions: Mapping[str, torch.Tensor],
    *,
    base_df: pd.DataFrame,
    categories: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    target_pred_dfs = []
    for target_label, cat_labels in categories.items():
        target_pred_df = pd.DataFrame(
            predictions[target_label],
            columns=[f"{target_label}_{cat}" for cat in cat_labels],
            index=base_df.index,
        )
        hard_prediction = np.array(cat_labels)[predictions[target_label].argmax(dim=1)]
        target_pred_df[f"{target_label}_pred"] = hard_prediction

        target_pred_dfs.append(target_pred_df)

    preds_df = pd.concat(
        [base_df.loc[:, base_df.columns.isin(categories.keys())], *target_pred_dfs],
        axis=1,
    ).copy()
    return preds_df


def flatten_batched_dicts(dicts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # `trainer.predict` gives us a bunch of dictionaries, with `target_labels`
    # as keys and `batch_size` predictions each.  We reconstruct it into a
    # single dict with `target_labels` as the keys and _all_ predictions for
    # that label as values.
    keys = list(dicts[0].keys())

    return {k: torch.cat([x[k] for x in dicts]) for k in keys}


class DummyBiggestBatchFirstCallback(Callback):
    """A callback that runs a forward pass with a big batch of dummy data.

    This is useful when each batch contains a different number of instances, so that the model can allocate enough memory for the biggest batch.
    This may fix CUDA out of memory errors.
    """

    def __init__(self, dummy_batch):
        self.dummy_batch = dummy_batch

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(
            f"Running a forward pass with a big batch of dummy data to allocate enough memory for the biggest batch..."
        )
        pl_module.train()
        batch = pl_module.transfer_batch_to_device(self.dummy_batch, pl_module.device, -1)
        loss = pl_module.step(batch, step_name=None)
        loss.backward()
        pl_module.zero_grad()


def pathlist(paths: Sequence[str]) -> Sequence[Path]:
    return [Path(p) for p in paths]


def add_zero_vector_on_dims(x: torch.Tensor, dims: Sequence[int]):
    """Add a zero vector to x on the specified dimensions."""
    for dim in dims:
        x = torch.cat([x, torch.zeros(*x.shape[:dim], 1, *x.shape[dim + 1 :]).type_as(x)], dim=dim)
    return x


def summarize_dataset(targets: ListConfig, df: pd.DataFrame):
    ret = f"Number of patients: {len(df)}\n"
    for target in targets:
        target_info = f"Target: {target.column} ({target.type})\n"
        if target.type == "categorical":
            value_counts = df[target.column].value_counts(normalize=True)
            inv_value_counts = 1.0 / value_counts
            inv_value_counts /= inv_value_counts.sum()
            weights = target.weights or ([1.0] * len(target.classes))
            weights = pd.Series(weights, index=target.classes) / sum(weights)
            merged = (
                value_counts.to_frame("freq")
                .join(inv_value_counts.to_frame("inv freq"), how="outer")
                .join(weights.to_frame("weight"), how="outer")
            )
            target_info += f"  {merged.to_string(float_format=lambda x: f'{x:.4f}')}\n"
        elif target.type == "continuous":
            target_info += df[target.column].describe().to_string()
        ret += indent(target_info, "  ")
    return ret
