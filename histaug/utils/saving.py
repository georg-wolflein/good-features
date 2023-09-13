import torch
from pathlib import Path
from typing import List, Dict, NamedTuple, Sequence, Optional
import numpy as np
import zarr


def save_features(
    file: Path,
    feats: torch.Tensor,
    feats_augs: Dict[str, torch.Tensor],
    labels: Optional[torch.Tensor] = None,
    files: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    chunk_size: int = 2048,
):
    f = zarr.open_group(str(file), mode="w")
    if classes is not None:
        f.attrs["classes"] = classes
    if labels is not None:
        f.create_dataset("labels", data=labels.numpy(), chunks=False)
    if files is not None:
        f.create_dataset("files", data=np.array(files, dtype=str), chunks=False)
    f.create_dataset("feats", data=feats.numpy(), chunks=(chunk_size, *feats.shape[1:]))
    aug_group = f.create_group("feats_augs")
    for aug_name, feats_aug in feats_augs.items():
        aug_group.create_dataset(aug_name, data=feats_aug.numpy(), chunks=(chunk_size, *feats_aug.shape[1:]))


class LoadedFeatures(NamedTuple):
    feats: np.ndarray
    feats_augs: Dict[str, np.ndarray]
    labels: Optional[np.ndarray]
    files: Optional[np.ndarray]


def load_features(path: Path, remove_classes: Sequence[str] = ()) -> LoadedFeatures:
    f = zarr.open_group(str(path), mode="r")
    classes = np.array(f.attrs["classes"]) if "classes" in f.attrs else None

    feats = f["feats"][:]
    labels = classes[f["labels"][:]] if classes is not None and "labels" in f else None
    files = f["files"][:] if "files" in f else None

    feats_augs = {k: f["feats_augs"][k][:] for k in f["feats_augs"].keys()}

    # Remove classes
    if len(remove_classes) > 0:
        remove_mask = np.isin(labels, remove_classes)
        feats = feats[~remove_mask]
        labels = labels[~remove_mask]
        files = files[~remove_mask]
        feats_augs = {k: v[~remove_mask] for k, v in feats_augs.items()}
    return LoadedFeatures(feats=feats, feats_augs=feats_augs, labels=labels, files=files)
