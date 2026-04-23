"""Numpy-array dataset loaders for the active learning benchmark.

The :func:`probly.evaluation.active_learning.active_learning_loop` expects
pre-materialised numpy arrays ``(x_train, y_train, x_test, y_test)`` rather
than DataLoaders. The helpers here return datasets in that shape for the
three families used in §4.3 (the paper's "evaluation ceiling" experiment):

- OpenML tabular (ids 6, 155, 156 or any other numeric id)
- Fashion-MNIST (28x28 grayscale, LeNet-compatible)
- CIFAR-10 (32x32 RGB, ResNet18-compatible)

For image datasets the arrays are normalised identically to the torchvision
transforms used elsewhere in ``probly_benchmark.data``. For tabular data,
categorical features are one-hot encoded and continuous features are
z-scored using statistics computed on the train split only.
"""

from __future__ import annotations

import ssl

import numpy as np
import torch
import torchvision
from torchvision import datasets

from probly_benchmark.paths import DATA_PATH

_CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def load_fashion_mnist_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Fashion-MNIST as ``(x_train, y_train, x_test, y_test)`` numpy arrays.

    Images are materialised as ``float32`` arrays of shape ``(N, 1, 28, 28)``
    scaled to ``[0, 1]``; labels are ``int64`` of shape ``(N,)``. This matches
    the input contract of :class:`probly_benchmark.models.LeNetLogits`.
    """
    ssl._create_default_https_context = ssl._create_unverified_context  # ty:ignore[invalid-assignment]  # noqa: SLF001
    train = datasets.FashionMNIST("~/.cache/fashion_mnist", train=True, download=True)
    test = datasets.FashionMNIST("~/.cache/fashion_mnist", train=False, download=True)
    x_train = train.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def load_cifar10_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CIFAR-10 as ``(x_train, y_train, x_test, y_test)`` numpy arrays.

    Images are materialised as ``float32`` arrays of shape ``(N, 3, 32, 32)``
    normalised with the standard CIFAR-10 per-channel mean and std used
    throughout ``probly_benchmark``. Labels are ``int64`` of shape ``(N,)``.
    This matches the input contract of ``probly_benchmark.resnet.ResNet18``.
    """
    train = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True)

    def _to_chw_normalised(ds: torchvision.datasets.CIFAR10) -> np.ndarray:
        # ds.data is uint8 HWC; convert to float32 CHW in [0,1] then normalise.
        arr = np.asarray(ds.data, dtype=np.float32) / 255.0
        arr = arr.transpose(0, 3, 1, 2)
        arr -= _CIFAR10_MEAN[None, :, None, None]
        arr /= _CIFAR10_STD[None, :, None, None]
        return np.ascontiguousarray(arr)

    x_train = _to_chw_normalised(train)
    y_train = np.asarray(train.targets, dtype=np.int64)
    x_test = _to_chw_normalised(test)
    y_test = np.asarray(test.targets, dtype=np.int64)
    return x_train, y_train, x_test, y_test


def load_openml_arrays(
    dataset_id: int,
    test_split: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return an OpenML classification dataset as numpy arrays plus its class count.

    Categorical features are one-hot encoded, continuous features are z-scored
    using train-split statistics only, and rows with NaNs after encoding are
    dropped. Labels are remapped to contiguous integers ``0..n_classes-1``.
    The split is stratified on the label.

    Args:
        dataset_id: OpenML dataset id (e.g., ``6`` for letter, ``155``, ``156``).
        test_split: Fraction of rows held out for the test split.
        seed: Random seed for the stratified split.

    Returns:
        ``(x_train, y_train, x_test, y_test, n_classes)`` where ``x_*`` are
        ``float32`` of shape ``(N, n_features)`` and ``y_*`` are ``int64`` of
        shape ``(N,)``.
    """
    import openml  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    x_df, y_series, categorical_mask, _ = ds.get_data(target=ds.default_target_attribute)
    x_df = pd.DataFrame(x_df)

    cat_cols = [c for c, is_cat in zip(x_df.columns, categorical_mask, strict=True) if is_cat]
    num_cols = [c for c in x_df.columns if c not in cat_cols]

    if cat_cols:
        x_df = pd.get_dummies(x_df, columns=cat_cols, dummy_na=False)

    # Drop rows that still have NaNs after encoding; aligns y accordingly.
    mask = x_df.notna().all(axis=1)
    x_df = x_df.loc[mask]
    y_series = pd.Series(y_series).loc[mask]

    y = LabelEncoder().fit_transform(y_series.to_numpy()).astype(np.int64)
    x = x_df.to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=seed, stratify=y)

    if num_cols:
        num_col_positions = [x_df.columns.get_loc(c) for c in num_cols]
        mu = x_train[:, num_col_positions].mean(axis=0)
        sd = x_train[:, num_col_positions].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        x_train[:, num_col_positions] = (x_train[:, num_col_positions] - mu) / sd
        x_test[:, num_col_positions] = (x_test[:, num_col_positions] - mu) / sd

    n_classes = int(y.max() + 1)
    return x_train, y_train, x_test, y_test, n_classes


def load_al_dataset(
    name: str,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Dispatch ``name`` to one of the AL numpy loaders.

    Supports: ``"fashion_mnist"`` (10 classes), ``"cifar10"`` (10 classes),
    and ``"openml:<id>"`` for any OpenML classification dataset.

    Args:
        name: Dataset name; one of ``"fashion_mnist"``, ``"cifar10"``, or
            ``"openml:<id>"``.
        seed: Forwarded to :func:`load_openml_arrays` for the stratified split.

    Returns:
        ``(x_train, y_train, x_test, y_test, n_classes)``.
    """
    key = name.lower()
    if key == "fashion_mnist":
        xtr, ytr, xte, yte = load_fashion_mnist_arrays()
        return xtr, ytr, xte, yte, 10
    if key == "cifar10":
        xtr, ytr, xte, yte = load_cifar10_arrays()
        return xtr, ytr, xte, yte, 10
    if key.startswith("openml:"):
        dataset_id = int(key.split(":", 1)[1])
        return load_openml_arrays(dataset_id, seed=seed)
    msg = f"Unknown AL dataset name {name!r}. Use 'fashion_mnist', 'cifar10', or 'openml:<id>'."
    raise ValueError(msg)


def to_torch(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Small convenience for moving a numpy array into a contiguous torch tensor."""
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype)
