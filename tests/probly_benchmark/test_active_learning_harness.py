"""Integration tests for the active-learning benchmark harness.

Exercises:
- `BenchmarkEstimator.fit/predict_proba/embed` for plain deterministic.
- A full `active_learning_loop` iteration for deterministic+margin and
  ensemble+mutual_info on synthetic tabular data so the flexdispatched
  training path (ensemble per-member CE) is hit without network access.
- The tabular numpy loader dispatcher behaviour on bad names.
- `make_acquisition` builders for every registered name.
- The NAUC plot helper on a tiny synthetic DataFrame.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("probly_benchmark")
pytest.importorskip("torch")

import pandas as pd
import torch

from probly.evaluation.active_learning import active_learning_loop
from probly.evaluation.active_learning._utils import margin_sampling
from probly_benchmark.al_acquisitions import make_acquisition, requires_embed
from probly_benchmark.al_data import load_al_dataset
from probly_benchmark.al_estimator import BenchmarkEstimator


@pytest.fixture
def synthetic_tabular() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 6)).astype(np.float32)
    y = (x[:, 0] + rng.standard_normal(300) * 0.3 > 0).astype(np.int64)
    xt = rng.standard_normal((80, 6)).astype(np.float32)
    yt = (xt[:, 0] > 0).astype(np.int64)
    return x, y, xt, yt


def test_estimator_deterministic_fit_predict_embed(synthetic_tabular):
    x, y, xt, _ = synthetic_tabular
    est = BenchmarkEstimator(
        method_name="deterministic",
        method_params=None,
        base_model_spec={"kind": "tabular_mlp", "in_features": 6, "hidden_dim": 32},
        num_classes=2,
        n_epochs=3,
        batch_size=32,
        pred_batch_size=64,
        device=torch.device("cpu"),
    )
    est.fit(x, y)
    probs = est.predict_proba(xt)
    assert probs.shape == (xt.shape[0], 2)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    preds = est.predict(xt)
    assert preds.shape == (xt.shape[0],)
    assert preds.dtype == np.int64

    embeddings = est.embed(xt)
    assert embeddings.shape == (xt.shape[0], 32)


def test_al_loop_with_deterministic_margin(synthetic_tabular):
    x, y, xt, yt = synthetic_tabular
    est = BenchmarkEstimator(
        method_name="deterministic",
        method_params=None,
        base_model_spec={"kind": "tabular_mlp", "in_features": 6, "hidden_dim": 32},
        num_classes=2,
        n_epochs=2,
        batch_size=32,
        pred_batch_size=64,
        device=torch.device("cpu"),
    )
    _, _, scores, nauc = active_learning_loop(
        est,
        x,
        y,
        xt,
        yt,
        query_fn=margin_sampling,
        metric="accuracy",
        pool_size=30,
        n_iterations=2,
        seed=0,
    )
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert np.isfinite(nauc)


def test_al_loop_with_ensemble_mutual_info(synthetic_tabular):
    x, y, xt, yt = synthetic_tabular
    est = BenchmarkEstimator(
        method_name="ensemble",
        method_params={"num_members": 3, "reset_params": True},
        base_model_spec={"kind": "tabular_mlp", "in_features": 6, "hidden_dim": 32},
        num_classes=2,
        n_epochs=2,
        batch_size=32,
        pred_batch_size=64,
        device=torch.device("cpu"),
        acquisition=make_acquisition("mutual_info"),
    )
    _, _, scores, nauc = active_learning_loop(
        est,
        x,
        y,
        xt,
        yt,
        metric="accuracy",
        pool_size=30,
        n_iterations=2,
        seed=0,
    )
    assert len(scores) == 2
    assert np.isfinite(nauc)


def test_al_loop_with_badge(synthetic_tabular):
    x, y, xt, yt = synthetic_tabular
    est = BenchmarkEstimator(
        method_name="deterministic",
        method_params=None,
        base_model_spec={"kind": "tabular_mlp", "in_features": 6, "hidden_dim": 32},
        num_classes=2,
        n_epochs=2,
        batch_size=32,
        pred_batch_size=64,
        device=torch.device("cpu"),
        acquisition=make_acquisition("badge", {"pool_size": 10}),
    )
    _, _, scores, nauc = active_learning_loop(
        est,
        x,
        y,
        xt,
        yt,
        metric="accuracy",
        pool_size=10,
        n_iterations=2,
        seed=0,
    )
    assert len(scores) == 2
    assert np.isfinite(nauc)


def test_load_al_dataset_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown AL dataset"):
        load_al_dataset("not_a_real_dataset")


def test_make_acquisition_registry_names():
    names = [
        "random",
        "margin",
        "badge",
        "entropy",
        "mutual_info",
        "credal_width",
        "ddu_density",
        "postnet_precision",
    ]
    for name in names:
        cfg = {"pool_size": 5} if name == "badge" else None
        assert callable(make_acquisition(name, cfg))
    assert requires_embed("badge")
    assert not requires_embed("margin")


def test_make_acquisition_rejects_unknown():
    with pytest.raises(KeyError):
        make_acquisition("not_real")


def test_plot_nauc_bars_runs(tmp_path: Path):
    scripts_path = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_path))
    try:
        from active_learning_plot import plot_nauc_bars
    finally:
        sys.path.pop(0)

    df = pd.DataFrame(
        {
            "method": ["deterministic", "deterministic", "ensemble", "ensemble", "deterministic"],
            "acquisition": ["margin", "margin", "mutual_info", "mutual_info", "random"],
            "dataset": ["openml:6"] * 5,
            "seed": [0, 1, 0, 1, 0],
            "nauc": [0.6, 0.65, 0.55, 0.58, 0.45],
            "final_score": [0.8, 0.8, 0.75, 0.76, 0.5],
        }
    )
    out = tmp_path / "nauc.png"
    plot_nauc_bars(df, out)
    assert out.exists()
