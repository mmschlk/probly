"""Acquisition callables for the active learning benchmark.

Every acquisition has the same signature: ``(estimator, pool_x) -> scores``
where ``scores`` is a ``numpy.ndarray`` of shape ``(n_pool,)`` with the
higher-is-more-uncertain convention used by
:func:`probly.evaluation.active_learning.active_learning_loop`.

Two broad categories:

- *Tailored AL baselines* (work on any classifier): ``random``, ``margin``,
  ``badge``. BADGE needs the number of samples to select, which is closed
  over inside :func:`make_acquisition` because the AL loop does not pass
  it to the score function.
- *Method-bundled uncertainty acquisitions*: ``entropy`` (mean predictive
  entropy), ``mutual_info`` (predictive-minus-expected entropy),
  ``credal_width`` (member-level MI spread), ``ddu_density`` (negative log
  density from a DDU density head), ``postnet_precision`` (negative Dirichlet
  precision from a posterior network).

The acquisition name and any tunables (``pool_size`` for BADGE,
``num_samples`` for MC-style entropy/MI) are resolved by
:func:`make_acquisition` from the Hydra ``acquisition`` config group.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from probly.evaluation.active_learning._utils import (
    badge_query,
    margin_sampling,
    total_entropy,
)

if TYPE_CHECKING:
    from probly_benchmark.al_estimator import BenchmarkEstimator


type Acquisition = Callable[["BenchmarkEstimator", np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Tailored AL baselines (operate on any classifier)
# ---------------------------------------------------------------------------


def random_acquisition(_est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
    """Uniform random scores. Higher-is-better ordering is irrelevant under ties."""
    rng = np.random.default_rng()
    return rng.random(pool_x.shape[0])


def margin_acquisition(est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
    """Negative margin between the top two predicted class probabilities.

    Uses ``predict_proba`` on the underlying classifier (single mean prob
    distribution for ensembles; single softmax pass for deterministic).
    """
    probs = est.predict_proba(pool_x)
    return margin_sampling(probs[:, np.newaxis, :])


def make_badge_acquisition(pool_size: int) -> Acquisition:
    """Return a BADGE acquisition closed over the AL loop's per-iter query size.

    BADGE k-means++s in gradient-embedding space to select ``pool_size``
    samples per iteration. We convert the discrete selection into a ranked
    score array so the AL loop's ``lexsort`` picks the same indices in order.
    """

    def _badge(est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
        embeddings = est.embed(pool_x)
        probs = est.predict_proba(pool_x)
        n = min(pool_size, pool_x.shape[0])
        selected = badge_query(embeddings, probs, n)
        scores = np.zeros(pool_x.shape[0], dtype=np.float64)
        scores[selected] = np.arange(len(selected), 0, -1, dtype=np.float64)
        return scores

    return _badge


# ---------------------------------------------------------------------------
# Method-bundled uncertainty acquisitions
# ---------------------------------------------------------------------------


def entropy_acquisition(est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
    """Entropy of the mean predictive distribution.

    For ensembles and MC methods this is computed over ``member_probs``
    so that per-member diversity is captured; for single-pass methods it
    reduces to the entropy of one softmax.
    """
    probs = est.member_probs(pool_x)  # (n_pool, n_members, n_classes)
    return total_entropy(probs)


def mutual_info_acquisition(est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
    """Predictive entropy minus expected entropy of members (BALD / MI).

    Requires a stochastic predictor (ensemble, credal ensemble, dropout).
    """
    probs = est.member_probs(pool_x)  # (n_pool, n_members, n_classes)
    eps = np.finfo(probs.dtype).eps
    mean_probs = probs.mean(axis=1)
    predictive_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, eps, 1.0)), axis=-1)
    member_entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=-1).mean(axis=1)
    return predictive_entropy - member_entropy


def credal_width_acquisition(est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
    """Spread of per-member predictive entropy as a credal-set width proxy.

    Rewards samples where member predictions disagree most, in the spirit
    of credal-set imprecision. The specific functional (max minus min of
    per-member predictive entropy) is a practical surrogate; see the paper's
    appendix for the full motivation.
    """
    probs = est.member_probs(pool_x)  # (n_pool, n_members, n_classes)
    eps = np.finfo(probs.dtype).eps
    per_member_entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=-1)  # (n_pool, n_members)
    return per_member_entropy.max(axis=1) - per_member_entropy.min(axis=1)


def ddu_density_acquisition(est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
    """Negative log-density from a fitted DDU density head (higher = more OOD-like)."""
    log_density = est.ddu_log_density(pool_x)
    return -log_density


def postnet_precision_acquisition(est: BenchmarkEstimator, pool_x: np.ndarray) -> np.ndarray:
    """Negative Dirichlet precision as the PostNet uncertainty score.

    A low ``alpha.sum()`` means the posterior is dispersed and the model is
    uncertain; taking the negation keeps the higher-is-more-uncertain
    convention of the AL loop.
    """
    alpha = est.postnet_alpha(pool_x)
    return -alpha.sum(axis=-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_badge(cfg: dict[str, Any]) -> Acquisition:
    return make_badge_acquisition(pool_size=int(cfg["pool_size"]))


_BUILDERS: dict[str, Callable[[dict[str, Any]], Acquisition]] = {
    "random": lambda _cfg: random_acquisition,
    "margin": lambda _cfg: margin_acquisition,
    "badge": _build_badge,
    "entropy": lambda _cfg: entropy_acquisition,
    "mutual_info": lambda _cfg: mutual_info_acquisition,
    "credal_width": lambda _cfg: credal_width_acquisition,
    "ddu_density": lambda _cfg: ddu_density_acquisition,
    "postnet_precision": lambda _cfg: postnet_precision_acquisition,
}


def make_acquisition(name: str, cfg: dict[str, Any] | None = None) -> Acquisition:
    """Resolve an acquisition name into a ``(estimator, pool_x) -> scores`` callable.

    ``cfg`` collects Hydra-side parameters (currently only ``pool_size`` for
    BADGE). Unknown parameters are ignored. Raises ``KeyError`` for unknown
    acquisition names.
    """
    key = name.lower()
    if key not in _BUILDERS:
        msg = f"Unknown acquisition {name!r}. Available: {sorted(_BUILDERS)}."
        raise KeyError(msg)
    return _BUILDERS[key](cfg or {})


def requires_embed(name: str) -> bool:
    """Whether the acquisition needs ``estimator.embed`` to be available."""
    return name.lower() == "badge"
