"""Estimator adapter that plugs benchmark uncertainty methods into the AL loop.

:func:`probly.evaluation.active_learning.active_learning_loop` calls
``model.fit(x_labeled, y_labeled)`` / ``model.predict_proba(x_pool)`` on a
numpy Estimator protocol. Our benchmark uncertainty methods (ensemble,
credal, DDU, posterior network, ...) live behind ``build_model`` and expect
a DataLoader + dispatched training path.

:class:`BenchmarkEstimator` bridges the two: each ``fit`` rebuilds the full
uncertainty method from scratch, wraps the labeled numpy subset in a
DataLoader, and dispatches to the correct training routine for each
predictor type (mirroring :mod:`probly_benchmark.train` but without the
wandb/scheduler/early-stopping machinery that AL-iteration retraining does
not need).

The adapter also exposes an optional ``uncertainty_scores`` method — when
wired, it short-circuits the AL loop's ``query_fn`` path so that
method-bundled acquisitions (mutual information on ensemble members, DDU
density, credal width, ...) can be computed directly from the predictor's
native outputs.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from probly.method.credal_ensembling import CredalEnsemblingPredictor
from probly.method.ddu import DDUPredictor
from probly.method.dropout import DropoutPredictor
from probly.method.ensemble import EnsemblePredictor
from probly.method.posterior_network import PosteriorNetworkPredictor, posterior_network
from probly_benchmark.builders import get_method
from probly_benchmark.models import LeNetLogits, TabularMLP, get_base_model
from probly_benchmark.train import _fit_ddu_density_head
from probly_benchmark.train_funcs import (
    train_epoch,
    train_epoch_cross_entropy,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _filter_params(fn: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of ``params`` accepted by ``fn``."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return params
    return {k: v for k, v in params.items() if k in sig.parameters}


def _class_counts(y: np.ndarray, num_classes: int) -> list[int]:
    """Return per-class counts for an integer label array."""
    counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    return counts.astype(int).tolist()


def build_base(spec: dict[str, Any], num_classes: int, *, use_encoder: bool = False) -> nn.Module:
    """Build a freshly-initialised base network from a ``(kind, ...)`` spec.

    Supported ``kind`` values: ``"tabular_mlp"`` (requires ``in_features``,
    optional ``hidden_dim``), ``"lenet_logits"`` (28x28 grayscale), and
    ``"resnet18"`` (32x32 RGB, local CIFAR-style ResNet). Any other string is
    forwarded to :func:`probly_benchmark.models.get_base_model`.

    Args:
        spec: Dict with key ``"kind"`` plus model-specific fields.
        num_classes: Output dimension of the classification head.
        use_encoder: When ``True``, replace the classification head with
            ``nn.Identity`` so that ``get_output_dim`` returns the penultimate
            feature dim. Used for PostNet encoders.
    """
    kind = spec["kind"]
    if kind == "tabular_mlp":
        in_features = int(spec["in_features"])
        hidden_dim = int(spec.get("hidden_dim", 1024))
        model = TabularMLP(in_features, num_classes, hidden_dim=hidden_dim)
        if use_encoder:
            model.linear = nn.Identity()
        return model
    if kind == "lenet_logits":
        model = LeNetLogits(num_classes)
        if use_encoder:
            model.linear = nn.Identity()
        return model
    if kind == "resnet18":
        base = get_base_model("resnet18", num_classes, pretrained=False)
        if use_encoder:
            base.linear = nn.Identity()
        return base
    # Fall-through to the shared registry for anything else.
    encoder_name = f"{kind}_encoder" if use_encoder else kind
    return get_base_model(encoder_name, num_classes, spec.get("pretrained", False))


def build_method_model(
    method_name: str,
    method_params: dict[str, Any],
    base: nn.Module,
    *,
    num_classes: int,
    model_type: str,
    class_counts: list[int] | None = None,
) -> nn.Module:
    """Wrap a freshly-built base network in an uncertainty method.

    Mirrors :func:`probly_benchmark.builders.build_model` but accepts a
    pre-built base / encoder (so callers that need custom construction —
    e.g. ``TabularMLP(in_features=...)`` — can plug it in without extending
    ``get_base_model``).

    Args:
        method_name: Registry name. ``"deterministic"`` returns ``base``
            unchanged.
        method_params: Keyword arguments forwarded to the method constructor.
        base: Pre-built base model (or encoder, for PostNet).
        num_classes: Number of target classes.
        model_type: Predictor type routed as ``predictor_type=`` to the method.
        class_counts: Optional per-class training counts; required for
            PostNet.
    """
    if method_name == "deterministic":
        return base
    if method_name == "posterior_network":
        counts = class_counts if class_counts is not None else [1] * num_classes
        return cast(
            "nn.Module",
            posterior_network(
                base,
                num_classes=num_classes,
                class_counts=counts,
                predictor_type=model_type,
                **_filter_params(posterior_network, method_params),
            ),
        )
    method_fn = get_method(method_name)
    return method_fn(base, predictor_type=model_type, **_filter_params(method_fn, method_params))


def _move_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    """Move a predictor (single module or ensemble-like iterable) to ``device``."""
    if isinstance(model, (EnsemblePredictor, CredalEnsemblingPredictor)):
        for member in model:
            cast("nn.Module", member).to(device)
        return model
    return model.to(device)


def _iter_train_targets(model: nn.Module) -> list[nn.Module]:
    """Return the list of nn.Modules whose parameters are optimised.

    Ensembles train one independent optimiser per member; everything else
    trains the wrapped predictor as a single unit.
    """
    if isinstance(model, (EnsemblePredictor, CredalEnsemblingPredictor)):
        return [cast("nn.Module", m) for m in model]
    return [model]


def _run_training(
    model: nn.Module,
    loader: DataLoader,
    *,
    n_epochs: int,
    optimizer_name: str,
    optimizer_params: dict[str, Any],
    device: torch.device,
    method_name: str,
    train_kwargs: dict[str, Any],
) -> None:
    """Train ``model`` for ``n_epochs`` over ``loader`` with method-aware dispatch.

    Ensembles and credal-ensemblings train each member with cross-entropy.
    Bayesian predictors derive a KL penalty from the dataset size. DDU runs a
    cross-entropy phase and then fits its density head on the training set.
    All other methods go through the flexdispatched ``train_epoch``.
    """
    targets = _iter_train_targets(model)
    optimizers = [_make_optimizer(optimizer_name, m.parameters(), optimizer_params) for m in targets]

    # Per-method training kwargs passed into the dispatched train_epoch.
    extra: dict[str, Any] = dict(train_kwargs)
    if method_name == "bayesian":
        dataset = getattr(loader, "dataset", None)
        fallback_batch = loader.batch_size or 1
        n = len(dataset) if dataset is not None else len(loader) * fallback_batch
        extra.setdefault("kl_penalty", 1.0 / max(1, n))

    uses_member_ce = isinstance(model, (EnsemblePredictor, CredalEnsemblingPredictor))

    for _ in range(n_epochs):
        for m in targets:
            m.train()
        for inputs_, targets_ in loader:
            inputs = inputs_.to(device, non_blocking=True)
            labels = targets_.to(device, non_blocking=True)
            if uses_member_ce:
                for member, opt in zip(targets, optimizers, strict=True):
                    train_epoch_cross_entropy(member, inputs, labels, opt)
            elif method_name == "deterministic":
                # A plain nn.Module has no flexdispatch registration; call the
                # cross-entropy helper directly.
                train_epoch_cross_entropy(model, inputs, labels, optimizers[0])
            else:
                train_epoch(model, inputs, labels, optimizers[0], **extra)

    if isinstance(model, DDUPredictor):
        _fit_ddu_density_head(model, loader, device, amp_enabled=False)


def _make_optimizer(name: str, params: Any, kwargs: dict[str, Any]) -> optim.Optimizer:  # noqa: ANN401
    """Thin optimiser factory covering the AL harness's needs."""
    key = name.lower()
    if key == "adam":
        return optim.Adam(params, **kwargs)
    if key == "sgd":
        return optim.SGD(params, **kwargs)
    msg = f"Unsupported optimizer for AL: {name!r}. Use 'adam' or 'sgd'."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _forward_logits(model: nn.Module, x: torch.Tensor, pred_batch_size: int) -> torch.Tensor:
    """Return raw logits concatenated over batched forward passes, CPU tensor."""
    model.eval()
    parts: list[torch.Tensor] = []
    for start in range(0, x.shape[0], pred_batch_size):
        chunk = x[start : start + pred_batch_size]
        parts.append(model(chunk).detach().cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _forward_softmax(model: nn.Module, x: torch.Tensor, pred_batch_size: int) -> torch.Tensor:
    """Return softmax probabilities on CPU for a single-pass classifier."""
    logits = _forward_logits(model, x, pred_batch_size)
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def _forward_member_softmax(model: nn.Module, x: torch.Tensor, pred_batch_size: int) -> torch.Tensor:
    """Return per-member softmax probs with shape ``(n_samples, n_members, n_classes)``."""
    members = [cast("nn.Module", m) for m in cast("Any", model)]
    parts: list[torch.Tensor] = []
    for member in members:
        member.eval()
        per_batch: list[torch.Tensor] = []
        for start in range(0, x.shape[0], pred_batch_size):
            chunk = x[start : start + pred_batch_size]
            per_batch.append(F.softmax(member(chunk), dim=-1).detach().cpu())
        parts.append(torch.cat(per_batch, dim=0))
    return torch.stack(parts, dim=1)


@torch.no_grad()
def _forward_mc_softmax(model: nn.Module, x: torch.Tensor, pred_batch_size: int, num_samples: int) -> torch.Tensor:
    """Stochastic MC forward passes with train() mode for dropout-style models."""
    model.train()  # keep dropout active
    parts: list[torch.Tensor] = []
    for _ in range(num_samples):
        per_batch: list[torch.Tensor] = []
        for start in range(0, x.shape[0], pred_batch_size):
            chunk = x[start : start + pred_batch_size]
            per_batch.append(F.softmax(model(chunk), dim=-1).detach().cpu())
        parts.append(torch.cat(per_batch, dim=0))
    return torch.stack(parts, dim=1)


@torch.no_grad()
def _forward_postnet_probs(model: nn.Module, x: torch.Tensor, pred_batch_size: int) -> torch.Tensor:
    """Return Dirichlet-mean probabilities for a posterior network."""
    model.eval()
    parts: list[torch.Tensor] = []
    for start in range(0, x.shape[0], pred_batch_size):
        chunk = x[start : start + pred_batch_size]
        alpha = model(chunk)
        probs = alpha / alpha.sum(dim=-1, keepdim=True)
        parts.append(probs.detach().cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _forward_ddu_probs(model: DDUPredictor, x: torch.Tensor, pred_batch_size: int) -> torch.Tensor:
    """Return softmax from a DDU predictor's classification head."""
    m = cast("Any", model)
    m.eval()
    parts: list[torch.Tensor] = []
    for start in range(0, x.shape[0], pred_batch_size):
        chunk = x[start : start + pred_batch_size]
        feats = m.encoder(chunk)
        logits = m.classification_head(feats)
        parts.append(F.softmax(logits, dim=-1).detach().cpu())
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class BenchmarkEstimator:
    """Sklearn-like wrapper that retrains a benchmark uncertainty method each ``fit``.

    Args:
        method_name: One of ``"deterministic"``, ``"dropout"``, ``"ensemble"``,
            ``"credal_ensembling"``, ``"posterior_network"``, ``"ddu"``. Other
            method names registered in ``builders.METHODS`` also work but have
            not been hand-verified for AL retraining.
        method_params: ``method.params`` dict from the Hydra config.
        base_model_spec: ``{"kind": "tabular_mlp", "in_features": 16}`` etc.
            See :func:`build_base` for the recognised ``kind`` values.
        num_classes: Number of target classes.
        n_epochs: Number of training epochs per ``fit`` call.
        batch_size: Mini-batch size used by the training loader.
        pred_batch_size: Mini-batch size used by every inference path.
        device: Torch device for training and inference.
        optimizer_name: ``"adam"`` or ``"sgd"``.
        optimizer_params: Optimiser kwargs (e.g. ``{"lr": 1e-3}``).
        train_kwargs: Method-specific kwargs forwarded to ``train_epoch``
            (e.g. ``{"entropy_weight": 1e-5}`` for posterior network).
        model_type: Predictor type routed to ``predictor_type=`` in method
            constructors. Typically ``"logit_classifier"``.
        num_samples: MC forward passes used by ``predict_proba`` / per-sample
            uncertainty for stochastic methods (``dropout``, ``bayesian``).
        acquisition: Optional ``(estimator, pool_x) -> scores`` callable. When
            supplied, :meth:`uncertainty_scores` delegates to it and the AL
            loop routes through that path instead of ``query_fn``.
    """

    def __init__(
        self,
        *,
        method_name: str,
        method_params: dict[str, Any] | None,
        base_model_spec: dict[str, Any],
        num_classes: int,
        n_epochs: int,
        batch_size: int,
        pred_batch_size: int,
        device: torch.device,
        optimizer_name: str = "adam",
        optimizer_params: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        model_type: str = "logit_classifier",
        num_samples: int = 1,
        acquisition: Callable[[BenchmarkEstimator, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Initialise the adapter. See class-level docstring for argument semantics."""
        self.method_name = method_name.lower()
        self.method_params = dict(method_params or {})
        self.base_model_spec = base_model_spec
        self.num_classes = num_classes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.pred_batch_size = pred_batch_size
        self.device = device
        self.optimizer_name = optimizer_name
        self.optimizer_params = dict(optimizer_params or {"lr": 1e-3})
        self.train_kwargs = dict(train_kwargs or {})
        self.model_type = model_type
        self.num_samples = max(1, int(num_samples))
        self._acquisition = acquisition
        self.model: nn.Module | None = None

        if acquisition is not None:
            # Dynamically bind so that the AL loop's ``hasattr`` check picks it
            # up only when an acquisition is configured.
            self.uncertainty_scores = self._uncertainty_scores  # type: ignore[method-assign]

    # ---------------------- training ----------------------

    def fit(self, x: np.ndarray, y: np.ndarray) -> BenchmarkEstimator:
        """Rebuild the method from scratch and train on ``(x, y)``."""
        y_int = y.astype(np.int64)
        use_encoder = self.method_name == "posterior_network"
        base = build_base(self.base_model_spec, self.num_classes, use_encoder=use_encoder)
        model = build_method_model(
            self.method_name,
            self.method_params,
            base,
            num_classes=self.num_classes,
            model_type=self.model_type,
            class_counts=_class_counts(y_int, self.num_classes) if use_encoder else None,
        )
        model = _move_to_device(model, self.device)

        x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
        y_t = torch.as_tensor(y_int, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(x_t, y_t),
            batch_size=min(self.batch_size, x_t.shape[0]),
            shuffle=True,
            drop_last=False,
        )

        _run_training(
            model,
            loader,
            n_epochs=self.n_epochs,
            optimizer_name=self.optimizer_name,
            optimizer_params=self.optimizer_params,
            device=self.device,
            method_name=self.method_name,
            train_kwargs=self.train_kwargs,
        )

        self.model = model
        return self

    # ---------------------- prediction ----------------------

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, n_classes)``.

        For ensembles and credal ensembles the per-member softmax probs are
        averaged. For dropout/bayesian a single forward pass is used (the
        ``num_samples``-pass MC path is reserved for :meth:`member_probs`).
        """
        probs = self._probs_impl(x)
        return probs.numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return argmax class predictions."""
        return self.predict_proba(x).argmax(axis=-1).astype(np.int64)

    def member_probs(self, x: np.ndarray) -> np.ndarray:
        """Return stacked probabilities of shape ``(n_samples, n_members, n_classes)``.

        For ensembles, members are individual predictors. For MC methods
        (dropout, bayesian), stochastic forward passes are treated as
        "members".
        """
        x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
        model = self._require_model()
        if isinstance(model, (EnsemblePredictor, CredalEnsemblingPredictor)):
            return _forward_member_softmax(model, x_t, self.pred_batch_size).numpy()
        if isinstance(model, DropoutPredictor) or self.method_name == "bayesian":
            return _forward_mc_softmax(model, x_t, self.pred_batch_size, self.num_samples).numpy()
        # Single-sample fall-through: add a singleton member axis.
        probs = self._probs_impl(x)
        return probs.unsqueeze(1).numpy()

    def embed(self, x: np.ndarray) -> np.ndarray:
        """Return penultimate-layer embeddings for BADGE.

        Requires the base network to expose ``.embed``. Only defined on plain
        deterministic classifiers here.
        """
        if self.method_name != "deterministic":
            msg = "embed() is only supported for method=deterministic in this harness."
            raise AttributeError(msg)
        model = self._require_model()
        if not hasattr(model, "embed"):
            msg = f"Base model {type(model).__name__} does not expose an embed() method."
            raise AttributeError(msg)
        x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            model.eval()
            parts: list[torch.Tensor] = []
            for start in range(0, x_t.shape[0], self.pred_batch_size):
                chunk = x_t[start : start + self.pred_batch_size]
                parts.append(cast("Any", model).embed(chunk).detach().cpu())
        return torch.cat(parts, dim=0).numpy()

    # ---------------------- PostNet alpha (optional) ----------------------

    def postnet_alpha(self, x: np.ndarray) -> np.ndarray:
        """Return the raw Dirichlet concentration parameters from a PostNet.

        Raises ``TypeError`` if the underlying model is not a PostNet.
        """
        model = self._require_model()
        if not isinstance(model, PosteriorNetworkPredictor):
            msg = "postnet_alpha requires a PosteriorNetworkPredictor."
            raise TypeError(msg)
        model.eval()
        x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, x_t.shape[0], self.pred_batch_size):
                chunk = x_t[start : start + self.pred_batch_size]
                parts.append(model(chunk).detach().cpu())
        return torch.cat(parts, dim=0).numpy()

    # ---------------------- DDU density (optional) ----------------------

    def ddu_log_density(self, x: np.ndarray) -> np.ndarray:
        """Return per-sample log-density from a fitted DDU density head.

        Raises AttributeError if the estimator is not a DDU predictor.
        """
        model = self._require_model()
        if not isinstance(model, DDUPredictor):
            msg = "ddu_log_density requires a DDUPredictor."
            raise TypeError(msg)
        m = cast("Any", model)
        m.eval()
        x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
        parts: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, x_t.shape[0], self.pred_batch_size):
                chunk = x_t[start : start + self.pred_batch_size]
                feats = m.encoder(chunk)
                log_probs = m.density_head(feats)
                # density_head(feats) typically returns log-density per sample or
                # per class; reduce to per-sample by taking the max (class-mixture
                # style). Callers that want a different reduction can derive it
                # from the returned encoder features instead.
                if log_probs.ndim == 2:
                    log_probs = log_probs.max(dim=-1).values
                parts.append(log_probs.detach().cpu())
        return torch.cat(parts, dim=0).numpy()

    # ---------------------- uncertainty hook ----------------------

    def _uncertainty_scores(self, x: np.ndarray) -> np.ndarray:
        """Forward to the configured acquisition callable."""
        if self._acquisition is None:
            msg = "No acquisition configured."
            raise RuntimeError(msg)
        return np.asarray(self._acquisition(self, x))

    # ---------------------- internals ----------------------

    def _require_model(self) -> nn.Module:
        if self.model is None:
            msg = "BenchmarkEstimator.fit must be called before prediction."
            raise RuntimeError(msg)
        return self.model

    def _probs_impl(self, x: np.ndarray) -> torch.Tensor:
        model = self._require_model()
        x_t = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=self.device)
        if isinstance(model, (EnsemblePredictor, CredalEnsemblingPredictor)):
            return _forward_member_softmax(model, x_t, self.pred_batch_size).mean(dim=1)
        if isinstance(model, PosteriorNetworkPredictor):
            return _forward_postnet_probs(model, x_t, self.pred_batch_size)
        if isinstance(model, DDUPredictor):
            return _forward_ddu_probs(model, x_t, self.pred_batch_size)
        if isinstance(model, DropoutPredictor):
            return _forward_mc_softmax(model, x_t, self.pred_batch_size, self.num_samples).mean(dim=1)
        # Plain deterministic or unknown: single softmax pass.
        return _forward_softmax(model, x_t, self.pred_batch_size)
