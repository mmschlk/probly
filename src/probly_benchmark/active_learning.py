r"""Hydra entrypoint for the active learning benchmark (paper §4.3).

One invocation runs one ``(method, dataset, acquisition, seed)`` combination
end-to-end and logs the per-iteration accuracy curve plus final NAUC to
wandb. Multiple combinations are expected to run as a Hydra multirun.

Example (tabular, single seed):

    uv run python -m probly_benchmark.active_learning \\
        recipe=mlp_openml_6 method=deterministic acquisition=margin \\
        seed=0 al.init_pool=100 al.n_iterations=2 al.per_iter=50 \\
        wandb.enabled=false

Example (multirun over seeds):

    uv run python -m probly_benchmark.active_learning -m \\
        recipe=lenet_fmnist method=dropout acquisition=mutual_info \\
        seed=0,1,2,3,4,5,6,7,8,9
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import wandb.util

from probly.evaluation.active_learning import active_learning_loop
from probly_benchmark import utils
from probly_benchmark.al_acquisitions import make_acquisition, requires_embed
from probly_benchmark.al_data import load_al_dataset
from probly_benchmark.al_estimator import BenchmarkEstimator


def _resolve_base_model_spec(cfg_base_model: DictConfig, x_train_shape: tuple[int, ...]) -> dict[str, Any]:
    """Build the ``base_model_spec`` dict, filling ``in_features`` for tabular MLPs.

    For image models the shape is implicit in the class; for ``tabular_mlp`` we
    infer ``in_features`` from the training array's last dimension.
    """
    spec_obj = OmegaConf.to_container(cfg_base_model, resolve=True)
    if not isinstance(spec_obj, dict):
        msg = f"base_model config must resolve to a dict, got {type(spec_obj).__name__}."
        raise TypeError(msg)
    spec = dict(spec_obj)
    if spec["kind"] == "tabular_mlp" and "in_features" not in spec:
        spec["in_features"] = int(x_train_shape[-1])
    return spec


@hydra.main(version_base=None, config_path="configs/", config_name="active_learning")
def main(cfg: DictConfig) -> None:
    """Run one active-learning experiment."""
    print("=== Active learning configuration ===")
    print(OmegaConf.to_yaml(cfg))

    utils.set_seed(cfg.seed)

    x_train, y_train, x_test, y_test, num_classes = load_al_dataset(cfg.dataset, seed=int(cfg.seed))
    print(f"Dataset {cfg.dataset}: x_train={x_train.shape} x_test={x_test.shape} n_classes={num_classes}")

    device = utils.get_device(cfg.get("device", None))
    print(f"Running on device: {device}")

    # --- Config resolution ---------------------------------------------------
    base_model_spec = _resolve_base_model_spec(cfg.base_model, x_train.shape)
    method_params: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.params, resolve=True) if cfg.method.get("params") else {}
    )  # ty: ignore[invalid-assignment]
    train_kwargs: dict[str, Any] = (
        OmegaConf.to_container(cfg.method.train, resolve=True) if cfg.method.get("train") else {}
    )  # ty: ignore[invalid-assignment]
    optimizer_params: dict[str, Any] = (
        OmegaConf.to_container(cfg.optimizer.params, resolve=True) if cfg.optimizer.get("params") else {}
    )  # ty: ignore[invalid-assignment]

    acq_name = str(cfg.acquisition.name)
    acquisition = make_acquisition(acq_name, {"pool_size": int(cfg.al.per_iter)})

    if requires_embed(acq_name) and cfg.method.name != "deterministic":
        msg = (
            f"Acquisition {acq_name!r} requires .embed(); only supported for method=deterministic. "
            f"Got method={cfg.method.name!r}."
        )
        raise ValueError(msg)

    # --- Estimator -----------------------------------------------------------
    estimator = BenchmarkEstimator(
        method_name=str(cfg.method.name),
        method_params=method_params,
        base_model_spec=base_model_spec,
        num_classes=num_classes,
        n_epochs=int(cfg.per_iter_epochs),
        batch_size=int(cfg.batch_size),
        pred_batch_size=int(cfg.al.pred_batch_size),
        device=device,
        optimizer_name=str(cfg.optimizer.name),
        optimizer_params=optimizer_params,
        train_kwargs=train_kwargs,
        num_samples=int(cfg.acquisition.get("num_samples", 1)),
        acquisition=acquisition,
    )

    # --- wandb ---------------------------------------------------------------
    run_id = wandb.util.generate_id()
    run = wandb.init(
        id=run_id,
        name=f"al_{cfg.method.name}_{acq_name}_{cfg.dataset}_{cfg.seed}",
        entity=cfg.wandb.get("entity", None),
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty: ignore
        mode="online" if cfg.wandb.enabled else "disabled",
        save_code=True,
        tags=["active_learning"],
    )

    # --- AL loop -------------------------------------------------------------
    _, _, scores, nauc = active_learning_loop(
        estimator,
        x_train,
        y_train,
        x_test,
        y_test,
        metric=cfg.al.metric,
        pool_size=int(cfg.al.init_pool),
        n_iterations=int(cfg.al.n_iterations),
        seed=int(cfg.seed),
        num_samples=int(cfg.acquisition.get("num_samples", 1)),
    )

    # --- Logging -------------------------------------------------------------
    _log_curve(run, scores)
    run.summary["al/nauc"] = nauc
    run.summary["al/final_score"] = scores[-1] if scores else float("nan")
    run.summary["al/n_iterations_completed"] = len(scores)

    print(f"Final NAUC = {nauc:.4f}; score curve = {scores}")
    run.finish()


def _log_curve(run: Any, scores: list[float]) -> None:  # noqa: ANN401
    """Log the per-iteration scores as a wandb table plus step-level metrics."""
    table = wandb.Table(columns=["iteration", "score"])
    for i, s in enumerate(scores):
        table.add_data(i, float(s))
        run.log({"al/iteration": i, "al/score": float(s)}, step=i)
    run.log({"al/curve": table})


if __name__ == "__main__":
    main()
