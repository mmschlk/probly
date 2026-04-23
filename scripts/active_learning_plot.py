"""Render the evaluation-ceiling NAUC figure from wandb AL runs.

Bars: normalised AUC (NAUC) of each uncertainty-method + acquisition
combination grouped by dataset, averaged over seeds with 95% CI.
Lines: tailored AL baselines (BADGE, margin, random on ``deterministic``)
drawn as horizontal references per dataset, matching the paper's
"evaluation ceiling" framing.

Usage:

    uv run python scripts/active_learning_plot.py \
        --entity probly --project test --out out/nauc.pdf

The script is read-only for wandb and keeps the raw per-seed records in a
DataFrame so the same data can drive appendix curves or alternative views.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAILORED_ACQUISITIONS = {"random", "margin", "badge"}


def collect_nauc_runs(entity: str, project: str, tag: str = "active_learning") -> pd.DataFrame:
    """Return a DataFrame of finished AL runs with one row per seed.

    Columns: ``method``, ``acquisition``, ``dataset``, ``seed``, ``nauc``,
    ``final_score``.
    """
    import wandb as _wandb  # noqa: PLC0415

    api = _wandb.Api()
    runs: list[dict[str, Any]] = []
    for run in api.runs(f"{entity}/{project}", filters={"tags": tag}):
        if run.state != "finished":
            continue
        summary = run.summary
        config = run.config
        runs.append(
            {
                "method": config.get("method", {}).get("name", "unknown"),
                "acquisition": config.get("acquisition", {}).get("name", "unknown"),
                "dataset": config.get("dataset", "unknown"),
                "seed": config.get("seed", -1),
                "nauc": float(summary.get("al/nauc", float("nan"))),
                "final_score": float(summary.get("al/final_score", float("nan"))),
            }
        )
    return pd.DataFrame(runs)


def plot_nauc_bars(df: pd.DataFrame, out_path: Path) -> None:
    """Save a grouped bar plot of NAUC with tailored-AL reference lines.

    Args:
        df: DataFrame from :func:`collect_nauc_runs`.
        out_path: File to save the figure to (extension determines format).
    """
    datasets = sorted(df["dataset"].unique())
    uncertainty_df = df[~df["acquisition"].isin(TAILORED_ACQUISITIONS)]
    tailored_df = df[df["acquisition"].isin(TAILORED_ACQUISITIONS)]

    methods = sorted(uncertainty_df["method"].unique())

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(4.5 * len(datasets), 4),
        sharey=True,
        squeeze=False,
    )
    for ax, ds in zip(axes[0], datasets, strict=True):
        sub = uncertainty_df[uncertainty_df["dataset"] == ds]
        means = sub.groupby("method")["nauc"].mean().reindex(methods)
        cis = sub.groupby("method")["nauc"].sem().reindex(methods) * 1.96

        ax.bar(np.arange(len(methods)), means.to_numpy(), yerr=cis.to_numpy(), capsize=3)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_title(str(ds))

        for acq in sorted(tailored_df["acquisition"].unique()):
            acq_sub = tailored_df[(tailored_df["dataset"] == ds) & (tailored_df["acquisition"] == acq)]
            if acq_sub.empty:
                continue
            ax.axhline(acq_sub["nauc"].mean(), linestyle="--", label=acq)
        ax.legend(loc="lower right", fontsize=8)

    axes[0][0].set_ylabel("NAUC")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Parse CLI args and render the NAUC figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--tag", default="active_learning")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = collect_nauc_runs(args.entity, args.project, tag=args.tag)
    if df.empty:
        print(f"No runs found with tag {args.tag!r} in {args.entity}/{args.project}.")
        return
    plot_nauc_bars(df, args.out)
    print(f"Wrote {args.out} from {len(df)} runs.")


if __name__ == "__main__":
    main()
