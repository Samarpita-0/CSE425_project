
"""
Task 3 report figures (simplified).

Figures:
  ablation_bars.png     -- macro-F1 / micro-F1 / AUC-PR per method
  loss_curves.png       -- 2x2 subplots: train+val loss per method
  lr_schedule.png       -- LR schedule per method

Also displayed inline when imported from a notebook.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path("/content/CSE425_project")
PLOTS = ROOT / "results/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

METHODS = ["bert_only", "gnn_only", "concat", "cross_attention"]


# loaders
def load_metrics() -> pd.DataFrame:
    rows = []
    for m in METHODS:
        p = ROOT / f"models/{m}_graphsage/test_metrics.json"
        if p.exists():
            rows.append({"method": m, **json.loads(p.read_text())})
    return pd.DataFrame(rows)


def _read_scalar(log_dir: Path, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if not log_dir.exists():
        return None, None
    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return None, None
    ev = acc.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def fig_1(df: pd.DataFrame):
    metrics = ["macro_f1", "micro_f1", "auc_pr"]
    labels  = ["Macro-F1", "Micro-F1", "AUC-PR"]
    x = np.arange(len(df)); w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        bars = ax.bar(x + i*w, df[m], width=w, label=lab)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(df["method"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, df[metrics].max().max() * 1.2)
    ax.legend(loc="upper left")
    ax.set_title("Fusion Method Ablation (Test Set)")
    plt.tight_layout()
    plt.savefig(PLOTS / "ablation_bars.png", dpi=150)
    return fig


def fig_2():
    """One subplot per method: train loss (dashed) vs val loss (solid)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    plotted = 0

    for ax, m in zip(axes, METHODS):
        ld = ROOT / f"runs/{m}_graphsage"
        s_tr, tr = _read_scalar(ld, "train/loss")
        s_va, va = _read_scalar(ld, "val/loss")

        if tr is None and va is None:
            ax.set_title(f"{m} — no logs")
            ax.text(0.5, 0.5, "missing TensorBoard logs",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray")
            continue

        if tr is not None:
            ax.plot(s_tr, tr, "--", color="#1f77b4", label="train")
        if va is not None:
            ax.plot(s_va, va, "-", color="#d62728", label="val")

        if tr is not None:
            ax.scatter([s_tr[-1]], [tr[-1]], color="#1f77b4", zorder=5)
            ax.annotate(f"{tr[-1]:.3f}", (s_tr[-1], tr[-1]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        if va is not None:
            ax.scatter([s_va[-1]], [va[-1]], color="#d62728", zorder=5)
            ax.annotate(f"{va[-1]:.3f}", (s_va[-1], va[-1]),
                        xytext=(4, -12), textcoords="offset points", fontsize=8)

        ax.set_title(m)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plotted += 1

    fig.suptitle("Train / Val Loss Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS / "loss_curves.png", dpi=150)
    if plotted == 0:
        print("no loss scalars found — check runs/<method>_graphsage/")
    return fig


def fig_3():
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for m in METHODS:
        ld = ROOT / f"runs/{m}_graphsage"
        for tag in ["train/lr", "lr", "learning_rate"]:
            s, v = _read_scalar(ld, tag)
            if s is not None:
                ax.plot(s, v, marker="o", label=m)
                plotted = True
                break
    if not plotted:
        ax.text(0.5, 0.5, "no LR scalar found\n"
                "(add writer.add_scalar('train/lr', ...) in train.py)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        ax.legend()
    ax.set_title("LR schedule (ReduceLROnPlateau)")
    plt.tight_layout()
    plt.savefig(PLOTS / "lr_schedule.png", dpi=150)
    return fig


# main
def main(show: bool = True):
    print("=" * 60)
    print("Generating Report Figures")
    print("=" * 60)

    figs = {}

    df = load_metrics()
    if df.empty:
        print("no test_metrics.json found — cannot build 1")
    else:
        print(f"methods with metrics: {df['method'].tolist()}")
        figs["1 — Ablation bars"] = fig_1(df)
        print("  ablation_bars.png")

    figs["2 — Train/Val loss"] = fig_2()
    print("  loss_curves.png")

    figs["3 — LR schedule"] = fig_3()
    print("  lr_schedule.png")

    if show:
        try:
            from IPython.display import display
            for title, fig in figs.items():
                print(f"\n### {title}")
                display(fig)
        except ImportError:
            print("\n(IPython not available — figures saved to disk only)")

    print("\n" + "=" * 60)
    print(f"All figures written to {PLOTS}")
    print("=" * 60)


if __name__ == "__main__":
    main(show=True)
