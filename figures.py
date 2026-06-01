"""figures.py — JSON->PDF figure regenerator for the CS229 report.

Reads results/eval/offline.json and results/eval/online.json and writes three
PDFs to results/figures/:

    fig02_accuracy.pdf  — FIG-02 accuracy by VP bucket (LR vs GBT × mode)
    fig07_online.pdf    — FIG-07 online win rates (agent × baseline)
    fig_roc.pdf         — FIG-11 ROC overlay for unified LR vs GBT

Usage:
    uv run python figures.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Fixed CreationDate so consecutive runs produce byte-identical PDFs.
# matplotlib's PDF backend embeds a timestamp by default; pinning it
# removes the only non-deterministic element in the output.
_PDF_CREATION_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

OFFLINE_JSON = Path("results/eval/offline.json")
ONLINE_JSON  = Path("results/eval/online.json")
RESULTS_DIR  = Path("results/figures")

BUCKETS = [
    "vp_02-04",
    "vp_04-06",
    "vp_06-08",
    "vp_08-10",
    "vp_10-12",
    "vp_12-15",
    "vp_15-15",
]

AGENTS = ["lr-unified", "lr-per_bucket", "xgb-unified", "xgb-per_bucket"]

BASELINES = [
    "RandomPlayer",
    "WeightedRandomPlayer",
    "VictoryPointPlayer",
    "AlphaBetaPlayer",
]

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_offline() -> dict:
    """Load offline.json, hard-failing with a clear message if missing."""
    if not OFFLINE_JSON.exists():
        raise FileNotFoundError(
            f"{OFFLINE_JSON} missing — run `uv run python eval.py` first"
        )
    return json.loads(OFFLINE_JSON.read_text())


def load_online() -> dict:
    """Load online.json, hard-failing with a clear message if missing."""
    if not ONLINE_JSON.exists():
        raise FileNotFoundError(
            f"{ONLINE_JSON} missing — run `uv run python eval_online.py` first"
        )
    return json.loads(ONLINE_JSON.read_text())


def get_cell(
    offline: dict, estimator: str, bucket_mode: str, bucket: str
) -> dict | None:
    """Return the first cell matching (estimator, bucket_mode, bucket), or None."""
    for cell in offline["cells"]:
        if (
            cell["estimator"] == estimator
            and cell["bucket_mode"] == bucket_mode
            and cell["bucket"] == bucket
        ):
            return cell
    return None


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------


def fig02_accuracy(offline: dict) -> Path:
    """FIG-02: grouped bar chart — accuracy by VP bucket, LR vs GBT × mode.

    Four series × seven buckets.  Asymmetric CI error bars from acc_ci.
    Per-bucket N annotated once below the x-axis (first series only).
    """
    series = [
        ("lr",  "per_bucket",     "LR per-bucket"),
        ("lr",  "unified_sliced", "LR unified-sliced"),
        ("gbt", "per_bucket",     "GBT per-bucket"),
        ("gbt", "unified_sliced", "GBT unified-sliced"),
    ]

    K = len(series)
    width = 0.8 / K
    x = np.arange(len(BUCKETS))

    fig, ax = plt.subplots(figsize=(9, 5))

    for k, (estimator, bucket_mode, label) in enumerate(series):
        vals = []
        lo_errs = []
        hi_errs = []
        ns = []

        for bucket in BUCKETS:
            cell = get_cell(offline, estimator, bucket_mode, bucket)
            if cell is None:
                vals.append(np.nan)
                lo_errs.append(0)
                hi_errs.append(0)
                ns.append(0)
            else:
                vals.append(cell["accuracy"])
                lo_errs.append(max(0, cell["accuracy"] - cell["acc_ci"][0]))
                hi_errs.append(max(0, cell["acc_ci"][1] - cell["accuracy"]))
                ns.append(cell["n_snapshots"])

        offset = (k - (K - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=[lo_errs, hi_errs],
            capsize=3,
            label=label,
        )

        # Annotate per-bucket N once (first series only) — placed just inside
        # the top of the axes area (y=0.97 axes fraction, va='top') so labels
        # sit below the title and above the topmost bars/CI whiskers.
        # Uses get_xaxis_transform(): x in data coords, y in axes fraction.
        if k == 0:
            for xi, n in zip(x, ns):
                ax.text(
                    xi,
                    0.97,
                    f"N={n:,}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    transform=ax.get_xaxis_transform(),
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [b.removeprefix("vp_") for b in BUCKETS],
        rotation=0,
        ha="center",
        fontsize=8,
    )
    ax.set_xlabel("VP stage")
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "Accuracy by VP bucket (LR vs GBT × per-bucket vs unified-sliced)"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()

    out = RESULTS_DIR / "fig02_accuracy.pdf"
    fig.savefig(
        out, format="pdf", bbox_inches="tight",
        metadata={"CreationDate": _PDF_CREATION_DATE},
    )
    plt.close(fig)
    return out


def fig07_online(online: dict) -> Path:
    """FIG-07: grouped bar chart — online win rates, agent × baseline.

    Four agents × four baselines.  Wilson 95% CI error bars.
    Per-baseline n_resolved annotated once below the x-axis (first agent).
    Dashed gray 0.5 reference line.
    """
    K = len(AGENTS)
    width = 0.8 / K
    x = np.arange(len(BASELINES))

    fig, ax = plt.subplots(figsize=(9, 5))

    for k, agent in enumerate(AGENTS):
        vals = []
        lo_errs = []
        hi_errs = []
        ns = []

        for baseline in BASELINES:
            cell = online[agent][baseline]
            vals.append(cell["win_rate"])
            lo_errs.append(max(0, cell["win_rate"] - cell["wilson_lo"]))
            hi_errs.append(max(0, cell["wilson_hi"] - cell["win_rate"]))
            ns.append(cell["n_resolved"])

        offset = (k - (K - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            yerr=[lo_errs, hi_errs],
            capsize=3,
            label=agent,
        )

        # Annotate per-baseline n_resolved once (first agent only) — placed
        # just inside the top of the axes area (y=0.97 axes fraction, va='top')
        # so labels sit below the title and above the topmost bars/CI whiskers.
        # Uses get_xaxis_transform(): x in data coords, y in axes fraction.
        if k == 0:
            for xi, n in zip(x, ns):
                ax.text(
                    xi,
                    0.97,
                    f"N_res={n}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    transform=ax.get_xaxis_transform(),
                )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [b.replace("Player", "") for b in BASELINES],
        rotation=0,
        ha="center",
        fontsize=8,
    )
    ax.set_ylabel("Win rate (Wilson 95% CI)")
    ax.set_title(
        "Online win rate per (agent × baseline), 600 games balanced seats"
    )
    # Move legend below the axes so it doesn't occlude the rightmost N label.
    # Four agents fit in one row; bottom margin enlarged to avoid clipping.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    ax.set_ylim(0, 1.05)
    fig.subplots_adjust(bottom=0.22)

    out = RESULTS_DIR / "fig07_online.pdf"
    fig.savefig(
        out, format="pdf", bbox_inches="tight",
        metadata={"CreationDate": _PDF_CREATION_DATE},
    )
    plt.close(fig)
    return out


def fig_roc(offline: dict) -> Path:
    """FIG-11 (ROC half): unified LR vs GBT ROC overlay with AUC in legend."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for est, label in [("lr", "LR-unified"), ("gbt", "GBT-unified")]:
        cell = get_cell(offline, est, "unified", "all")
        fpr = cell["roc"]["fpr"]
        tpr = cell["roc"]["tpr"]
        auc = cell["auc"]
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: unified LR vs GBT (test split)")
    ax.legend(loc="lower right", fontsize=9)

    out = RESULTS_DIR / "fig_roc.pdf"
    fig.savefig(
        out, format="pdf", bbox_inches="tight",
        metadata={"CreationDate": _PDF_CREATION_DATE},
    )
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    offline = load_offline()
    online  = load_online()
    paths = [
        fig02_accuracy(offline),
        fig07_online(online),
        fig_roc(offline),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
