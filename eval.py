"""eval.py — Phase 3 offline evaluation with cluster-bootstrap CIs.

Loads persisted LR + GBT pipelines from results/{lr,gbt}/, re-predicts on
the test split only, and writes results/eval/offline.json + 2 reliability PNGs.

Pre-registered constants:
  THRESHOLD = 0.02          # effect-size threshold for LR-vs-XGB verdict
  N_BOOTSTRAP = 1000        # cluster bootstrap iterations
  BOOTSTRAP_SEED = 42       # rng seed (byte-stable CIs across re-runs)
  VP_BUCKETS                # locked at training time, mirrored here
  RELIABILITY_BINS = 10     # equal-width bins for ECE + reliability plot

Usage:
    uv run python eval.py
"""
from __future__ import annotations

import json
import math
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe; must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ---------- Pre-registered constants (anti-p-hacking guard) ----------
THRESHOLD: float = 0.02
N_BOOTSTRAP: int = 1000
BOOTSTRAP_SEED: int = 42
VP_BUCKETS: list[tuple[int, int]] = [
    (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 99),
]
RELIABILITY_BINS: int = 10
MIN_BUCKET_N: int = 50  # eval-time min-N per (test-split, bucket) cell

DATA_PATH: Path = Path("data/snapshots.parquet")
SPLITS_PATH: Path = Path("data/splits.json")
LR_DIR: Path = Path("results/lr")
GBT_DIR: Path = Path("results/gbt")
EVAL_DIR: Path = Path("results/eval")


# ---------- Metrics ----------
def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = RELIABILITY_BINS) -> float:
    """Expected Calibration Error — n_bins equal-width bins on [0, 1].
    Lifted verbatim from train_logreg.py:37.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += mask.mean() * abs(acc - conf)
    return float(ece_val)


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bins: int = RELIABILITY_BINS):
    """Per-bin (mean_prob, mean_label) for reliability diagrams.
    Same binning as ece(). Skip empty bins.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        xs.append(float(y_prob[mask].mean()))
        ys.append(float(y_true[mask].mean()))
    return np.asarray(xs), np.asarray(ys)


# ---------- Cluster bootstrap (the one new mathematical operation) ----------
def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    game_ids: np.ndarray,
    metric_fn,
    rng: np.random.Generator,
    n: int = N_BOOTSTRAP,
) -> tuple[float, float]:
    """Percentile 95% CI by resampling distinct game_ids with replacement.

    Steps per iter:
      1. Draw G game_ids with replacement (G = number of distinct game_ids).
      2. Concatenate row indices of all sampled game_ids.
      3. Compute metric on the concatenated subset.
    """
    codes, uniques = pd.factorize(game_ids)
    G = len(uniques)
    # Precompute row indices per cluster code once
    groups = [np.flatnonzero(codes == k) for k in range(G)]

    vals = np.empty(n, dtype=np.float64)
    for b in range(n):
        sampled = rng.integers(0, G, G)
        idx = np.concatenate([groups[k] for k in sampled])
        vals[b] = metric_fn(y_true[idx], y_prob[idx])

    # nanpercentile so AUC/AUPRC metric_fns that return NaN on single-class
    # bootstrap draws still produce a CI from the well-defined draws.
    # No-op for accuracy/log_loss/ece/brier (label sets are pinned, no NaNs).
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------- Per-cell evaluate ----------
def evaluate(
    *,
    name: str,
    estimator: str,
    bucket_mode: str,
    bucket: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    game_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Compute 4 metrics + cluster-bootstrap CIs for one cell.
    Clips probabilities + casts to float64 once at the top.
    """
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6).astype(np.float64)
    y_true = np.asarray(y_true).astype(np.int64)

    acc_fn = lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int))
    # Pin label set so single-class slices (or bootstrap draws that
    # accidentally sample only one class) don't crash log_loss / brier.
    ll_fn = lambda yt, yp: log_loss(yt, yp, labels=[0, 1])
    ece_fn = ece
    brier_fn = lambda yt, yp: brier_score_loss(yt, yp, pos_label=1)
    # AUC / AUPRC are undefined for single-class subsamples (roc_auc_score
    # raises ValueError); fall back to NaN so the bootstrap percentile call
    # still produces a CI from the well-defined draws.
    def auc_fn(yt, yp):
        if len(np.unique(yt)) < 2:
            return float("nan")
        return roc_auc_score(yt, yp)
    def auprc_fn(yt, yp):
        if len(np.unique(yt)) < 2:
            return float("nan")
        return average_precision_score(yt, yp)

    acc = acc_fn(y_true, y_prob)
    ll = ll_fn(y_true, y_prob)
    ec = ece_fn(y_true, y_prob)
    bri = brier_fn(y_true, y_prob)
    auc = auc_fn(y_true, y_prob)
    auprc = auprc_fn(y_true, y_prob)

    acc_ci = cluster_bootstrap_ci(y_true, y_prob, game_ids, acc_fn, rng)
    ll_ci = cluster_bootstrap_ci(y_true, y_prob, game_ids, ll_fn, rng)
    ece_ci = cluster_bootstrap_ci(y_true, y_prob, game_ids, ece_fn, rng)
    brier_ci = cluster_bootstrap_ci(y_true, y_prob, game_ids, brier_fn, rng)
    auc_ci = cluster_bootstrap_ci(y_true, y_prob, game_ids, auc_fn, rng)
    auprc_ci = cluster_bootstrap_ci(y_true, y_prob, game_ids, auprc_fn, rng)

    return {
        "name": name,
        "estimator": estimator,
        "bucket_mode": bucket_mode,
        "bucket": bucket,
        "n_snapshots": int(len(y_true)),
        "n_games": int(pd.Series(game_ids).nunique()),
        "accuracy": float(acc),
        "acc_ci": [float(acc_ci[0]), float(acc_ci[1])],
        "log_loss": float(ll),
        "ll_ci": [float(ll_ci[0]), float(ll_ci[1])],
        "ece": float(ec),
        "ece_ci": [float(ece_ci[0]), float(ece_ci[1])],
        "brier": float(bri),
        "brier_ci": [float(brier_ci[0]), float(brier_ci[1])],
        "auc": float(auc),
        "auc_ci": [float(auc_ci[0]), float(auc_ci[1])],
        "auprc": float(auprc),
        "auprc_ci": [float(auprc_ci[0]), float(auprc_ci[1])],
    }


# ---------- Comparison object construction (LR vs GBT, accuracy only) ----------
def build_comparisons(cells: list[dict]) -> list[dict]:
    """One verdict per (bucket_mode, bucket) cell that has both LR and GBT rows.
    Two-hurdle test: abs(delta) > THRESHOLD AND CIs disjoint.
    """
    idx: dict[tuple[str, str], dict[str, dict]] = {}
    for c in cells:
        key = (c["bucket_mode"], c["bucket"])
        idx.setdefault(key, {})[c["estimator"]] = c

    comparisons = []
    for (mode, bucket), by_est in sorted(idx.items()):
        if "lr" not in by_est or "gbt" not in by_est:
            continue
        lr_c, xgb_c = by_est["lr"], by_est["gbt"]
        delta = xgb_c["accuracy"] - lr_c["accuracy"]
        ci_lr, ci_xgb = lr_c["acc_ci"], xgb_c["acc_ci"]
        ci_disjoint = (ci_lr[1] < ci_xgb[0]) or (ci_xgb[1] < ci_lr[0])
        meets = (abs(delta) > THRESHOLD) and ci_disjoint
        if not meets:
            verdict = "tie"
        else:
            verdict = "xgb_wins" if delta > 0 else "lr_wins"
        comparisons.append({
            "bucket_mode": mode,
            "bucket": bucket,
            "lr_value": float(lr_c["accuracy"]),
            "xgb_value": float(xgb_c["accuracy"]),
            "delta": float(delta),
            "ci_lr": list(ci_lr),
            "ci_xgb": list(ci_xgb),
            "ci_disjoint": bool(ci_disjoint),
            "meets_threshold": bool(meets),
            "verdict": verdict,
        })
    return comparisons


# ---------- Misc helpers ----------
def bucket_label(lo: int, hi: int) -> str:
    return f"vp_{lo:02d}-{min(hi, 15):02d}"


def print_result(r: dict) -> None:
    print(
        f"  {r['name']:<32}  n={r['n_snapshots']:>8,}  games={r['n_games']:>6,}  "
        f"acc={r['accuracy']:.4f} [{r['acc_ci'][0]:.4f},{r['acc_ci'][1]:.4f}]  "
        f"loss={r['log_loss']:.4f}  ece={r['ece']:.4f}  brier={r['brier']:.4f}"
    )


def _downsample(a: np.ndarray, n: int = 100) -> list[float]:
    """Uniformly downsample a 1-D array to <=n points; return as a Python list.
    Used to keep ROC / PR curve arrays in offline.json small (<= 100 pts).
    """
    if len(a) <= n:
        return [float(x) for x in a]
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return [float(x) for x in a[idx]]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------- Per-estimator driver ----------
def evaluate_estimator(
    name: str,
    results_dir: Path,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    rng: np.random.Generator,
    cell_arrays: dict,
) -> list[dict]:
    """Load `name`'s pipelines from `results_dir`, predict on test split.

    Always emits a unified-on-full-test cell. Then for each VP bucket where
    the slice has >= 100 rows: always emits a `unified_sliced` row; emits a
    `per_bucket` row only when `pipeline_{label}.joblib` exists on disk
    (matches the train-time min-N gate).

    Mutates `cell_arrays`: keyed by (estimator, bucket_mode, bucket) ->
    (y_true, y_prob_float64_clipped). Used by `plot_reliability`. The
    unified-on-full-test cell is NOT recorded in cell_arrays (the plotter
    only renders per_bucket and unified_sliced grids).
    """
    cells: list[dict] = []
    pipe_unified = joblib.load(results_dir / "pipeline_unified.joblib")

    # Unified-on-full-test cell
    X_full = test_df[feature_cols].values.astype(np.float32)
    y_full = test_df["label"].values
    gids_full = test_df["game_id"].values
    prob = pipe_unified.predict_proba(X_full)[:, 1]
    cells.append(evaluate(
        name=f"{name}/unified",
        estimator=name, bucket_mode="unified", bucket="all",
        y_true=y_full, y_prob=prob, game_ids=gids_full, rng=rng,
    ))
    # Stash unified-on-all arrays for FIG-11 ROC/PR curves (main() attaches
    # downsampled roc/pr dicts to only these two cells — not the sliced ones).
    cell_arrays[(name, "unified", "all")] = (
        y_full.copy(), np.clip(prob, 1e-6, 1 - 1e-6).astype(np.float64),
    )

    for lo, hi in VP_BUCKETS:
        label = bucket_label(lo, hi)
        mask = (test_df["max_vp"] >= lo) & (test_df["max_vp"] < hi)
        if mask.sum() < MIN_BUCKET_N:
            print(f"  {name}/{label}: skipped (n={int(mask.sum())} < {MIN_BUCKET_N})")
            continue
        slice_df = test_df.loc[mask].reset_index(drop=True)
        X = slice_df[feature_cols].values.astype(np.float32)
        y = slice_df["label"].values
        gids = slice_df["game_id"].values

        # unified_sliced row (always available)
        prob_us = pipe_unified.predict_proba(X)[:, 1]
        cells.append(evaluate(
            name=f"{name}/unified_sliced/{label}",
            estimator=name, bucket_mode="unified_sliced", bucket=label,
            y_true=y, y_prob=prob_us, game_ids=gids, rng=rng,
        ))
        cell_arrays[(name, "unified_sliced", label)] = (
            y.copy(), np.clip(prob_us, 1e-6, 1 - 1e-6).astype(np.float64),
        )

        # per_bucket row (only if pipeline exists on disk)
        bucket_path = results_dir / f"pipeline_{label}.joblib"
        if not bucket_path.exists():
            print(f"  {name}/per_bucket/{label}: pipeline missing on disk, skipped")
            continue
        pipe_bucket = joblib.load(bucket_path)
        prob_pb = pipe_bucket.predict_proba(X)[:, 1]
        cells.append(evaluate(
            name=f"{name}/per_bucket/{label}",
            estimator=name, bucket_mode="per_bucket", bucket=label,
            y_true=y, y_prob=prob_pb, game_ids=gids, rng=rng,
        ))
        cell_arrays[(name, "per_bucket", label)] = (
            y.copy(), np.clip(prob_pb, 1e-6, 1 - 1e-6).astype(np.float64),
        )

    return cells


# ---------- Reliability diagram grid ----------
def plot_reliability(
    cells: list[dict],
    cell_arrays: dict,
    bucket_mode: str,
    out_path: Path,
) -> None:
    """Render an N-panel reliability-diagram grid (one panel per bucket).

    For each bucket panel, overlay LR (blue) and GBT (orange) reliability
    curves with y=x reference and per-estimator ECE annotation. Skip
    missing series cleanly.
    """
    rows = [c for c in cells if c["bucket_mode"] == bucket_mode]
    buckets = sorted({c["bucket"] for c in rows})
    if not buckets:
        print(f"  plot_reliability({bucket_mode}): no cells, skipping")
        return

    ncols = 4
    nrows = math.ceil(len(buckets) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False,
    )

    for ax, bucket in zip(axes.flat, buckets):
        for estimator, color, y_offset in [
            ("lr", "tab:blue", 0.0),
            ("gbt", "tab:orange", 0.08),
        ]:
            arr = cell_arrays.get((estimator, bucket_mode, bucket))
            if arr is None:
                continue
            y_true, y_prob = arr
            xs, ys = reliability_bins(y_true, y_prob)
            if len(xs) > 0:
                ax.plot(xs, ys, marker="o", label=estimator.upper(), color=color)
            ec = next(
                (c["ece"] for c in rows
                 if c["estimator"] == estimator and c["bucket"] == bucket),
                None,
            )
            if ec is not None:
                ax.text(
                    0.05, 0.95 - y_offset,
                    f"{estimator.upper()} ECE={ec:.3f}",
                    transform=ax.transAxes, fontsize=9, color=color, va="top",
                )
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(bucket)
        ax.set_xlabel("Predicted prob")
        ax.set_ylabel("Empirical win rate")
        ax.legend(loc="lower right", fontsize=8)

    for ax in axes.flat[len(buckets):]:
        ax.axis("off")

    fig.suptitle(f"Reliability — {bucket_mode}", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------- main() — Wave 2 production runner ----------
def main() -> None:
    from schema import FEATURE_ORDERING  # local import, matches train_*.py

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    splits = json.loads(SPLITS_PATH.read_text())
    test_ids = set(splits["test"])
    test_df = df[df["game_id"].isin(test_ids)].reset_index(drop=True)
    print(
        f"  test snapshots = {len(test_df):,}  "
        f"test games = {test_df.game_id.nunique():,}"
    )

    # Single rng threaded through every evaluate() / cluster_bootstrap_ci call
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cell_arrays: dict = {}
    cells: list[dict] = []

    print("\nLR cells:")
    cells += evaluate_estimator(
        "lr", LR_DIR, test_df, list(FEATURE_ORDERING), rng, cell_arrays,
    )
    print("\nGBT cells:")
    cells += evaluate_estimator(
        "gbt", GBT_DIR, test_df, list(FEATURE_ORDERING), rng, cell_arrays,
    )

    # Attach downsampled ROC + PR arrays to ONLY the two unified-on-all cells
    # (lr/unified, gbt/unified). FIG-11 (Phase 5) plots these two curves;
    # per-bucket / unified_sliced cells don't carry curves to keep file small.
    for cell in cells:
        if cell["bucket_mode"] != "unified" or cell["bucket"] != "all":
            continue
        arr = cell_arrays.get((cell["estimator"], "unified", "all"))
        if arr is None:
            continue
        y_true_u, y_prob_u = arr
        fpr, tpr, _ = roc_curve(y_true_u, y_prob_u)
        prec, rec, _ = precision_recall_curve(y_true_u, y_prob_u)
        cell["roc"] = {"fpr": _downsample(fpr), "tpr": _downsample(tpr)}
        cell["pr"] = {"precision": _downsample(prec), "recall": _downsample(rec)}

    print("\nPer-cell results:")
    for cell in cells:
        print_result(cell)

    comparisons = build_comparisons(cells)
    print("\nLR-vs-GBT verdicts:")
    for cmp in comparisons:
        print(
            f"  {cmp['bucket_mode']:<15} {cmp['bucket']:<10}  "
            f"verdict={cmp['verdict']:<10}  "
            f"delta={cmp['delta']:+.4f}  "
            f"ci_lr=[{cmp['ci_lr'][0]:.4f},{cmp['ci_lr'][1]:.4f}]  "
            f"ci_xgb=[{cmp['ci_xgb'][0]:.4f},{cmp['ci_xgb'][1]:.4f}]"
        )

    out = {
        "threshold": THRESHOLD,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "git_commit": _git_commit(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_n_rows": int(len(df)),
        "test_n_rows": int(len(test_df)),
        "test_n_games": int(test_df["game_id"].nunique()),
        "cells": cells,
        "comparisons": comparisons,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "offline.json").write_text(json.dumps(out, indent=2))

    plot_reliability(
        cells, cell_arrays, "per_bucket",
        EVAL_DIR / "figures" / "reliability_per_bucket.png",
    )
    plot_reliability(
        cells, cell_arrays, "unified_sliced",
        EVAL_DIR / "figures" / "reliability_unified_sliced.png",
    )

    print(f"\nSaved {EVAL_DIR}/offline.json + 2 reliability PNGs.")


if __name__ == "__main__":
    main()
