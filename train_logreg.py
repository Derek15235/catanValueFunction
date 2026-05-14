"""train_logreg.py — L2 logistic regression, per-VP-bucket and unified.

Loads data/snapshots.parquet + data/splits.json, trains models, and prints
accuracy / log-loss / ECE (with bootstrap CIs) for each VP bucket and overall.

Usage:
    uv run python train_logreg.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/snapshots.parquet")
SPLITS_PATH = Path("data/splits.json")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results/lr")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
# VP bucket boundaries: each bucket is [lo, hi) except last is [lo, ∞)
VP_BUCKETS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 99)]


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(metric_fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    ll = log_loss(y_true, y_prob)
    ec = ece(y_true, y_prob)

    acc_lo, acc_hi = bootstrap_ci(
        y_true, y_prob, lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int))
    )
    ll_lo, ll_hi = bootstrap_ci(y_true, y_prob, log_loss)
    ece_lo, ece_hi = bootstrap_ci(y_true, y_prob, ece)

    return {
        "name": name,
        "n": len(y_true),
        "accuracy": acc,
        "acc_ci": (acc_lo, acc_hi),
        "log_loss": ll,
        "ll_ci": (ll_lo, ll_hi),
        "ece": ec,
        "ece_ci": (ece_lo, ece_hi),
    }


def print_top_coefficients(clf, feature_names: list[str], n: int = 10) -> None:
    coef = clf.coef_[0]
    idx = np.argsort(np.abs(coef))[::-1][:n]
    for i in idx:
        print(f"      {feature_names[i]:<40}  {coef[i]:+.4f}")


def print_result(r: dict) -> None:
    print(
        f"  {r['name']:<22}  n={r['n']:>8,}  "
        f"acc={r['accuracy']:.4f} [{r['acc_ci'][0]:.4f},{r['acc_ci'][1]:.4f}]  "
        f"loss={r['log_loss']:.4f} [{r['ll_ci'][0]:.4f},{r['ll_ci'][1]:.4f}]  "
        f"ece={r['ece']:.4f} [{r['ece_ci'][0]:.4f},{r['ece_ci'][1]:.4f}]"
    )


def bucket_label(lo: int, hi: int) -> str:
    return f"vp_{lo:02d}-{min(hi,15):02d}"


def main() -> None:
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    splits = json.loads(SPLITS_PATH.read_text())

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    from schema import FEATURE_ORDERING
    feature_cols = FEATURE_ORDERING

    train_df = df[df["game_id"].isin(train_ids)]
    val_df = df[df["game_id"].isin(val_ids)]
    test_df = df[df["game_id"].isin(test_ids)]

    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values

    def _make_pipeline(C: float, max_iter: int = 2000) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)),
        ])

    def _coef(pipe: Pipeline) -> np.ndarray:
        return pipe.named_steps["clf"].coef_[0]

    def _prob(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
        return pipe.predict_proba(X)[:, 1]

    def _serializable(r: dict) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}

    def _save_coef_csv(pipe: Pipeline, name: str) -> None:
        coef = _coef(pipe)
        rows = sorted(
            [{"feature": f, "coefficient": float(coef[i]), "abs_coefficient": float(abs(coef[i]))}
             for i, f in enumerate(feature_cols)],
            key=lambda x: x["abs_coefficient"], reverse=True,
        )
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"coef_{name}.csv", index=False)

    # --- Tune C on val set ---
    print("\nTuning C (regularization strength)...")
    best_C, best_acc, pipe_unified = None, -1.0, None
    for C in [0.1, 1.0, 10.0, 100.0]:
        pipe = _make_pipeline(C, max_iter=1000)
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_val, pipe.predict(X_val))
        print(f"  C={C:<6}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_C, pipe_unified = acc, C, pipe

    print(f"  → best C={best_C}")

    # --- Unified model ---
    print("\n=== Unified model ===")
    print("  top features:")
    print_top_coefficients(pipe_unified.named_steps["clf"], feature_cols)
    unified_results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = _prob(pipe_unified, X)
        r = evaluate(f"unified/{split_name}", y, prob)
        print_result(r)
        unified_results[split_name] = _serializable(r)

    joblib.dump(pipe_unified, RESULTS_DIR / "pipeline_unified.joblib")
    _save_coef_csv(pipe_unified, "unified")

    # --- Per-bucket models ---
    print("\n=== Per-bucket models (test set) ===")
    bucket_results = []
    for lo, hi in VP_BUCKETS:
        label = bucket_label(lo, hi)

        tr = train_df[(train_df["max_vp"] >= lo) & (train_df["max_vp"] < hi)]
        te = test_df[(test_df["max_vp"] >= lo) & (test_df["max_vp"] < hi)]

        if len(tr) < 500 or len(te) < 100:
            print(f"  {label}: skipped (train={len(tr)}, test={len(te)})")
            continue

        X_tr = tr[feature_cols].values.astype(np.float32)
        y_tr = tr["label"].values
        X_te = te[feature_cols].values.astype(np.float32)
        y_te = te["label"].values

        pipe = _make_pipeline(best_C)
        pipe.fit(X_tr, y_tr)

        prob = _prob(pipe, X_te)
        r = evaluate(label, y_te, prob)
        print_result(r)
        print(f"    top features:")
        print_top_coefficients(pipe.named_steps["clf"], feature_cols)
        bucket_results.append(_serializable(r))

        joblib.dump(pipe, RESULTS_DIR / f"pipeline_{label}.joblib")
        _save_coef_csv(pipe, label)

    # --- Unified model sliced by bucket (for comparison) ---
    print("\n=== Unified model sliced by VP bucket (test set) ===")
    unified_slice_results = []
    for lo, hi in VP_BUCKETS:
        label = bucket_label(lo, hi)
        mask = (test_df["max_vp"] >= lo) & (test_df["max_vp"] < hi)
        if mask.sum() < 100:
            continue
        X_slice = test_df.loc[mask, feature_cols].values.astype(np.float32)
        y = test_df.loc[mask, "label"].values
        prob = _prob(pipe_unified, X_slice)
        r = evaluate(f"unified/{label}", y, prob)
        print_result(r)
        unified_slice_results.append(_serializable(r))

    # --- Save metrics.json ---
    output = {
        "best_C": best_C,
        "unified": unified_results,
        "per_bucket": bucket_results,
        "unified_sliced": unified_slice_results,
    }
    out_path = RESULTS_DIR / "metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/: metrics.json, pipeline_*.joblib, coef_*.csv")


if __name__ == "__main__":
    main()
