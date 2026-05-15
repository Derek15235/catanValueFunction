"""train_gbt.py — XGBoost gradient-boosted trees, per-VP-bucket and unified.

Mirrors train_logreg.py: tunes one hyperparameter on the val set, reports
accuracy / log-loss / ECE (with bootstrap CIs) for each VP bucket and overall.
Uses early stopping on the val set to pick n_estimators.

Usage:
    uv run python train_gbt.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/snapshots.parquet")
SPLITS_PATH = Path("data/splits.json")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results/gbt")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
# VP bucket boundaries: each bucket is [lo, hi) except last is [lo, ∞)
VP_BUCKETS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 99)]

DEPTH_GRID = [4, 6, 8]
EARLY_STOPPING_ROUNDS = 20


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


def print_top_importance(clf, feature_names: list[str], n: int = 10) -> None:
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:n]
    for i in idx:
        print(f"      {feature_names[i]:<40}  {imp[i]:.4f}")


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

    def _make_pipeline(clf: XGBClassifier) -> Pipeline:
        # XGBoost is scale-invariant; passthrough keeps Pipeline API consistent with LR.
        return Pipeline([("passthrough", FunctionTransformer()), ("clf", clf)])

    def _train_xgb(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray, max_depth: int) -> XGBClassifier:
        clf = XGBClassifier(
            max_depth=max_depth,
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=0,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return clf

    def _prob(clf: XGBClassifier, X: np.ndarray) -> np.ndarray:
        return clf.predict_proba(X)[:, 1]

    def _serializable(r: dict) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}

    def _save_importance_csv(clf: XGBClassifier, name: str) -> None:
        imp = clf.feature_importances_
        rows = sorted(
            [{"feature": f, "gain_importance": float(imp[i])}
             for i, f in enumerate(feature_cols)],
            key=lambda x: x["gain_importance"], reverse=True,
        )
        pd.DataFrame(rows).to_csv(RESULTS_DIR / f"importance_{name}.csv", index=False)

    # --- Tune max_depth on val set ---
    print("\nTuning max_depth...")
    best_depth, best_acc, clf_unified = None, -1.0, None
    for depth in DEPTH_GRID:
        clf = _train_xgb(X_train, y_train, X_val, y_val, depth)
        acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"  max_depth={depth:<3}  val_acc={acc:.4f}  best_iter={clf.best_iteration}")
        if acc > best_acc:
            best_acc, best_depth, clf_unified = acc, depth, clf

    print(f"  → best max_depth={best_depth}")

    # --- Unified model ---
    print("\n=== Unified model ===")
    print("  top features (gain):")
    print_top_importance(clf_unified, feature_cols)
    unified_results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = _prob(clf_unified, X)
        r = evaluate(f"unified/{split_name}", y, prob)
        print_result(r)
        unified_results[split_name] = _serializable(r)

    joblib.dump(_make_pipeline(clf_unified), RESULTS_DIR / "pipeline_unified.joblib")
    _save_importance_csv(clf_unified, "unified")

    # --- Per-bucket models ---
    print("\n=== Per-bucket models (test set) ===")
    bucket_results = []
    for lo, hi in VP_BUCKETS:
        label = bucket_label(lo, hi)

        tr = train_df[(train_df["max_vp"] >= lo) & (train_df["max_vp"] < hi)]
        va = val_df[(val_df["max_vp"] >= lo) & (val_df["max_vp"] < hi)]
        te = test_df[(test_df["max_vp"] >= lo) & (test_df["max_vp"] < hi)]

        if len(tr) < 500 or len(te) < 100:
            print(f"  {label}: skipped (train={len(tr)}, test={len(te)})")
            continue

        X_tr = tr[feature_cols].values.astype(np.float32)
        y_tr = tr["label"].values
        X_va = va[feature_cols].values.astype(np.float32)
        y_va = va["label"].values
        X_te = te[feature_cols].values.astype(np.float32)
        y_te = te["label"].values

        clf = _train_xgb(X_tr, y_tr, X_va, y_va, best_depth)

        prob = _prob(clf, X_te)
        r = evaluate(label, y_te, prob)
        print_result(r)
        print(f"    top features (gain):")
        print_top_importance(clf, feature_cols)
        bucket_results.append(_serializable(r))

        joblib.dump(_make_pipeline(clf), RESULTS_DIR / f"pipeline_{label}.joblib")
        _save_importance_csv(clf, label)

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
        prob = _prob(clf_unified, X_slice)
        r = evaluate(f"unified/{label}", y, prob)
        print_result(r)
        unified_slice_results.append(_serializable(r))

    # --- Save metrics.json ---
    output = {
        "best_depth": best_depth,
        "unified": unified_results,
        "per_bucket": bucket_results,
        "unified_sliced": unified_slice_results,
    }
    out_path = RESULTS_DIR / "metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/: metrics.json, pipeline_*.joblib, importance_*.csv")


if __name__ == "__main__":
    main()