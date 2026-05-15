"""train_gbt.py — XGBoost gradient-boosted trees, per-VP-bucket and unified.

Mirrors train_logreg.py structure exactly so results are directly comparable.
Tunes max_depth on the val set, uses early stopping to find n_estimators.
Saves metrics.json, pipeline_*.joblib, and feature_importance_*.csv.

Usage:
    uv run python train_gbt.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/snapshots.parquet")
SPLITS_PATH = Path("data/splits.json")
RESULTS_DIR = Path("results/gbt")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
VP_BUCKETS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 99)]

# Fixed XGBoost hyperparameters — only max_depth is tuned
FIXED_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss",
    random_state=0,
    n_jobs=-1,
)
DEPTH_GRID = [4, 6, 8]
EARLY_STOPPING_ROUNDS = 20


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece_val += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece_val)


def bootstrap_ci(y_true, y_prob, metric_fn, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    vals = [metric_fn(y_true[idx := rng.integers(0, len(y_true), len(y_true))],
                      y_prob[idx]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(int))
    ll = log_loss(y_true, y_prob)
    ec = ece(y_true, y_prob)
    acc_ci = bootstrap_ci(y_true, y_prob,
                          lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)))
    ll_ci = bootstrap_ci(y_true, y_prob, log_loss)
    ece_ci = bootstrap_ci(y_true, y_prob, ece)
    return dict(name=name, n=len(y_true),
                accuracy=acc, acc_ci=acc_ci,
                log_loss=ll, ll_ci=ll_ci,
                ece=ec, ece_ci=ece_ci)


def print_result(r: dict) -> None:
    print(
        f"  {r['name']:<22}  n={r['n']:>8,}  "
        f"acc={r['accuracy']:.4f} [{r['acc_ci'][0]:.4f},{r['acc_ci'][1]:.4f}]  "
        f"loss={r['log_loss']:.4f} [{r['ll_ci'][0]:.4f},{r['ll_ci'][1]:.4f}]  "
        f"ece={r['ece']:.4f} [{r['ece_ci'][0]:.4f},{r['ece_ci'][1]:.4f}]"
    )


def bucket_label(lo: int, hi: int) -> str:
    return f"vp_{lo:02d}-{min(hi, 15):02d}"


def _serializable(r: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}


def train_xgb(X_tr, y_tr, X_val, y_val, max_depth: int) -> XGBClassifier:
    clf = XGBClassifier(max_depth=max_depth, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        **FIXED_PARAMS)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def make_pipeline(clf) -> Pipeline:
    # XGBoost is scale-invariant; passthrough keeps the Pipeline API consistent
    # with train_logreg.py so the lookahead agent can use both interchangeably.
    return Pipeline([("passthrough", FunctionTransformer()), ("clf", clf)])


def save_importance_csv(clf: XGBClassifier, feature_cols: list, name: str) -> None:
    importance = clf.feature_importances_
    rows = sorted(
        [{"feature": f, "gain_importance": float(importance[i])}
         for i, f in enumerate(feature_cols)],
        key=lambda x: x["gain_importance"], reverse=True,
    )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"importance_{name}.csv", index=False)


def print_top_importance(clf: XGBClassifier, feature_cols: list, n: int = 10) -> None:
    idx = np.argsort(clf.feature_importances_)[::-1][:n]
    for i in idx:
        print(f"      {feature_cols[i]:<40}  {clf.feature_importances_[i]:.4f}")


def main() -> None:
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    splits = json.loads(SPLITS_PATH.read_text())

    from schema import FEATURE_ORDERING
    feature_cols = FEATURE_ORDERING

    train_df = df[df["game_id"].isin(set(splits["train"]))]
    val_df   = df[df["game_id"].isin(set(splits["val"]))]
    test_df  = df[df["game_id"].isin(set(splits["test"]))]
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["label"].values
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["label"].values

    # --- Tune max_depth on val set ---
    print("\nTuning max_depth...")
    best_depth, best_acc, best_clf = None, -1.0, None
    for depth in DEPTH_GRID:
        clf = train_xgb(X_train, y_train, X_val, y_val, depth)
        acc = accuracy_score(y_val, clf.predict(X_val))
        print(f"  max_depth={depth}  val_acc={acc:.4f}  "
              f"best_iteration={clf.best_iteration}")
        if acc > best_acc:
            best_acc, best_depth, best_clf = acc, depth, clf

    print(f"  → best max_depth={best_depth}")
    clf_unified = best_clf

    # --- Unified model ---
    print("\n=== Unified model ===")
    print("  top features (gain):")
    print_top_importance(clf_unified, feature_cols)
    unified_results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = clf_unified.predict_proba(X)[:, 1]
        r = evaluate(f"unified/{split_name}", y, prob)
        print_result(r)
        unified_results[split_name] = _serializable(r)

    pipe_unified = make_pipeline(clf_unified)
    joblib.dump(pipe_unified, RESULTS_DIR / "pipeline_unified.joblib")
    save_importance_csv(clf_unified, feature_cols, "unified")

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

        clf = train_xgb(X_tr, y_tr, X_va, y_va, best_depth)

        prob = clf.predict_proba(X_te)[:, 1]
        r = evaluate(label, y_te, prob)
        print_result(r)
        print(f"    top features (gain):")
        print_top_importance(clf, feature_cols)
        bucket_results.append(_serializable(r))

        joblib.dump(make_pipeline(clf), RESULTS_DIR / f"pipeline_{label}.joblib")
        save_importance_csv(clf, feature_cols, label)

    # --- Unified model sliced by bucket ---
    print("\n=== Unified model sliced by VP bucket (test set) ===")
    unified_slice_results = []
    for lo, hi in VP_BUCKETS:
        label = bucket_label(lo, hi)
        mask = (test_df["max_vp"] >= lo) & (test_df["max_vp"] < hi)
        if mask.sum() < 100:
            continue
        X_slice = test_df.loc[mask, feature_cols].values.astype(np.float32)
        y = test_df.loc[mask, "label"].values
        prob = clf_unified.predict_proba(X_slice)[:, 1]
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
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/: metrics.json, pipeline_*.joblib, importance_*.csv")


if __name__ == "__main__":
    main()
