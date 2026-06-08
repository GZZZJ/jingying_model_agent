#!/usr/bin/env python3
"""Diagnose overfitting vs feature leakage for the 73-feature GCard model.

Compares train vs valid AUC and flags suspicious feature names.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from jingying_agent.config import load_yaml

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 0,
}

SUSPICIOUS_PATTERNS = {
    "future": "Contains 'future' - likely using future-period data",
    "suc_": "Contains 'suc_' - may reference future success outcomes",
    "_suc_": "Contains '_suc_' - may reference future success outcomes",
    "after_": "Contains 'after_' - may reference post-event data",
    "next_": "Contains 'next_' - may reference future period",
}

# ============================================================================
# Copy essential helpers from 08_refine_wide_features.py (avoids import issues)
# ============================================================================

def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def sql_identifier(name: str) -> str:
    return name if name.replace("_", "").isalnum() and not name[0].isdigit() else f"`{name}`"


def build_sampling_sql(cfg: dict, features: list[str], max_rows_override: int | None = None) -> str:
    """Build SQL to pull selected features + base columns from wide table."""
    input_cfg = cfg["feature_refine"]["input"]
    sampling = cfg["feature_refine"]["sampling"]
    base_columns = list(dict.fromkeys(input_cfg["base_columns"]))
    select_columns = base_columns + [f for f in features if f not in base_columns]
    select_expr = ",\n  ".join(sql_identifier(c) for c in select_columns)
    sql = f"select\n  {select_expr}\nfrom {input_cfg['wide_table']}"
    if sampling.get("where"):
        sql += f"\nwhere {sampling['where']}"
    max_rows = max_rows_override or sampling.get("max_rows")
    if max_rows:
        sql += f"\nlimit {int(max_rows)}"
    return sql + "\n"


def load_wide_sample(sql: str) -> pd.DataFrame:
    from tmlpatch.database import TMLSQLClient
    client = TMLSQLClient()
    try:
        return client.sql(sql).to_pandas()
    finally:
        client.stop()


def coerce_feature_frame(df: pd.DataFrame, features: list[str], cfg: dict) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Coerce feature columns to numeric, drop low-quality features."""
    preprocessing = cfg["feature_refine"]["preprocessing"]
    sentinels = preprocessing.get("missing_sentinels", [])
    min_non_null_rate = float(preprocessing.get("min_non_null_rate", 0.0))
    drop_constant = bool(preprocessing.get("drop_constant", True))

    available = [f for f in features if f in df.columns]
    print(f"[COERCE] {len(available)}/{len(features)} features found in data")

    x = df.loc[:, available].copy()
    stats, kept = [], []
    for feature in available:
        series = pd.to_numeric(x[feature], errors="coerce")
        if sentinels:
            series = series.replace(sentinels, np.nan)
        series = series.replace([np.inf, -np.inf], np.nan)
        non_null_rate = float(series.notna().mean())
        unique_count = int(series.nunique(dropna=True))
        drop_reason = ""
        if non_null_rate < min_non_null_rate:
            drop_reason = "low_non_null_rate"
        elif drop_constant and unique_count <= 1:
            drop_reason = "constant"
        else:
            kept.append(feature)
            x[feature] = series
        stats.append({"feature": feature, "non_null_rate": non_null_rate, "unique_count": unique_count, "drop_reason": drop_reason})

    drop_counts = {}
    for s in stats:
        if s["drop_reason"]:
            drop_counts[s["drop_reason"]] = drop_counts.get(s["drop_reason"], 0) + 1
    print(f"[COERCE] kept={len(kept)}/{len(available)}, drops={drop_counts}")
    return x.loc[:, kept], kept, pd.DataFrame(stats)


def make_dataset_parts(df: pd.DataFrame, x: pd.DataFrame, cfg: dict):
    """Split into train/valid based on split_column."""
    input_cfg = cfg["feature_refine"]["input"]
    label_col = input_cfg["label_column"]
    split_col = input_cfg["split_column"]
    train_mask = (df[split_col] == input_cfg["train_value"]) & df[label_col].isin([0, 1])
    valid_mask = (df[split_col] == input_cfg["valid_value"]) & df[label_col].isin([0, 1])
    print(f"[SPLIT] train={train_mask.sum()}, valid={valid_mask.sum()}")
    return (
        x.loc[train_mask].reset_index(drop=True),
        df.loc[train_mask, label_col].astype(int).reset_index(drop=True),
        x.loc[valid_mask].reset_index(drop=True),
        df.loc[valid_mask, label_col].astype(int).reset_index(drop=True),
    )


def fill_for_model(train_x: pd.DataFrame, valid_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    return train_x.fillna(medians).fillna(0), valid_x.fillna(medians).fillna(0)


# ============================================================================
# Main diagnostic
# ============================================================================

def flag_suspicious(features: list[str]) -> list[dict]:
    flagged = []
    for feat in features:
        feat_lower = feat.lower()
        for pattern, reason in SUSPICIOUS_PATTERNS.items():
            if pattern in feat_lower:
                flagged.append({"feature": feat, "pattern": pattern, "reason": reason})
                break
    return flagged


def train_and_evaluate(train_x, train_y, valid_x, valid_y, features: list[str]) -> dict:
    print(f"[TRAIN] train: {train_x.shape}, valid: {valid_x.shape}")
    print(f"[TRAIN] y_train mean={train_y.mean():.4f}, y_valid mean={valid_y.mean():.4f}")

    params = dict(LGBM_PARAMS)
    n_boost = 400
    es = 50

    model = lgb.LGBMClassifier(**params, n_estimators=n_boost)
    model.fit(
        train_x, train_y,
        eval_set=[(train_x, train_y), (valid_x, valid_y)],
        eval_names=["train", "valid"],
        callbacks=[lgb.early_stopping(es), lgb.log_evaluation(50)],
    )

    train_pred = model.predict_proba(train_x)[:, 1]
    valid_pred = model.predict_proba(valid_x)[:, 1]

    train_auc = roc_auc_score(train_y, train_pred)
    valid_auc = roc_auc_score(valid_y, valid_pred)

    imp_df = pd.DataFrame({
        "feature": features,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    # Also train with fewer features to see if overfitting reduces
    top10_features = imp_df.head(10)["feature"].tolist()
    print(f"\n[TRAIN-TOP10] Re-training with only top 10 features...")
    top10_idx = [list(features).index(f) for f in top10_features]
    model10 = lgb.LGBMClassifier(**params, n_estimators=n_boost)
    model10.fit(
        train_x.iloc[:, top10_idx], train_y,
        eval_set=[(train_x.iloc[:, top10_idx], train_y), (valid_x.iloc[:, top10_idx], valid_y)],
        eval_names=["train", "valid"],
        callbacks=[lgb.early_stopping(es), lgb.log_evaluation(50)],
    )
    top10_train_auc = roc_auc_score(train_y, model10.predict_proba(train_x.iloc[:, top10_idx])[:, 1])
    top10_valid_auc = roc_auc_score(valid_y, model10.predict_proba(valid_x.iloc[:, top10_idx])[:, 1])

    return {
        "train_auc": train_auc,
        "valid_auc": valid_auc,
        "auc_gap": train_auc - valid_auc,
        "best_iter": model.best_iteration_,
        "importance": imp_df,
        "top10_train_auc": top10_train_auc,
        "top10_valid_auc": top10_valid_auc,
        "top10_auc_gap": top10_train_auc - top10_valid_auc,
    }


def main():
    config = load_yaml(PROJECT_DIR / "configs" / "refine_features.yaml")
    features_path = PROJECT_DIR / "runs" / "feature_refine_wide" / "final_500_features.txt"

    with open(features_path) as f:
        features = [line.strip() for line in f if line.strip()]
    print(f"[LOAD] {len(features)} features from {features_path}")

    # Flag suspicious features
    suspicious = flag_suspicious(features)
    print(f"\n[SUSPICIOUS] {len(suspicious)} features with potentially suspicious names:")
    for s in suspicious:
        print(f"  {s['feature']}  ← {s['reason']}")

    # Pull data - use larger sample for more reliable diagnosis
    print("\n[PULL] Fetching data with only the 73 selected features...")
    sample_cfg = config["feature_refine"]["sampling"].copy()
    # Loosen filter slightly for diagnostic - we only have 73 cols now, much faster
    t0 = time.time()
    sql = build_sampling_sql(config, features)
    print(f"[PULL] SQL (first 400 chars):\n{sql[:400]}...")
    df = load_wide_sample(sql)
    elapsed = time.time() - t0
    print(f"[PULL] Got {len(df)} rows, {len(df.columns)} cols in {elapsed:.1f}s")

    label_col = config["feature_refine"]["input"]["label_column"]
    split_col = config["feature_refine"]["input"]["split_column"]
    print(f"[PULL] label_dist={df[label_col].value_counts().to_dict()}")
    print(f"[PULL] split_dist={df[split_col].value_counts().to_dict()}")

    # Coerce & split
    x, kept_features, stats_df = coerce_feature_frame(df, features, config)
    train_x, train_y, valid_x, valid_y = make_dataset_parts(df, x, config)
    train_x, valid_x = fill_for_model(train_x, valid_x)

    # Train 73-feature model
    print("\n" + "=" * 60)
    print("TRAINING DIAGNOSTIC MODEL (73 features)")
    print("=" * 60)
    result = train_and_evaluate(train_x, train_y, valid_x, valid_y, kept_features)

    # Train 66-feature model (73 minus 7 suspicious 'suc'/'after' features, keep 'future' for now)
    remove_patterns = ["suc_", "_suc_", "after_", "next_"]
    clean_features = [f for f in kept_features if not any(p in f.lower() for p in remove_patterns)]
    clean_indices = [kept_features.index(f) for f in clean_features]
    print(f"\n{'=' * 60}")
    print(f"TRAINING WITHOUT SUSPICIOUS FEATURES ({len(clean_features)} features)")
    print(f"Removed: {set(kept_features) - set(clean_features)}")
    print("=" * 60)
    clean_result = train_and_evaluate(
        train_x.iloc[:, clean_indices], train_y,
        valid_x.iloc[:, clean_indices], valid_y,
        clean_features,
    )

    # ============================================================================
    # Report
    # ============================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"{'':<30} {'All 73':>10} {'No Suc/Next':>12}")
    print(f"  {'Train AUC':<28} {result['train_auc']:>10.4f} {clean_result['train_auc']:>12.4f}")
    print(f"  {'Valid AUC':<28} {result['valid_auc']:>10.4f} {clean_result['valid_auc']:>12.4f}")
    print(f"  {'AUC Gap':<28} {result['auc_gap']:>10.4f} {clean_result['auc_gap']:>12.4f}")
    print(f"  {'Best Iter':<28} {result['best_iter']:>10} {clean_result['best_iter']:>12}")
    print(f"  {'Top-10 Train AUC':<28} {result['top10_train_auc']:>10.4f} {clean_result['top10_train_auc']:>12.4f}")
    print(f"  {'Top-10 Valid AUC':<28} {result['top10_valid_auc']:>10.4f} {clean_result['top10_valid_auc']:>12.4f}")
    print(f"  {'Top-10 AUC Gap':<28} {result['top10_auc_gap']:>10.4f} {clean_result['top10_auc_gap']:>12.4f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    # Overfitting check
    if result["auc_gap"] > 0.05:
        print(f"OVERFITTING: AUC gap = {result['auc_gap']:.4f} (>0.05)")
    elif result["auc_gap"] > 0.02:
        print(f"MILD OVERFITTING: AUC gap = {result['auc_gap']:.4f} (0.02-0.05)")
    else:
        print(f"NO OVERFITTING: AUC gap = {result['auc_gap']:.4f} (<0.02)")

    # Leakage check - key indicator is VERY high valid AUC
    if result["valid_auc"] > 0.90:
        print(f"CRITICAL: Valid AUC = {result['valid_auc']:.4f} > 0.90 — almost certainly FEATURE LEAKAGE (特征穿越)")
        print("  Normal credit risk models rarely exceed 0.75-0.80 AUC.")
        print("  The model is predicting the target nearly perfectly on unseen OOT data.")
    elif result["valid_auc"] > 0.85:
        print(f"WARNING: Valid AUC = {result['valid_auc']:.4f} > 0.85 — likely feature leakage")
    elif result["valid_auc"] > 0.80:
        print(f"SUSPICIOUS: Valid AUC = {result['valid_auc']:.4f} > 0.80 — possible leakage")
    else:
        print(f"Valid AUC = {result['valid_auc']:.4f} — in normal range")

    # Check if removing suspicious features helps
    auc_drop = result["valid_auc"] - clean_result["valid_auc"]
    if auc_drop > 0.03:
        print(f"Dropping suc/next features reduces valid AUC by {auc_drop:.4f} — suspicious features contribute heavily")
    else:
        print(f"Dropping suc/next features only reduces valid AUC by {auc_drop:.4f} — leakage may be elsewhere")

    # Top features
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES BY GAIN (All 73 model)")
    print("=" * 60)
    imp = result["importance"]
    for i, (_, row) in enumerate(imp.head(20).iterrows()):
        flag = " <<< LEAKAGE?" if any(p in row["feature"].lower() for p in SUSPICIOUS_PATTERNS) else ""
        pct = row["gain"] / imp["gain"].sum() * 100
        print(f"  {i+1:2d}. {row['feature']}: gain={row['gain']:.1f} ({pct:.1f}%), split={row['split']}{flag}")

    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 60)
    print(f"  Suspicious features: {len(suspicious)}/{len(kept_features)}")
    print(f"  Top-1 gain dominance: {imp.iloc[0]['gain'] / imp['gain'].sum():.1%}")
    print(f"  Top-3 gain dominance: {imp.head(3)['gain'].sum() / imp['gain'].sum():.1%}")
    print(f"  'unpaid_principal_future_light_add_heavy' alone explains {imp[imp['feature'].str.contains('future')]['gain'].sum() / imp['gain'].sum():.1%} of total gain")

    if result["valid_auc"] > 0.85:
        print("\n  RECOMMENDATION: Investigate feature definitions for leakage.")
        print("  - Check if 'future' features use post-observation-period data")
        print("  - Check if 'suc_ord' features count future successful orders")
        print("  - Consider running D01/D02 with stricter feature-level temporal checks")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
