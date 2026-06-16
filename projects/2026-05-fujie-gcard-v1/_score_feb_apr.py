#!/usr/bin/env python
"""Score Feb-Apr 2026 sample with trained 300-feature model, compute by-month AUC/KS."""
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

project_dir = Path(__file__).resolve().parent

# ── 1. Load model ──────────────────────────────────────────────────
print("[1/5] Loading model...")
model_path = project_dir / "runs/20260615_train_300/modeling/main_lgbm/model.pkl"
if not model_path.exists():
    # Fallback: search for model.pkl in other run dirs
    alt_path = project_dir / "runs/model_train/main_lgbm/model.pkl"
    if alt_path.exists():
        model_path = alt_path
    else:
        print("ERROR: model.pkl not found")
        sys.exit(1)
import pickle
with open(model_path, "rb") as f:
    model = pickle.load(f)
print(f"  Model loaded: best_iter={model.best_iteration}")

# ── 2. Load feature list ──────────────────────────────────────────
with open(project_dir / "runs/20260615_train_300/modeling/main_lgbm/actual_feature_list.txt") as f:
    features = [l.strip() for l in f if l.strip()]
print(f"  Features: {len(features)}")

# Load preprocessing (fill values)
with open(project_dir / "runs/20260615_train_300/modeling/main_lgbm/preprocessing.json") as f:
    preproc = json.load(f)
fill_values = preproc.get("fill_values", {})
sentinels = preproc.get("missing_sentinels", [-999, -998])

# ── 3. Load data ──────────────────────────────────────────────────
print("[2/5] Loading data...")
t0 = time.time()
needed_cols = ["uid", "mdl_dte", "ds", "final_flag", "blue_customer_flag",
               "ftr_30d_ord_flag", "gcard_v6"] + features
raw = pd.read_feather(
    "/root/notebook/draft/2026年2月到4月筛选300维度特征样本.feather",
    columns=[c for c in needed_cols if c != "uid"]
)
print(f"  Loaded {len(raw)} rows in {time.time()-t0:.0f}s")

# Convert string features to float32
str_cols = 0
for col in features:
    if col in raw.columns and raw[col].dtype == object:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float32)
        str_cols += 1
print(f"  Converted {str_cols} str columns to float32")

# Create month column
raw["mdl_month"] = pd.to_datetime(raw["mdl_dte"], errors="coerce").dt.to_period("M").astype(str)
print(f"  Months: {sorted(raw['mdl_month'].unique())}")

# ── 4. Score ──────────────────────────────────────────────────────
print("[3/5] Scoring...")
available = [f for f in features if f in raw.columns]
X = raw[available].copy()

# Handle sentinels and inf
for col in available:
    X[col] = X[col].replace(sentinels, np.nan).replace([np.inf, -np.inf], np.nan)

# Fill NA with preproc medians
for col in available:
    if col in fill_values:
        X[col] = X[col].fillna(fill_values[col])
X = X.fillna(0)

raw["model_score"] = model.predict(X[available].values, num_iteration=model.best_iteration)
del X; gc.collect()
print(f"  Scoring done")

# Also load gcard_v6 score if available
has_v6 = "gcard_v6" in raw.columns and raw["gcard_v6"].notna().sum() > 0
if has_v6:
    raw["gcard_v6"] = pd.to_numeric(raw["gcard_v6"], errors="coerce").astype(np.float32)

# ── 5. Compute by-month metrics ──────────────────────────────────
print("[4/5] Computing metrics...")
label = "ftr_30d_ord_flag"

results = []
for month in sorted(raw["mdl_month"].unique()):
    m = raw[raw["mdl_month"] == month]

    # DEV + OOT combined (user requested)
    for split_name, split_values in [("DEV", ["DEV"]), ("OOT", ["OOT"]),
                                      ("DEV-OOS", ["DEV-OOS"]), ("OOT-OOS", ["OOT-OOS"]),
                                      ("DEV+OOT", ["DEV", "OOT"])]:
        mask = m["final_flag"].isin(split_values) & m[label].isin([0, 1])
        sub = m[mask]
        if len(sub) < 100 or sub[label].nunique() < 2:
            results.append({
                "month": month, "split": split_name,
                "n_samples": len(sub), "bad_rate": None,
                "model_auc": None, "model_ks": None,
                "v6_auc": None, "v6_ks": None,
            })
            continue

        y = sub[label].astype(int)
        pred = sub["model_score"].values
        model_auc = float(roc_auc_score(y, pred))
        model_ks = float(ks_2samp(pred[y == 1], pred[y == 0]).statistic)

        v6_auc, v6_ks = None, None
        if has_v6 and sub["gcard_v6"].notna().sum() > 100:
            v6_pred = sub["gcard_v6"].values
            v6_mask = ~np.isnan(v6_pred) & y.notna().values
            if v6_mask.sum() > 100:
                v6_auc = float(roc_auc_score(y[v6_mask], v6_pred[v6_mask]))
                v6_ks = float(ks_2samp(v6_pred[v6_mask][y[v6_mask] == 1],
                                       v6_pred[v6_mask][y[v6_mask] == 0]).statistic)

        results.append({
            "month": month, "split": split_name,
            "n_samples": len(sub),
            "positive": int(y.sum()),
            "bad_rate": round(float(y.mean()), 4),
            "model_auc": round(model_auc, 4), "model_ks": round(model_ks, 4),
            "v6_auc": round(v6_auc, 4) if v6_auc else None,
            "v6_ks": round(v6_ks, 4) if v6_ks else None,
        })

df_result = pd.DataFrame(results)
print("\n[DONE] Results:")
print(df_result.to_string(index=False))

# ── 6. Save ───────────────────────────────────────────────────────
out_path = project_dir / "runs/20260615_train_300/evaluation/feb_apr_2026_monthly_metrics.csv"
df_result.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
