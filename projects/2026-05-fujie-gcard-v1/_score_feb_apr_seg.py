#!/usr/bin/env python
"""Score Feb-Apr 2026 by segment, compute by-month x segment AUC/KS."""
import gc, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

project_dir = Path(__file__).resolve().parent

# ── 1. Load model & features ──────────────────────────────────────
with open(project_dir / "runs/20260615_train_300/modeling/main_lgbm/model.pkl", "rb") as f:
    model = pickle.load(f)
with open(project_dir / "runs/20260615_train_300/modeling/main_lgbm/actual_feature_list.txt") as f:
    features = [l.strip() for l in f if l.strip()]
with open(project_dir / "runs/20260615_train_300/modeling/main_lgbm/preprocessing.json") as f:
    preproc = json.load(f)
fill_values = preproc.get("fill_values", {})
sentinels = preproc.get("missing_sentinels", [-999, -998])
print(f"Model loaded: {len(features)} features, best_iter={model.best_iteration}")

# ── 2. Load data ──────────────────────────────────────────────────
needed = ["uid","mdl_dte","final_flag","blue_customer_flag","ftr_30d_ord_flag","gcard_v6"] + features
raw = pd.read_feather("/root/notebook/draft/2026年2月到4月筛选300维度特征样本.feather",
                       columns=[c for c in needed if c != "uid"])
raw["mdl_month"] = pd.to_datetime(raw["mdl_dte"]).dt.strftime("%Y-%m")
# Convert v6 to numeric
raw["gcard_v6"] = pd.to_numeric(raw["gcard_v6"], errors="coerce").astype(np.float32)
print(f"Data: {len(raw)} rows, months={sorted(raw['mdl_month'].unique())}")

# ── 3. Score ──────────────────────────────────────────────────────
available = [f for f in features if f in raw.columns]
X = raw[available].copy()
for col in available:
    X[col] = X[col].replace(sentinels, np.nan).replace([np.inf, -np.inf], np.nan)
for col in available:
    if col in fill_values:
        X[col] = X[col].fillna(fill_values[col])
X = X.fillna(0)
raw["model_score"] = model.predict(X[available].values, num_iteration=model.best_iteration)
del X; gc.collect()
print("Scoring done")

# ── 4. Compute by-month x segment metrics ─────────────────────────
label = "ftr_30d_ord_flag"
seg_col = "blue_customer_flag"
results = []

for month in sorted(raw["mdl_month"].unique()):
    m = raw[raw["mdl_month"] == month]
    for seg in sorted(m[seg_col].dropna().unique()):
        for split_vals, split_name in [(["DEV","OOT"], "DEV+OOT"),
                                        (["DEV"], "DEV"), (["OOT"], "OOT"),
                                        (["DEV-OOS"], "DEV-OOS"), (["OOT-OOS"], "OOT-OOS")]:
            sub = m[m[seg_col] == seg]
            sub = sub[sub["final_flag"].isin(split_vals) & sub[label].isin([0,1])]
            if len(sub) < 100 or sub[label].nunique() < 2:
                continue

            y = sub[label].astype(int)
            pred = sub["model_score"].values
            model_auc = round(float(roc_auc_score(y, pred)), 4)
            model_ks = round(float(ks_2samp(pred[y==1], pred[y==0]).statistic), 4)

            v6_auc, v6_ks = None, None
            v6_pred = sub["gcard_v6"].values
            v6_ok = ~np.isnan(v6_pred)
            if v6_ok.sum() > 100 and y[v6_ok].nunique() > 1:
                v6_auc = round(float(roc_auc_score(y[v6_ok], v6_pred[v6_ok])), 4)
                v6_ks = round(float(ks_2samp(v6_pred[v6_ok][y[v6_ok]==1],
                                              v6_pred[v6_ok][y[v6_ok]==0]).statistic), 4)

            results.append({
                "month": month, "split": split_name, "segment": seg,
                "n_samples": len(sub), "positive": int(y.sum()),
                "bad_rate": round(float(y.mean()), 4),
                "model_auc": model_auc, "model_ks": model_ks,
                "v6_auc": v6_auc, "v6_ks": v6_ks,
            })

df_r = pd.DataFrame(results)
# Compute uplift
df_r["auc_uplift"] = (df_r["model_auc"] - df_r["v6_auc"]).round(4)
df_r["ks_uplift"] = (df_r["model_ks"] - df_r["v6_ks"]).round(4)

print("\n=== DEV+OOT by Month x Segment ===")
devoot = df_r[df_r["split"] == "DEV+OOT"].sort_values(["month","segment"])
print(devoot.to_string(index=False))

print("\n=== OOT-OOS by Month x Segment ===")
ootoos = df_r[df_r["split"] == "OOT-OOS"].sort_values(["month","segment"])
print(ootoos.to_string(index=False))

print("\n=== DEV-OOS by Month x Segment ===")
devoos = df_r[df_r["split"] == "DEV-OOS"].sort_values(["month","segment"])
print(devoos.to_string(index=False))

# Save
out = project_dir / "runs/20260615_train_300/evaluation/feb_apr_2026_segment_monthly.csv"
df_r.to_csv(out, index=False)
print(f"\nSaved to {out}")
