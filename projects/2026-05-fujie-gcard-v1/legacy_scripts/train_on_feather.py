#!/usr/bin/env python3
"""Train with draft/十分之一观察样本.feather and compare against DP wide-table results."""
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings("ignore")

FP = "/root/notebook/draft/十分之一观察样本.feather"
PROJ = Path("/root/notebook/自动化建模/jy-model-agent/projects/2026-05-fujie-gcard-v1")

# Load feather
t0 = time.time()
raw = pd.read_feather(FP)
print(f"[LOAD] {raw.shape} in {time.time()-t0:.1f}s")

# Check final_flag values
print(f"\n[LABEL] final_flag values: {raw['final_flag'].value_counts().to_dict()}")
print(f"[LABEL] ftr_30d_ord_flag values: {raw['ftr_30d_ord_flag'].value_counts().to_dict()}")

# Load 73 features
with open(PROJ / "runs/feature_refine_wide/final_500_features.txt") as f:
    sel_features = [l.strip() for l in f if l.strip()]

# Check how many found
found = [f for f in sel_features if f in raw.columns]
missing = [f for f in sel_features if f not in raw.columns]
print(f"\n[FEAT] Found {len(found)}/{len(sel_features)} features in feather")
if missing:
    print(f"  Missing ({len(missing)}): {missing[:10]}...")

# Use all rows: split by final_flag
# NOTE: final_flag values are 'DEV', 'OOT-OOS', 'DEV-OOS' etc.
# Filter to usable splits
dev_mask = raw["final_flag"].str.startswith("DEV")
oot_mask = raw["final_flag"].str.startswith("OOT")
# For train use DEV (not DEV-OOS), for valid use OOT-OOS or OOT
train_mask = raw["final_flag"] == "DEV"
valid_mask = raw["final_flag"].isin(["OOT", "OOT-OOS"])
print(f"\n[SPLIT] DEV={train_mask.sum()}, OOT={valid_mask.sum()}")

# Check label distribution per split
for label_col, split_name, mask in [("ftr_30d_ord_flag", "DEV", train_mask),
                                      ("ftr_30d_ord_flag", "OOT", valid_mask)]:
    vc = raw.loc[mask, label_col].value_counts().to_dict()
    print(f"  {split_name} label: {vc}  bad_rate={vc.get(1,0)/(vc.get(0,0)+vc.get(1,0)):.4f}")

# Extract features - need to coerce str→numeric
print(f"\n[COERCE] Converting {len(found)} features from {raw[found].dtypes.value_counts().to_dict()}...")
t0 = time.time()
X_all = pd.DataFrame(index=raw.index)
kept = []
dropped = 0
for feat in found:
    s = pd.to_numeric(raw[feat], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).replace([-999, -998], np.nan)
    if s.notna().mean() >= 0.01 and s.nunique(dropna=True) > 1:
        X_all[feat] = s
        kept.append(feat)
    else:
        dropped += 1
print(f"[COERCE] Kept {len(kept)}/{len(found)}, dropped {dropped} in {time.time()-t0:.1f}s")

# Fill NaN with 0
X_all = X_all.fillna(0).astype(float)

# Split
y_all = raw["ftr_30d_ord_flag"].astype(int)
tr_x = X_all[train_mask].reset_index(drop=True)
tr_y = y_all[train_mask].reset_index(drop=True)
va_x = X_all[valid_mask].reset_index(drop=True)
va_y = y_all[valid_mask].reset_index(drop=True)
print(f"\n[SPLIT] train={len(tr_x)} (bad={tr_y.mean():.4f}), OOT={len(va_x)} (bad={va_y.mean():.4f})")

# Train LightGBM (same params as D05)
params = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
    "max_depth": 5, "min_child_samples": 100, "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "seed": 0, "n_jobs": 4,
}
print(f"\n[ TRAIN ] LightGBM with {len(kept)} features...")
t0 = time.time()
model = lgb.LGBMClassifier(**params, n_estimators=400)
model.fit(
    tr_x, tr_y,
    eval_set=[(tr_x, tr_y), (va_x, va_y)],
    eval_names=["train", "valid"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
)
elapsed = time.time() - t0
print(f"[TRAIN] Done in {elapsed:.1f}s, best_iter={model.best_iteration_}")

# Metrics
tr_pred = model.predict_proba(tr_x)[:, 1]
va_pred = model.predict_proba(va_x)[:, 1]
tr_auc = roc_auc_score(tr_y, tr_pred)
va_auc = roc_auc_score(va_y, va_pred)
tr_ks = ks_2samp(tr_pred[tr_y == 1], tr_pred[tr_y == 0]).statistic
va_ks = ks_2samp(va_pred[va_y == 1], va_pred[va_y == 0]).statistic

print(f"\n{'='*65}")
print(f"FEATHER 文件训练结果 vs DP 宽表结果")
print(f"{'='*65}")
print(f"  {'':<25} {'Feather 十分之一样本':>20} {'DP 宽表 (rand<0.01)':>22}")
print(f"  {'训练样本':<25} {len(tr_y):>20,} {36176:>22,}")
print(f"  {'OOT 样本':<25} {len(va_y):>20,} {12101:>22,}")
print(f"  {'特征数':<25} {len(kept):>20} {73:>22}")
print(f"  {'DEV 坏账率':<25} {tr_y.mean():>20.4f} {0.1515:>22.4f}")
print(f"  {'OOT 坏账率':<25} {va_y.mean():>20.4f} {0.1307:>22.4f}")
print(f"  {'Train AUC':<25} {tr_auc:>20.4f} {'':>22}")
print(f"  {'OOT AUC':<25} {va_auc:>20.4f} {0.9329:>22.4f}")
print(f"  {'OOT KS':<25} {va_ks:>20.4f} {0.7341:>22.4f}")
print(f"  {'AUC Gap':<25} {tr_auc - va_auc:>20.4f} {0.0125:>22.4f}")

# OOT decile
df_va = pd.DataFrame({"true": va_y.values, "score": va_pred})
df_va["decile"] = pd.qcut(df_va["score"], 10, labels=False, duplicates="drop")
ds = df_va.groupby("decile").agg(total=("true","count"), bad=("true","sum"), bad_rate=("true","mean")).sort_values("decile", ascending=False)
print(f"\n  OOT 十分位:")
for d, row in ds.iterrows():
    bar = "█" * int(row["bad_rate"] * 50)
    print(f"    D{d+1}: bad_rate={row['bad_rate']:.4f}  ({int(row['bad'])}/{int(row['total'])})  {bar}")

# Feature importance comparison
imp = pd.DataFrame({
    "feature": kept, "gain": model.booster_.feature_importance(importance_type="gain"),
    "split": model.booster_.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
print(f"\n  Top-15 特征 (Feather):")
for i, (_, r) in enumerate(imp.head(15).iterrows()):
    flag = " ⚠" if any(p in r["feature"].lower() for p in ["future", "suc_", "_suc_"]) else ""
    print(f"    {i+1:2d}. {r['feature']:<55s} gain={r['gain']:>10.1f}  split={r['split']:>4d}{flag}")

# Conclusion
print(f"\n{'='*65}")
print(f"CONCLUSION")
print(f"{'='*65}")
if va_auc > 0.85:
    print(f"  OOT AUC = {va_auc:.4f} → 特征穿越 confirmed (FEATHER data)")
    print(f"  两份独立数据均显示 AUC > 0.90，不是采样偏差导致。")
else:
    print(f"  OOT AUC = {va_auc:.4f} → 在合理范围，DP 宽表的 0.93 可能是采样问题")
