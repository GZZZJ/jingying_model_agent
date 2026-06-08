#!/usr/bin/env python3
"""Show OOT detailed metrics for the selected-feature model."""
import argparse
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
REPO = SCRIPT.parents[3]
sys.path.insert(0, str(REPO))
from jingying_agent.config import load_yaml
from jingying_agent.dp_feather import default_dataset_paths, load_or_fetch_dp_feather, print_sql_review, write_dataset_metadata

parser = argparse.ArgumentParser(description="Show OOT detailed metrics for the selected-feature model.")
parser.add_argument("--dry-run-sql", action="store_true", help="Print SQL and metadata path without querying DP.")
parser.add_argument("--refresh-dp-cache", action="store_true", help="Refresh local feather cache from DP after SQL approval.")
parser.add_argument("--sql-approved", action="store_true", help="Confirm that the displayed DP SQL has been reviewed.")
parser.add_argument(
    "--features-path",
    default=None,
    help="Selected feature list. Defaults to feature_refine_wide if present, otherwise feature_refine_feather.",
)
args = parser.parse_args()

config = load_yaml(PROJECT / "configs" / "refine_features.yaml")

def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path

def default_features_path(project_dir: Path) -> Path:
    wide_path = project_dir / "runs" / "feature_refine_wide" / "final_500_features.txt"
    if wide_path.exists():
        return wide_path
    return project_dir / "runs" / "feature_refine_feather" / "final_500_features.txt"

# Load features
features_path = resolve_project_path(PROJECT, args.features_path) if args.features_path else default_features_path(PROJECT)
with open(features_path) as f:
    features = [l.strip() for l in f if l.strip()]

# Pull data
def sql_ident(n): return n if n.replace("_", "").isalnum() and not n[0].isdigit() else f"`{n}`"
cfg_in = config["feature_refine"]["input"]
base = list(dict.fromkeys(cfg_in["base_columns"]))
sel_cols = base + [f for f in features if f not in base]
sel_expr = ",\n  ".join(sql_ident(c) for c in sel_cols)
sql = f"select\n  {sel_expr}\nfrom {cfg_in['wide_table']}"
s = config["feature_refine"]["sampling"]
if s.get("where"): sql += f"\nwhere {s['where']}"
if s.get("max_rows"): sql += f"\nlimit {int(s['max_rows'])}"

dataset_id = "oot_report_selected_feature_sample"
feather_path, metadata_path = default_dataset_paths(PROJECT, dataset_id=dataset_id)
description = "所选特征OOT效果明细抽样，用于输出DEV/OOT AUC、KS、十分位和特征重要性。"
if args.dry_run_sql:
    write_dataset_metadata(
        project_dir=PROJECT,
        metadata_path=metadata_path,
        feather_path=feather_path,
        dataset_id=dataset_id,
        description=description,
        sql=sql,
        status="sql_review_required" if args.refresh_dp_cache or not feather_path.exists() else "ready",
        note="Review this SQL before running DP fetch. The feather file itself is gitignored.",
    )
    print_sql_review(
        dataset_id=dataset_id,
        description=description,
        feather_path=feather_path,
        metadata_path=metadata_path,
        sql=sql,
    )
    raise SystemExit(0)
df = load_or_fetch_dp_feather(
    project_dir=PROJECT,
    sql=sql,
    dataset_id=dataset_id,
    description=description,
    feather_path=feather_path,
    metadata_path=metadata_path,
    refresh=args.refresh_dp_cache,
    sql_approved=args.sql_approved,
)
print(f"[DATA] {len(df)} rows x {len(df.columns)} cols")

# Coerce
prep = config["feature_refine"]["preprocessing"]
sentinels = prep.get("missing_sentinels", [])
avail = [f for f in features if f in df.columns]
x = df[avail].copy()
kept = []
for feat in avail:
    s = pd.to_numeric(x[feat], errors="coerce")
    for sv in sentinels: s = s.replace(sv, np.nan)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().mean() >= float(prep.get("min_non_null_rate", 0)):
        if not prep.get("drop_constant", True) or s.nunique(dropna=True) > 1:
            kept.append(feat)
            x[feat] = s
x = x[kept]
print(f"[COERCE] {len(kept)} features")

# Split
label = cfg_in["label_column"]
split_c = cfg_in["split_column"]
train_m = (df[split_c] == cfg_in["train_value"]) & df[label].isin([0, 1])
valid_m = (df[split_c] == cfg_in["valid_value"]) & df[label].isin([0, 1])
tr_x = x[train_m].reset_index(drop=True)
tr_y = df.loc[train_m, label].astype(int).reset_index(drop=True)
va_x = x[valid_m].reset_index(drop=True)
va_y = df.loc[valid_m, label].astype(int).reset_index(drop=True)
med = tr_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
tr_x, va_x = tr_x.fillna(med).fillna(0), va_x.fillna(med).fillna(0)
print(f"[SPLIT] train={len(tr_x)} (bad={tr_y.mean():.4f}), OOT={len(va_x)} (bad={va_y.mean():.4f})")

# Train
params = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05, "num_leaves": 31,
    "max_depth": 5, "min_child_samples": 100, "subsample": 0.7, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "seed": 0,
}
model = lgb.LGBMClassifier(**params, n_estimators=400)
model.fit(tr_x, tr_y, eval_set=[(va_x, va_y)], eval_names=["OOT"],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

va_pred = model.predict_proba(va_x)[:, 1]
tr_pred = model.predict_proba(tr_x)[:, 1]

# === OOT Metrics ===
auc_oot = roc_auc_score(va_y, va_pred)
auc_tr = roc_auc_score(tr_y, tr_pred)
ks_oot = ks_2samp(va_pred[va_y == 1], va_pred[va_y == 0]).statistic
ks_tr = ks_2samp(tr_pred[tr_y == 1], tr_pred[tr_y == 0]).statistic

print(f"\n{'='*60}")
print(f"OOT 验证集详细效果")
print(f"{'='*60}")
print(f"  {'':<20} {'DEV (训练)':>15} {'OOT (验证)':>15}")
print(f"  {'样本数':<20} {len(tr_y):>15} {len(va_y):>15}")
print(f"  {'坏账率':<20} {tr_y.mean():>15.4f} {va_y.mean():>15.4f}")
print(f"  {'AUC':<20} {auc_tr:>15.4f} {auc_oot:>15.4f}")
print(f"  {'KS':<20} {ks_tr:>15.4f} {ks_oot:>15.4f}")
print(f"  {'最佳迭代':<20} {'':>15} {model.best_iteration_:>15}")

# Decile analysis
df_oot = pd.DataFrame({"true": va_y.values, "score": va_pred})
df_oot["decile"] = pd.qcut(df_oot["score"], 10, labels=False, duplicates="drop")
ds = df_oot.groupby("decile").agg(
    total=("true", "count"), bad=("true", "sum"), bad_rate=("true", "mean"),
    min_s=("score", "min"), max_s=("score", "max"),
).sort_values("decile", ascending=False)

print(f"\n{'='*60}")
print(f"OOT 十分位分析")
print(f"{'='*60}")
print(f"{'分位':<6} {'样本':<8} {'坏账数':<8} {'坏账率':<10} {'累计坏账%':<12} {'最低分':<10} {'最高分':<10}")
cum_bad = 0
total_bad = ds["bad"].sum()
for d, row in ds.iterrows():
    cum_bad += row["bad"]
    print(f'{d+1:<6} {int(row["total"]):<8} {int(row["bad"]):<8} {row["bad_rate"]:.4f}     {cum_bad/total_bad*100:>6.1f}%     {row["min_s"]:.4f}     {row["max_s"]:.4f}')

overall = va_y.mean()
top = ds.iloc[0]
bot = ds.iloc[-1]
print(f"\n  Top 10% lift:   {top['bad_rate']/overall:.1f}x (bad_rate={top['bad_rate']:.4f} vs overall={overall:.4f})")
print(f"  Top 20% recall: {ds.head(2)['bad'].sum()/total_bad*100:.1f}%")
print(f"  Top 30% recall: {ds.head(3)['bad'].sum()/total_bad*100:.1f}%")
print(f"  Bottom 10% bad_rate: {bot['bad_rate']:.4f}")

# Feature importance on OOT
imp = pd.DataFrame({
    "feature": kept, "gain": model.booster_.feature_importance(importance_type="gain"),
    "split": model.booster_.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
suspicious = ["future", "suc_", "_suc_", "after_", "next_"]
print(f"\n{'='*60}")
print(f"OOT 模型 Top-15 特征重要性")
print(f"{'='*60}")
for i, (_, r) in enumerate(imp.head(15).iterrows()):
    flag = " <<<" if any(p in r["feature"].lower() for p in suspicious) else ""
    print(f"  {i+1:2d}. {r['feature']:<55s} gain={r['gain']:>10.1f}  split={r['split']:>4d}{flag}")
