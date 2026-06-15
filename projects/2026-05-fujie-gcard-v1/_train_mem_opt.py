#!/usr/bin/env python
"""Memory-optimized training: convert string features to float32, then train.

Loads the feather, converts str columns to float32 in-place to avoid copies,
then trains LightGBM.  For scoring OOS data, processes in batches.
"""
import gc, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

project_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_dir.parent.parent))

config = yaml.safe_load((project_dir / "configs/train.yaml").read_text())
train_cfg = config["training"]
input_cfg = config["input"]
lgb_cfg = config["lightgbm"]
preproc_cfg = config.get("preprocessing", {})

# ── 1. Load feature list ──────────────────────────────────────────
feature_list_path = project_dir / "runs/feature_refine_feather/final_features.txt"
candidate_features = [
    line.strip() for line in feature_list_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]
print(f"[1/7] Feature list: {len(candidate_features)} features")

# ── 2. Load data & convert string→float32 in-place (memory-aware) ─
print(f"[2/7] Loading feather and converting string→float32 ...")
t0 = time.time()
label_col = input_cfg["label_column"]
split_col = input_cfg["split_column"]
base_cols = input_cfg.get("base_columns", [])
historical_scores = input_cfg.get("historical_score_columns", [])

needed_cols = list(dict.fromkeys(
    base_cols + historical_scores + [label_col, split_col] + candidate_features
))

raw = pd.read_feather(
    "/root/notebook/draft/筛选300维度特征样本.feather",
    columns=[c for c in needed_cols if c != "uid"]  # skip uid if not needed
)

# Convert string feature columns to float32 IN-PLACE to minimize memory
str_cols = [c for c in candidate_features if c in raw.columns and raw[c].dtype == object or hasattr(raw[c].dtype, 'type') and 'string' in str(raw[c].dtype)]
print(f"  Converting {len(str_cols)} string feature columns to float32...")
for i, col in enumerate(str_cols):
    raw[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float32)
    if (i + 1) % 50 == 0:
        print(f"  ... {i+1}/{len(str_cols)} columns converted")
        gc.collect()

# Convert remaining numeric features to float32 too
for col in candidate_features:
    if col in raw.columns and col not in str_cols:
        if raw[col].dtype == np.float64:
            raw[col] = raw[col].astype(np.float32)

gc.collect()
mem_gb = raw.memory_usage(deep=True).sum() / 1e9
n_rows = len(raw)
print(f"  Loaded {n_rows} rows, memory: {mem_gb:.1f} GB, time: {time.time()-t0:.0f}s")

# ── 3. Split into train/valid/oos ─────────────────────────────────
print(f"[3/7] Splitting data...")
train_values = train_cfg.get("train_values", ["DEV"])
valid_values = train_cfg.get("valid_values", ["OOT"])
oos_values = train_cfg.get("oos_values", ["DEV-OOS", "OOT-OOS"])

tr_mask = raw[split_col].isin(train_values) & raw[label_col].isin([0, 1])
va_mask = raw[split_col].isin(valid_values) & raw[label_col].isin([0, 1])

tr_raw = raw[tr_mask].copy()
va_raw = raw[va_mask].copy()
del tr_mask, va_mask
gc.collect()
print(f"  Train: {len(tr_raw)}, Valid: {len(va_raw)}")

# ── 4. Extract feature matrices ───────────────────────────────────
print(f"[4/7] Preparing feature matrices...")
available = [f for f in candidate_features if f in raw.columns]
tr_y = tr_raw[label_col].astype(int)
va_y = va_raw[label_col].astype(int)

tr_x = tr_raw[available].copy()
va_x = va_raw[available].copy()
del tr_raw, va_raw
gc.collect()

# Handle missing values
sentinels = preproc_cfg.get("missing_sentinels", [-999, -998])
for col in available:
    tr_x[col] = tr_x[col].replace(sentinels, np.nan).replace([np.inf, -np.inf], np.nan)
    va_x[col] = va_x[col].replace(sentinels, np.nan).replace([np.inf, -np.inf], np.nan)

# Fill NA with train medians
medians = tr_x.median(numeric_only=True).fillna(0)
tr_x = tr_x.fillna(medians).fillna(0)
va_x = va_x.fillna(medians).fillna(0)
gc.collect()
print(f"  tr_x: {tr_x.shape}, va_x: {va_x.shape}")

# ── 5. Train LightGBM ─────────────────────────────────────────────
print(f"[5/7] Training LightGBM...")
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

params = {
    "objective": lgb_cfg.get("objective", "binary"),
    "metric": lgb_cfg.get("metric", "auc"),
    "learning_rate": lgb_cfg.get("learning_rate", 0.05),
    "num_leaves": lgb_cfg.get("num_leaves", 31),
    "max_depth": lgb_cfg.get("max_depth", 5),
    "min_child_samples": lgb_cfg.get("min_child_samples", 100),
    "subsample": lgb_cfg.get("subsample", 0.7),
    "colsample_bytree": lgb_cfg.get("colsample_bytree", 0.7),
    "reg_alpha": lgb_cfg.get("reg_alpha", 0.1),
    "reg_lambda": lgb_cfg.get("reg_lambda", 1.0),
    "verbose": -1,
    "seed": train_cfg.get("random_seed", 0),
    "feature_fraction_seed": train_cfg.get("random_seed", 0),
    "bagging_seed": train_cfg.get("random_seed", 0),
}

train_ds = lgb.Dataset(tr_x, label=tr_y, feature_name=available, free_raw_data=False)
valid_ds = lgb.Dataset(va_x, label=va_y, feature_name=available, reference=train_ds, free_raw_data=False)
t_train = time.time()
model = lgb.train(
    params,
    train_ds,
    num_boost_round=lgb_cfg.get("num_boost_round", 1000),
    valid_sets=[valid_ds],
    callbacks=[
        lgb.early_stopping(lgb_cfg.get("early_stopping_rounds", 50), verbose=False),
        lgb.log_evaluation(period=20),
    ],
)
best_iter = model.best_iteration
train_time = time.time() - t_train

tr_pred = model.predict(tr_x, num_iteration=best_iter)
va_pred = model.predict(va_x, num_iteration=best_iter)
metrics = {
    "train_auc": float(roc_auc_score(tr_y, tr_pred)),
    "valid_auc": float(roc_auc_score(va_y, va_pred)),
    "train_ks": float(ks_2samp(tr_pred[tr_y == 1], tr_pred[tr_y == 0]).statistic),
    "valid_ks": float(ks_2samp(va_pred[va_y == 1], va_pred[va_y == 0]).statistic),
    "train_samples": int(len(tr_y)),
    "valid_samples": int(len(va_y)),
    "train_bad_rate": float(tr_y.mean()),
    "valid_bad_rate": float(va_y.mean()),
    "best_iteration": int(best_iter),
    "train_time_seconds": round(train_time, 1),
}
metrics["auc_gap"] = metrics["train_auc"] - metrics["valid_auc"]

print(f"\n  train_auc={metrics['train_auc']:.4f}, valid_auc={metrics['valid_auc']:.4f}")
print(f"  train_ks={metrics['train_ks']:.4f}, valid_ks={metrics['valid_ks']:.4f}")
print(f"  best_iter={metrics['best_iteration']}, time={train_time:.0f}s")

# Free train/valid data before scoring
del tr_x, va_x, tr_y, va_y, train_ds, valid_ds
gc.collect()

# ── 6. Save artifacts ─────────────────────────────────────────────
print(f"[6/7] Saving model artifacts...")
output_dir = project_dir / "runs/model_train/main_lgbm"
output_dir.mkdir(parents=True, exist_ok=True)

(output_dir / "metrics_train_valid.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

# Feature importance
importance = pd.DataFrame({
    "feature": available,
    "gain": model.feature_importance(importance_type="gain"),
    "split": model.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)
importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

# Run config
run_config = {
    "experiment": "main_lgbm",
    "data_source": "/root/notebook/draft/筛选300维度特征样本.feather",
    "train_values": train_values,
    "valid_values": valid_values,
    "oos_values": oos_values,
    "label_column": label_col,
    "split_column": split_col,
    "feature_list_path": str(feature_list_path),
    "candidate_feature_count": len(candidate_features),
    "actual_feature_count": len(available),
    "algorithm": "lightgbm",
    "params": {k: v for k, v in params.items() if not k.endswith("_seed") and k != "seed"},
    "random_seed": train_cfg.get("random_seed", 0),
    **metrics,
}
(output_dir / "run_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

# Save model
import pickle
with (output_dir / "model.pkl").open("wb") as f:
    pickle.dump(model, f)

# Candidate + actual feature lists
(output_dir / "candidate_feature_list.txt").write_text("\n".join(candidate_features) + "\n")
(output_dir / "actual_feature_list.txt").write_text("\n".join(available) + "\n")

# Preprocessing info
preprocessing = {
    "candidate_feature_count": len(candidate_features),
    "kept_feature_count": len(available),
    "dropped_feature_count": 0,
    "missing_sentinels": sentinels,
    "fill_strategy": "train_median_fill_zero",
}
(output_dir / "preprocessing.json").write_text(
    json.dumps(preprocessing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

# ── 7. Score OOS in batches ───────────────────────────────────────
print(f"[7/7] Scoring OOS data in batches...")
desired_base = ["uid", "mdl_dte", "ds", "final_flag", "blue_customer_flag",
                "zc_level", "ftr_30d_ord_flag", "ftr_30d_ord_amt",
                "prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m"]
available_hist = [c for c in historical_scores if c in raw.columns]
available_base = [c for c in desired_base if c in raw.columns]

# Score in batches to avoid OOM
batch_size = 500_000
n_batches = (n_rows + batch_size - 1) // batch_size
score_parts = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, n_rows)
    print(f"  Batch {i+1}/{n_batches}: rows {start_idx}-{end_idx}...")

    batch_raw = raw.iloc[start_idx:end_idx]
    batch_x = batch_raw[available].copy()
    # Fill NA
    for col in available:
        batch_x[col] = batch_x[col].replace(sentinels, np.nan).replace([np.inf, -np.inf], np.nan)
    batch_x = batch_x.fillna(medians).fillna(0)

    batch_scores = batch_raw[available_base].copy()
    batch_scores["model_score"] = model.predict(batch_x[available].values, num_iteration=best_iter)
    for col in available_hist:
        batch_scores[col] = batch_raw[col]

    score_parts.append(batch_scores)
    del batch_raw, batch_x, batch_scores
    gc.collect()

scores = pd.concat(score_parts, ignore_index=True)
del score_parts, raw
gc.collect()

score_output = output_dir / "scores.feather"
scores.reset_index(drop=True).to_feather(str(score_output))
print(f"  Scores written: {len(scores)} rows → {score_output}")
print(f"  Score file size: {score_output.stat().st_size / 1e9:.2f} GB")

# ── Done ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training complete! Total time: {time.time()-t0:.0f}s")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
