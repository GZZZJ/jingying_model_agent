#!/usr/bin/env python3
"""Formal training script for 复借G卡 main model.

Trains a single LightGBM model (main_lgbm) on the specified sample data
and feature list, then scores all samples and saves the unified prediction
table.

Usage:
    python3 scripts/03_train.py \
      --input-feather /root/notebook/draft/十分之一观察样本.feather \
      --feature-list runs/modeling_feature_set/feature_list.txt \
      --output-dir runs/model_train/main_lgbm \
      --score-output runs/model_scores/scores_all_splits.feather \
      --input-dir runs/modeling_input \
      --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "agent.py").exists():
            return candidate
    raise RuntimeError("Cannot locate repo root from script path")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(REPO_ROOT))

from jingying_agent.config import load_yaml


# ── helper functions ──────────────────────────────────────────────


def coerce_features(
    df: pd.DataFrame,
    features: list[str],
    sentinels: list[int],
    min_non_null_rate: float,
    drop_constant: bool,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Coerce features to numeric, drop low-quality columns.

    Returns (x_frame, kept_features, drop_detail_df).
    """
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    x = df.loc[:, available].copy()
    kept: list[str] = []
    stats: list[dict[str, Any]] = []

    for feature in available:
        series = pd.to_numeric(x[feature], errors="coerce")
        if sentinels:
            series = series.replace(sentinels, np.nan)
        series = series.replace([np.inf, -np.inf], np.nan)
        nn_rate = float(series.notna().mean())
        nuniq = int(series.nunique(dropna=True))
        drop_reason = ""
        if nn_rate < min_non_null_rate:
            drop_reason = "low_non_null_rate"
        elif drop_constant and nuniq <= 1:
            drop_reason = "constant"
        else:
            kept.append(feature)
            x[feature] = series
        stats.append({
            "feature": feature, "non_null_rate": nn_rate,
            "unique_count": nuniq, "drop_reason": drop_reason,
        })

    # Add missing features to drop detail
    for f in missing:
        stats.append({
            "feature": f, "non_null_rate": 0.0,
            "unique_count": 0, "drop_reason": "missing_from_data",
        })

    drop_detail = pd.DataFrame(stats)
    drop_counts = drop_detail["drop_reason"].value_counts().to_dict()
    print(f"[COERCE] kept={len(kept)}/{len(available)}, missing={len(missing)}, drops={drop_counts}")
    return x.loc[:, kept], kept, drop_detail


def fill_na_from_train(
    train_x: pd.DataFrame, valid_x: pd.DataFrame, *other_frames: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Fill NaN using train-set medians; fallback to 0."""
    medians = train_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    result = [train_x.fillna(medians).fillna(0), valid_x.fillna(medians).fillna(0)]
    for frame in other_frames:
        result.append(frame.fillna(medians).fillna(0))
    return tuple(result)


def build_preprocessing_json(
    kept_features: list[str],
    drop_detail: pd.DataFrame,
    medians: pd.Series,
    sentinels: list[int],
    min_non_null_rate: float,
) -> dict[str, Any]:
    """Build preprocessing.json payload."""
    dropped = drop_detail[drop_detail["drop_reason"] != ""]
    return {
        "candidate_feature_count": len(drop_detail),
        "kept_feature_count": len(kept_features),
        "dropped_feature_count": len(dropped),
        "missing_sentinels": sentinels,
        "min_non_null_rate": min_non_null_rate,
        "fill_strategy": "train_median_fill_zero",
        "drop_reason_counts": drop_detail["drop_reason"].value_counts().to_dict(),
        "fill_values": {f: float(v) for f, v in medians.items()},
    }


# ── main ───────────────────────────────────────────────────────────


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Train 复借G卡 main model")
    parser.add_argument("--input-feather", required=True, help="Path to modeling sample feather")
    parser.add_argument("--feature-list", required=True, help="Path to feature_list.txt")
    parser.add_argument("--output-dir", required=True, help="Training output directory")
    parser.add_argument("--score-output", required=True, help="Path for scores_all_splits.feather")
    parser.add_argument("--input-dir", default=str(project_dir / "runs/modeling_input"),
                        help="Directory for input snapshot")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config yaml")
    args = parser.parse_args()

    # Resolve paths
    input_feather = Path(args.input_feather)
    feature_list_path = project_dir / args.feature_list if not Path(args.feature_list).is_absolute() else Path(args.feature_list)
    output_dir = project_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    score_output = project_dir / args.score_output if not Path(args.score_output).is_absolute() else Path(args.score_output)
    input_snapshot_dir = project_dir / args.input_dir if not Path(args.input_dir).is_absolute() else Path(args.input_dir)

    # Load config
    config_path = project_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_yaml(config_path)
    train_cfg = cfg["training"]
    input_cfg = cfg["input"]
    lgb_cfg = cfg["lightgbm"]
    preproc_cfg = cfg["preprocessing"]

    output_dir.mkdir(parents=True, exist_ok=True)
    score_output.parent.mkdir(parents=True, exist_ok=True)
    input_snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load feature list ──────────────────────────────────────
    with open(feature_list_path, "r", encoding="utf-8") as fh:
        candidate_features = [line.strip() for line in fh if line.strip()]
    print(f"[FEATURE] Candidate features: {len(candidate_features)}")

    # ── 2. Load sample data ───────────────────────────────────────
    print(f"[LOAD] Reading {input_feather} ...")
    t0 = time.time()

    # Identify columns to read
    id_cols = input_cfg.get("id_columns", ["uid", "mdl_dte"])
    base_cols = input_cfg.get("base_columns", [])
    label_col = input_cfg["label_column"]
    split_col = input_cfg["split_column"]
    # Columns needed for scoring: base + features
    read_cols = list(dict.fromkeys(
        id_cols + base_cols + [label_col, split_col] + candidate_features
    ))
    # Only read columns that exist in the feather
    all_feather_cols = pd.read_feather(input_feather, columns=None).columns.tolist()
    read_cols = [c for c in read_cols if c in all_feather_cols]
    raw = pd.read_feather(input_feather, columns=read_cols)
    print(f"[LOAD] {raw.shape} in {time.time() - t0:.1f}s")

    # ── 3. Build train / valid / oos masks ────────────────────────
    train_values = train_cfg.get("train_values", ["DEV"])
    valid_values = train_cfg.get("valid_values", ["OOT"])
    oos_values = train_cfg.get("oos_values", ["DEV-OOS", "OOT-OOS"])

    train_mask = raw[split_col].isin(train_values) & raw[label_col].isin([0, 1])
    valid_mask = raw[split_col].isin(valid_values) & raw[label_col].isin([0, 1])
    oos_mask  = raw[split_col].isin(oos_values) & raw[label_col].isin([0, 1])

    print(f"[SPLIT] train={train_mask.sum()}, valid={valid_mask.sum()}, oos={oos_mask.sum()}")

    # ── 4. Coerce features ────────────────────────────────────────
    sentinels = preproc_cfg.get("missing_sentinels", [-999, -998])
    min_nn = float(preproc_cfg.get("min_non_null_rate", 0.01))
    drop_const = bool(preproc_cfg.get("drop_constant", True))

    # Only coerce on train+valid rows for fitting purposes
    modeling_features = [f for f in candidate_features if f in raw.columns]
    x_all, kept_features, drop_detail = coerce_features(
        raw, modeling_features, sentinels, min_nn, drop_const,
    )

    # ── 5. Split data ─────────────────────────────────────────────
    tr_x = x_all[train_mask].reset_index(drop=True)
    tr_y = raw.loc[train_mask, label_col].astype(int).reset_index(drop=True)
    va_x = x_all[valid_mask].reset_index(drop=True)
    va_y = raw.loc[valid_mask, label_col].astype(int).reset_index(drop=True)

    # ── 6. Fill NaN ───────────────────────────────────────────────
    tr_x, va_x = fill_na_from_train(tr_x, va_x)
    assert isinstance(tr_x, pd.DataFrame) and isinstance(va_x, pd.DataFrame)

    # Save medians for preprocessing.json
    medians = pd.Series({
        f: float(tr_x[f].median()) for f in kept_features
    })

    # ── 7. Save preprocessing snapshot ────────────────────────────
    preprocessing = build_preprocessing_json(kept_features, drop_detail, medians, sentinels, min_nn)
    with open(output_dir / "preprocessing.json", "w", encoding="utf-8") as fh:
        json.dump(preprocessing, fh, ensure_ascii=False, indent=2)
    with open(output_dir / "candidate_feature_list.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(candidate_features) + "\n")
    with open(output_dir / "actual_feature_list.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept_features) + "\n")
    drop_detail.to_csv(output_dir / "feature_drop_detail.csv", index=False, encoding="utf-8-sig")

    # ── 8. Train LightGBM ─────────────────────────────────────────
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from scipy.stats import ks_2samp

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
    num_boost_round = lgb_cfg.get("num_boost_round", 1000)
    early_stopping = lgb_cfg.get("early_stopping_rounds", 50)
    seed = train_cfg.get("random_seed", 0)

    print(f"[TRAIN] LightGBM with {len(kept_features)} features, seed={seed}")
    t0 = time.time()

    train_ds = lgb.Dataset(tr_x, label=tr_y, feature_name=kept_features, free_raw_data=False)
    valid_ds = lgb.Dataset(va_x, label=va_y, feature_name=kept_features, reference=train_ds, free_raw_data=False)

    model = lgb.train(
        params, train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[valid_ds],
        callbacks=[
            lgb.early_stopping(early_stopping, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t0
    best_iter = model.best_iteration
    print(f"[TRAIN] Done in {train_time:.1f}s, best_iter={best_iter}")

    # ── 9. Compute metrics ────────────────────────────────────────
    tr_pred = model.predict(tr_x, num_iteration=best_iter)
    va_pred = model.predict(va_x, num_iteration=best_iter)

    tr_auc = float(roc_auc_score(tr_y, tr_pred))
    va_auc = float(roc_auc_score(va_y, va_pred))
    tr_ks  = float(ks_2samp(tr_pred[tr_y == 1], tr_pred[tr_y == 0]).statistic)
    va_ks  = float(ks_2samp(va_pred[va_y == 1], va_pred[va_y == 0]).statistic)

    metrics = {
        "train_auc": tr_auc, "valid_auc": va_auc,
        "train_ks": tr_ks,   "valid_ks": va_ks,
        "auc_gap": tr_auc - va_auc,
        "train_samples": int(len(tr_y)),
        "valid_samples": int(len(va_y)),
        "train_bad_rate": float(tr_y.mean()),
        "valid_bad_rate": float(va_y.mean()),
        "best_iteration": best_iter,
        "train_time_seconds": round(train_time, 1),
    }
    print(f"[METRICS] Train AUC={tr_auc:.4f} KS={tr_ks:.4f} | Valid AUC={va_auc:.4f} KS={va_ks:.4f} | Gap={tr_auc - va_auc:.4f}")
    with open(output_dir / "metrics_train_valid.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    # ── 10. Feature importance ────────────────────────────────────
    importance = pd.DataFrame({
        "feature": kept_features,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    print("[IMPORTANCE] Top 10:")
    for i, (_, r) in enumerate(importance.head(10).iterrows()):
        print(f"  {i+1:2d}. {r['feature']:<55s} gain={r['gain']:>10.1f} split={r['split']:>4d}")

    # ── 11. Save model ────────────────────────────────────────────
    with open(output_dir / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    # ── 12. Save run_config ───────────────────────────────────────
    run_config = {
        "experiment": "main_lgbm",
        "data_source": str(input_feather),
        "train_values": train_values,
        "valid_values": valid_values,
        "oos_values": oos_values,
        "label_column": label_col,
        "split_column": split_col,
        "feature_list_path": str(feature_list_path),
        "candidate_feature_count": len(candidate_features),
        "actual_feature_count": len(kept_features),
        "algorithm": "lightgbm",
        "params": {k: v for k, v in params.items() if k not in ("seed", "feature_fraction_seed", "bagging_seed")},
        "random_seed": seed,
        "train_samples": int(len(tr_y)),
        "valid_samples": int(len(va_y)),
        "train_bad_rate": float(tr_y.mean()),
        "valid_bad_rate": float(va_y.mean()),
        "best_iteration": best_iter,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, ensure_ascii=False, indent=2)

    # ── 13. Score ALL samples ─────────────────────────────────────
    print("[SCORE] Scoring all samples ...")
    t0 = time.time()

    # Fill NaN on all data
    x_all_filled = x_all.fillna(medians).fillna(0)
    all_pred = model.predict(
        x_all_filled[kept_features].values,
        num_iteration=best_iter,
    )

    # Build output dataframe
    # Determine which base columns are actually in the raw data
    desired_base = [
        "uid", "mdl_dte", "ds", "final_flag", "blue_customer_flag",
        "zc_level", "ftr_30d_ord_flag", "ftr_30d_ord_amt",
        "prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m",
    ]
    score_cols = ["gcard_v2", "gcard_v4", "gcard_v5", "gcard_v6"]
    available_base = [c for c in desired_base if c in raw.columns]
    available_scores = [c for c in score_cols if c in raw.columns]

    scores = raw[available_base].copy()
    for sc in available_scores:
        scores[sc] = raw[sc]
    scores["model_score"] = all_pred

    scores.reset_index(drop=True).to_feather(str(score_output))
    print(f"[SCORE] Saved {len(scores)} rows × {len(scores.columns)} cols to {score_output} in {time.time() - t0:.1f}s")

    # ── 14. Save score column summary ─────────────────────────────
    score_summary = []
    for sc in ["model_score"] + available_scores:
        if sc in scores.columns:
            sc_series = pd.to_numeric(scores[sc], errors="coerce")
            sc_nn = int(sc_series.notna().sum())
            sc_mean = float(sc_series.mean()) if sc_nn > 0 else None
            score_summary.append({
                "score_column": sc,
                "non_null_count": sc_nn,
                "null_count": int(sc_series.isna().sum()),
                "mean": sc_mean,
                "available": True,
            })
        else:
            score_summary.append({
                "score_column": sc, "non_null_count": 0, "null_count": len(scores),
                "mean": None, "available": False,
            })
    pd.DataFrame(score_summary).to_csv(
        score_output.parent / "score_column_summary.csv", index=False, encoding="utf-8-sig",
    )

    # ── 15. Generate input snapshot (modeling_input) ───────────────
    print("[INPUT] Generating input snapshot ...")

    # input_config.json
    input_config = {
        "data_source": str(input_feather),
        "data_source_size_bytes": input_feather.stat().st_size if input_feather.exists() else None,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_column": label_col,
        "label_definition": "30天是否发起(ftr_30d_ord_flag)",
        "positive_label": 1,
        "split_column": split_col,
        "train_values": train_values,
        "valid_values": valid_values,
        "oos_values": oos_values,
        "segment_column": "blue_customer_flag",
        "segment_values": {"B2": "流失户", "E2": "次新户", "E3": "老户"},
        "asset_rating_column": "zc_level",
        "feature_list_source": str(feature_list_path),
        "feature_count_in_list": len(candidate_features),
        "feature_count_in_data": len(kept_features),
        "historical_score_columns": available_scores,
        "train_window_note": "数据时间窗口: ds range from feather",
    }
    with open(input_snapshot_dir / "input_config.json", "w", encoding="utf-8") as fh:
        json.dump(input_config, fh, ensure_ascii=False, indent=2)

    # input_schema.csv
    schema_rows = []
    for col in scores.columns:
        dtype = str(scores[col].dtype)
        nn = int(scores[col].notna().sum())
        nnull = int(scores[col].isna().sum())
        schema_rows.append({"column": col, "dtype": dtype, "non_null": nn, "null": nnull})
    pd.DataFrame(schema_rows).to_csv(input_snapshot_dir / "input_schema.csv", index=False, encoding="utf-8-sig")

    # sample_split_summary.csv
    split_summary = raw.groupby(split_col).agg(
        samples=("uid", "count"),
        positive=(label_col, "sum"),
        bad_rate=(label_col, "mean"),
    ).reset_index()
    split_summary.to_csv(input_snapshot_dir / "sample_split_summary.csv", index=False, encoding="utf-8-sig")

    # label_distribution.csv
    label_dist = raw[label_col].value_counts().reset_index()
    label_dist.columns = ["label", "count"]
    label_dist["ratio"] = label_dist["count"] / label_dist["count"].sum()
    label_dist.to_csv(input_snapshot_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")

    # segment_distribution.csv
    if "blue_customer_flag" in raw.columns:
        seg_dist = raw["blue_customer_flag"].value_counts().reset_index()
        seg_dist.columns = ["segment", "count"]
        seg_dist["ratio"] = seg_dist["count"] / seg_dist["count"].sum()
        seg_dist.to_csv(input_snapshot_dir / "segment_distribution.csv", index=False, encoding="utf-8-sig")

    # score_column_summary (copy to input dir too)
    pd.DataFrame(score_summary).to_csv(
        input_snapshot_dir / "score_column_summary.csv", index=False, encoding="utf-8-sig",
    )

    print("=" * 60)
    print(f"Training complete: {output_dir}")
    print(f"  Features: {len(kept_features)}")
    print(f"  Train AUC: {tr_auc:.4f}  KS: {tr_ks:.4f}")
    print(f"  Valid AUC: {va_auc:.4f}  KS: {va_ks:.4f}")
    print(f"  Scores: {score_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
