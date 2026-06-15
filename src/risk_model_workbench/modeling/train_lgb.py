"""LightGBM training utilities for binary business models."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any


def coerce_features(df, features: list[str], sentinels: list[int], min_non_null_rate: float, drop_constant: bool):
    """Coerce configured features to numeric and drop unusable columns."""
    import numpy as np
    import pandas as pd

    available = [feature for feature in features if feature in df.columns]
    missing = [feature for feature in features if feature not in df.columns]
    x = df.loc[:, available].copy()
    kept: list[str] = []
    stats: list[dict[str, Any]] = []

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
        stats.append(
            {
                "feature": feature,
                "non_null_rate": non_null_rate,
                "unique_count": unique_count,
                "drop_reason": drop_reason,
            }
        )

    for feature in missing:
        stats.append({"feature": feature, "non_null_rate": 0.0, "unique_count": 0, "drop_reason": "missing_from_data"})

    return x.loc[:, kept], kept, pd.DataFrame(stats)


def fill_na_from_train(train_x, valid_x, *other_frames):
    """Fill missing values with train-set medians and fallback zero."""
    medians = train_x.median(numeric_only=True).replace([float("inf"), float("-inf")], None).fillna(0)
    result = [train_x.fillna(medians).fillna(0), valid_x.fillna(medians).fillna(0)]
    for frame in other_frames:
        result.append(frame.fillna(medians).fillna(0))
    return tuple(result), medians


def train_lightgbm_from_feather(
    *,
    input_feather: str | Path,
    feature_list_path: str | Path,
    output_dir: str | Path,
    score_output: str | Path,
    input_snapshot_dir: str | Path,
    config: dict[str, Any],
    progress: Any | None = None,
) -> dict[str, Any]:
    """Train a LightGBM model and score all splits.

    Heavy dependencies are imported inside this function so basic CLI smoke tests
    do not require the full modeling stack.
    """
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp
    from sklearn.metrics import roc_auc_score

    input_feather = Path(input_feather)
    feature_list_path = Path(feature_list_path)
    output_dir = Path(output_dir)
    score_output = Path(score_output)
    input_snapshot_dir = Path(input_snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_output.parent.mkdir(parents=True, exist_ok=True)
    input_snapshot_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config["training"]
    input_cfg = config["input"]
    lgb_cfg = config["lightgbm"]
    preproc_cfg = config.get("preprocessing", {})

    candidate_features = [line.strip() for line in feature_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if progress:
        progress.emit(step="load_feature_list", message=f"训练特征列表读取完成，共 {len(candidate_features)} 个候选变量", percent=8)
    id_cols = input_cfg.get("id_columns", ["uid", "mdl_dte"])
    base_cols = input_cfg.get("base_columns", [])
    configured_historical_scores = input_cfg.get("historical_score_columns", [])
    label_col = input_cfg["label_column"]
    split_col = input_cfg["split_column"]
    read_cols = list(dict.fromkeys(id_cols + base_cols + configured_historical_scores + [label_col, split_col] + candidate_features))
    if progress:
        progress.emit(step="read_input_schema", message=f"正在读取训练数据字段：{input_feather}", percent=12)
    all_cols = pd.read_feather(input_feather, columns=None).columns.tolist()
    if progress:
        progress.emit(step="read_input_data", message=f"正在读取训练数据：{len(read_cols)} 个目标字段", percent=18)
    raw = pd.read_feather(input_feather, columns=[column for column in read_cols if column in all_cols])
    if progress:
        progress.emit(
            step="read_input_done",
            message=f"训练数据读取完成：{len(raw)} 行 {len(raw.columns)} 列",
            percent=25,
            metrics={"rows": int(len(raw)), "columns": int(len(raw.columns))},
        )

    train_values = train_cfg.get("train_values", ["DEV"])
    valid_values = train_cfg.get("valid_values", ["OOT"])
    oos_values = train_cfg.get("oos_values", ["DEV-OOS", "OOT-OOS"])
    train_mask = raw[split_col].isin(train_values) & raw[label_col].isin([0, 1])
    valid_mask = raw[split_col].isin(valid_values) & raw[label_col].isin([0, 1])

    sentinels = preproc_cfg.get("missing_sentinels", [-999, -998])
    min_non_null_rate = float(preproc_cfg.get("min_non_null_rate", 0.01))
    drop_constant = bool(preproc_cfg.get("drop_constant", True))
    x_all, kept_features, drop_detail = coerce_features(raw, candidate_features, sentinels, min_non_null_rate, drop_constant)
    if progress:
        progress.emit(
            step="preprocess_done",
            message=f"训练预处理完成：保留 {len(kept_features)}/{len(candidate_features)} 个变量",
            percent=38,
            metrics={
                "candidate_features": len(candidate_features),
                "kept_features": len(kept_features),
                "dropped_features": int((drop_detail["drop_reason"] != "").sum()),
            },
        )

    tr_x = x_all[train_mask].reset_index(drop=True)
    tr_y = raw.loc[train_mask, label_col].astype(int).reset_index(drop=True)
    va_x = x_all[valid_mask].reset_index(drop=True)
    va_y = raw.loc[valid_mask, label_col].astype(int).reset_index(drop=True)
    (tr_x, va_x), medians = fill_na_from_train(tr_x, va_x)

    preprocessing = {
        "candidate_feature_count": len(drop_detail),
        "kept_feature_count": len(kept_features),
        "dropped_feature_count": int((drop_detail["drop_reason"] != "").sum()),
        "missing_sentinels": sentinels,
        "min_non_null_rate": min_non_null_rate,
        "fill_strategy": "train_median_fill_zero",
        "drop_reason_counts": drop_detail["drop_reason"].value_counts().to_dict(),
        "fill_values": {feature: float(value) for feature, value in medians.items()},
    }
    (output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "candidate_feature_list.txt").write_text("\n".join(candidate_features) + "\n", encoding="utf-8")
    (output_dir / "actual_feature_list.txt").write_text("\n".join(kept_features) + "\n", encoding="utf-8")
    drop_detail.to_csv(output_dir / "feature_drop_detail.csv", index=False, encoding="utf-8-sig")

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
    train_ds = lgb.Dataset(tr_x, label=tr_y, feature_name=kept_features, free_raw_data=False)
    valid_ds = lgb.Dataset(va_x, label=va_y, feature_name=kept_features, reference=train_ds, free_raw_data=False)
    if progress:
        progress.emit(
            step="train_model",
            message=f"LightGBM 训练开始：训练样本 {len(tr_y)}，验证样本 {len(va_y)}，变量 {len(kept_features)} 个",
            percent=50,
            metrics={"train_samples": int(len(tr_y)), "valid_samples": int(len(va_y)), "features": len(kept_features)},
        )
    start = time.time()
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=lgb_cfg.get("num_boost_round", 1000),
        valid_sets=[valid_ds],
        callbacks=[lgb.early_stopping(lgb_cfg.get("early_stopping_rounds", 50), verbose=False), lgb.log_evaluation(period=50)],
    )
    best_iter = model.best_iteration
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
        "train_time_seconds": round(time.time() - start, 1),
    }
    metrics["auc_gap"] = metrics["train_auc"] - metrics["valid_auc"]
    if progress:
        progress.emit(
            step="train_model_done",
            message=(
                f"LightGBM 训练完成：valid_auc={metrics['valid_auc']:.4f}，"
                f"valid_ks={metrics['valid_ks']:.4f}，best_iter={metrics['best_iteration']}"
            ),
            percent=72,
            metrics=metrics,
        )
    (output_dir / "metrics_train_valid.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    importance = pd.DataFrame(
        {
            "feature": kept_features,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    explainability_cfg = config.get("explainability", {}).get("top_feature_woe", {})
    if explainability_cfg.get("enabled", True):
        from risk_model_workbench.explainability.woe import generate_top_feature_woe

        generate_top_feature_woe(
            raw,
            importance,
            output_dir=output_dir / "woe_top_features",
            label_col=label_col,
            split_col=split_col,
            top_n=int(explainability_cfg.get("top_n", 20)),
            n_bins=int(explainability_cfg.get("n_bins", 10)),
            base_split_value=explainability_cfg.get("base_split_value", "DEV"),
            missing_values=explainability_cfg.get("missing_sentinels", sentinels),
        )
    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

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
        "params": {key: value for key, value in params.items() if not key.endswith("_seed") and key != "seed"},
        "random_seed": train_cfg.get("random_seed", 0),
        **metrics,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    x_all_filled = x_all.fillna(medians).fillna(0)
    if progress:
        progress.emit(step="score_all", message="开始对全量样本打分", percent=82)
    all_pred = model.predict(x_all_filled[kept_features].values, num_iteration=best_iter)
    desired_base = [
        "uid",
        "mdl_dte",
        "ds",
        "final_flag",
        "blue_customer_flag",
        "zc_level",
        "ftr_30d_ord_flag",
        "ftr_30d_ord_amt",
        "prc_amt_xz_30d_3m",
        "ovd_amt_xz_30d_3m",
    ]
    historical_scores = [column for column in configured_historical_scores if column in raw.columns]
    scores = raw[[column for column in desired_base if column in raw.columns]].copy()
    for column in historical_scores:
        scores[column] = raw[column]
    scores["model_score"] = all_pred
    scores.reset_index(drop=True).to_feather(str(score_output))
    if progress:
        progress.emit(
            step="score_written",
            message=f"全量打分写入完成：{len(scores)} 行",
            percent=90,
            metrics={"rows": int(len(scores)), "score_output": str(score_output)},
        )

    pd.DataFrame(
        [
            {
                "score_column": column,
                "non_null_count": int(scores[column].notna().sum()),
                "null_count": int(scores[column].isna().sum()),
                "mean": float(scores[column].mean()),
                "available": True,
            }
            for column in ["model_score", *historical_scores]
        ]
    ).to_csv(score_output.parent / "score_column_summary.csv", index=False, encoding="utf-8-sig")

    _write_input_snapshot(
        raw=raw,
        scores=scores,
        input_snapshot_dir=input_snapshot_dir,
        input_feather=input_feather,
        feature_list_path=feature_list_path,
        label_col=label_col,
        split_col=split_col,
        train_values=train_values,
        valid_values=valid_values,
        oos_values=oos_values,
        candidate_feature_count=len(candidate_features),
        kept_feature_count=len(kept_features),
        historical_scores=historical_scores,
    )
    if progress:
        progress.emit(
            step="write_artifacts",
            status="done",
            message=f"训练产物写入完成：模型、指标、重要性和打分文件已生成",
            percent=100,
            metrics={"output_dir": str(output_dir), "score_output": str(score_output)},
        )
    return metrics


def _write_input_snapshot(**kwargs: Any) -> None:
    import pandas as pd

    raw = kwargs["raw"]
    scores = kwargs["scores"]
    output_dir = Path(kwargs["input_snapshot_dir"])
    label_col = kwargs["label_col"]
    split_col = kwargs["split_col"]
    input_config = {
        "data_source": str(kwargs["input_feather"]),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "label_column": label_col,
        "split_column": split_col,
        "train_values": kwargs["train_values"],
        "valid_values": kwargs["valid_values"],
        "oos_values": kwargs["oos_values"],
        "feature_list_source": str(kwargs["feature_list_path"]),
        "feature_count_in_list": kwargs["candidate_feature_count"],
        "feature_count_in_data": kwargs["kept_feature_count"],
        "historical_score_columns": kwargs["historical_scores"],
    }
    (output_dir / "input_config.json").write_text(json.dumps(input_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [{"column": col, "dtype": str(scores[col].dtype), "non_null": int(scores[col].notna().sum()), "null": int(scores[col].isna().sum())} for col in scores.columns]
    ).to_csv(output_dir / "input_schema.csv", index=False, encoding="utf-8-sig")
    raw.groupby(split_col).agg(samples=("uid", "count"), positive=(label_col, "sum"), bad_rate=(label_col, "mean")).reset_index().to_csv(
        output_dir / "sample_split_summary.csv", index=False, encoding="utf-8-sig"
    )
    label_dist = raw[label_col].value_counts().reset_index()
    label_dist.columns = ["label", "count"]
    label_dist["ratio"] = label_dist["count"] / label_dist["count"].sum()
    label_dist.to_csv(output_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")
    if "blue_customer_flag" in raw.columns:
        seg_dist = raw["blue_customer_flag"].value_counts().reset_index()
        seg_dist.columns = ["segment", "count"]
        seg_dist["ratio"] = seg_dist["count"] / seg_dist["count"].sum()
        seg_dist.to_csv(output_dir / "segment_distribution.csv", index=False, encoding="utf-8-sig")
