#!/usr/bin/env python3
"""Refine wide-table features with correlation and importance filters.

This module pulls a sampled dataset from a DP wide table through TMLSQL only
when explicitly approved. It is intended for the post batch-screening
convergence stage:

1. global correlation de-duplication
2. D03 random-importance filtering
3. D04 null-importance filtering
4. D05 baseline model importance top-N selection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "agent.py").exists():
            return candidate
    raise RuntimeError("Cannot locate repo root from script path")


REPO_ROOT = find_repo_root(SCRIPT_PATH)
sys.path.insert(0, str(REPO_ROOT))

from risk_model_workbench.config import load_yaml
from risk_model_workbench.dp_feather import (
    default_dataset_paths,
    load_or_fetch_dp_feather,
    print_sql_review,
    write_dataset_metadata,
)
from risk_model_workbench.manifest import write_manifest


DEFAULT_PROJECT_DIR = Path.cwd()


@dataclass(frozen=True)
class DatasetParts:
    train_x: pd.DataFrame
    train_y: pd.Series
    valid_x: pd.DataFrame
    valid_y: pd.Series


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine wide-table features to <=500 candidates.")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR), help="Project workspace directory.")
    parser.add_argument(
        "--config",
        default="configs/refine_features.yaml",
        help="Refine config path, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--dry-run-sql",
        action="store_true",
        help="Only print and save the DP sampling SQL; do not query DP or train models.",
    )
    parser.add_argument(
        "--refresh-dp-cache",
        action="store_true",
        help="Refresh the local feather cache from DP after SQL approval.",
    )
    parser.add_argument(
        "--sql-approved",
        action="store_true",
        help="Confirm that the displayed DP SQL has been reviewed and may be executed.",
    )
    return parser.parse_args(argv)


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def load_feature_list(project_dir: Path, cfg: dict[str, Any]) -> list[str]:
    feature_map_path = resolve_project_path(project_dir, cfg["input"]["feature_map"])
    with feature_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    features = [row["output_feature"] for row in rows if row.get("output_feature")]
    base_columns = set(cfg["input"].get("base_columns", []))
    id_columns = set(cfg["input"].get("id_columns", []))
    label_column = cfg["input"]["label_column"]
    split_column = cfg["input"]["split_column"]
    exclude = base_columns | id_columns | {label_column, split_column}
    return [feature for feature in features if feature not in exclude]


def sql_identifier(name: str) -> str:
    return name if name.replace("_", "").isalnum() and not name[0].isdigit() else f"`{name}`"


def build_sampling_sql(cfg: dict[str, Any], features: list[str]) -> str:
    input_cfg = cfg["input"]
    sampling = cfg["sampling"]
    base_columns = list(dict.fromkeys(input_cfg["base_columns"]))
    select_columns = base_columns + [feature for feature in features if feature not in base_columns]
    select_expr = ",\n  ".join(sql_identifier(column) for column in select_columns)
    sql = f"select\n  {select_expr}\nfrom {input_cfg['wide_table']}"
    if sampling.get("where"):
        sql += f"\nwhere {sampling['where']}"
    if sampling.get("max_rows"):
        sql += f"\nlimit {int(sampling['max_rows'])}"
    return sql + "\n"


def coerce_feature_frame(df: pd.DataFrame, features: list[str], cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    preprocessing = cfg["preprocessing"]
    sentinels = preprocessing.get("missing_sentinels", [])
    min_non_null_rate = float(preprocessing.get("min_non_null_rate", 0.0))
    drop_constant = bool(preprocessing.get("drop_constant", True))

    available = [feature for feature in features if feature in df.columns]
    if len(available) == 0:
        sample_features = features[:5]
        sample_df_cols = list(df.columns[:10])
        print(f"[WARN] available=0: df.columns[:10]={sample_df_cols}, "
              f"first features={sample_features}", file=sys.stderr)
    else:
        base_cols = list(cfg["input"].get("base_columns", []))[:5]
        for c in base_cols:
            if c in df.columns:
                print(f"[DEBUG] base_col={c}, dtype={df[c].dtype}, "
                      f"sample={list(df[c].head(3).values)}, "
                      f"non_null={df[c].notna().mean():.4f}")
        for f in available[:3]:
            print(f"[DEBUG] feature={f}, dtype={df[f].dtype}, "
                  f"sample={list(df[f].head(3).values)}, "
                  f"non_null={df[f].notna().mean():.4f}")
    x = df.loc[:, available].copy()
    stats = []
    kept = []
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
    drop_counts = {}
    for s in stats:
        if s["drop_reason"]:
            drop_counts[s["drop_reason"]] = drop_counts.get(s["drop_reason"], 0) + 1
    if kept:
        print(f"[PREPROCESS] kept={len(kept)}/{len(available)}, drops={drop_counts}")
    else:
        sample_dropped = [s for s in stats if s["drop_reason"]][:3]
        print(f"[PREPROCESS] ALL DROPPED: {drop_counts}, samples={sample_dropped}")
    return x.loc[:, kept], kept, pd.DataFrame(stats)


def make_dataset_parts(df: pd.DataFrame, x: pd.DataFrame, cfg: dict[str, Any]) -> DatasetParts:
    input_cfg = cfg["input"]
    label = input_cfg["label_column"]
    split = input_cfg["split_column"]
    train_mask = (df[split] == input_cfg["train_value"]) & df[label].isin([0, 1])
    valid_mask = (df[split] == input_cfg["valid_value"]) & df[label].isin([0, 1])
    if not train_mask.any() or not valid_mask.any():
        raise RuntimeError("Both train and valid splits are required for importance refinement.")
    return DatasetParts(
        train_x=x.loc[train_mask].reset_index(drop=True),
        train_y=df.loc[train_mask, label].astype(int).reset_index(drop=True),
        valid_x=x.loc[valid_mask].reset_index(drop=True),
        valid_y=df.loc[valid_mask, label].astype(int).reset_index(drop=True),
    )


def fill_for_model(train_x: pd.DataFrame, valid_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = train_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    return train_x.fillna(medians).fillna(0), valid_x.fillna(medians).fillna(0)


def univariate_auc_scores(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    from sklearn.metrics import roc_auc_score

    scores = {}
    y_values = y.to_numpy()
    for feature in x.columns:
        values = x[feature].fillna(x[feature].median()).fillna(0).to_numpy()
        try:
            auc = roc_auc_score(y_values, values)
            scores[feature] = abs(float(auc) - 0.5)
        except ValueError:
            scores[feature] = 0.0
    return pd.Series(scores).sort_values(ascending=False)


def global_corr_select(train_x: pd.DataFrame, train_y: pd.Series, cfg: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["global_corr"]
    if not step_cfg.get("enabled", True):
        return list(train_x.columns), pd.DataFrame()

    threshold = float(step_cfg["threshold"])
    scores = univariate_auc_scores(train_x, train_y)
    corr = train_x.loc[:, scores.index].corr().abs().fillna(0)
    kept: list[str] = []
    dropped = []
    for feature in scores.index:
        matched = [kept_feature for kept_feature in kept if corr.loc[feature, kept_feature] >= threshold]
        if matched:
            best_match = max(matched, key=lambda item: corr.loc[feature, item])
            dropped.append(
                {
                    "feature": feature,
                    "drop_reason": "global_corr",
                    "kept_feature": best_match,
                    "corr": float(corr.loc[feature, best_match]),
                    "feature_score": float(scores[feature]),
                    "kept_score": float(scores[best_match]),
                }
            )
        else:
            kept.append(feature)
    return kept, pd.DataFrame(dropped)


def lgb_params(cfg: dict[str, Any], seed: int) -> tuple[dict[str, Any], int, int]:
    lgb_cfg = cfg["lightgbm"]
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
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "verbosity": -1,
    }
    return params, int(lgb_cfg.get("num_boost_round", 400)), int(lgb_cfg.get("early_stopping_rounds", 50))


def train_lgbm(parts: DatasetParts, features: list[str], cfg: dict[str, Any], seed: int):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    train_x, valid_x = fill_for_model(parts.train_x.loc[:, features], parts.valid_x.loc[:, features])
    params, num_boost_round, early_stopping_rounds = lgb_params(cfg, seed)
    train_set = lgb.Dataset(train_x, label=parts.train_y, feature_name=features, free_raw_data=False)
    valid_set = lgb.Dataset(valid_x, label=parts.valid_y, feature_name=features, reference=train_set, free_raw_data=False)
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)]
    model = lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=[valid_set], callbacks=callbacks)
    pred = model.predict(valid_x, num_iteration=model.best_iteration)
    auc = float(roc_auc_score(parts.valid_y, pred))
    return model, auc


def model_importance(model, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": features,
            "split": model.feature_importance(importance_type="split"),
            "gain": model.feature_importance(importance_type="gain"),
        }
    )


def d03_random_importance(parts: DatasetParts, features: list[str], cfg: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["d03_random_importance"]
    if not step_cfg.get("enabled", True):
        return features, pd.DataFrame()

    rng = np.random.default_rng(int(cfg["random_seed"]))
    random_count = int(step_cfg.get("random_feature_count", 5))
    rounds = int(step_cfg.get("rounds", 3))
    min_survival_rate = float(step_cfg.get("min_survival_rate", 0.67))
    zero_importance_drop = bool(step_cfg.get("zero_importance_drop", True))
    survival = {feature: 0 for feature in features}
    rows = []

    for round_index in range(rounds):
        random_features = [f"__random_noise_{round_index}_{idx}" for idx in range(random_count)]
        train_x = parts.train_x.loc[:, features].copy()
        valid_x = parts.valid_x.loc[:, features].copy()
        for random_feature in random_features:
            train_x[random_feature] = rng.normal(size=len(train_x))
            valid_x[random_feature] = rng.normal(size=len(valid_x))
        round_parts = DatasetParts(train_x, parts.train_y, valid_x, parts.valid_y)
        model_features = features + random_features
        model, auc = train_lgbm(round_parts, model_features, cfg, seed=int(cfg["random_seed"]) + 100 + round_index)
        importance = model_importance(model, model_features)
        random_imp = importance[importance["feature"].isin(random_features)]
        gain_threshold = float(random_imp["gain"].max())
        split_threshold = float(random_imp["split"].max())
        real_imp = importance[~importance["feature"].isin(random_features)]
        round_surv = int(((real_imp["gain"] > gain_threshold) & ((not zero_importance_drop) | (real_imp["split"] > 0))).sum())
        print(f"[D03] round={round_index} n_feat={len(features)} auc={auc:.4f} "
              f"gain_th={gain_threshold:.2f} max_real_gain={real_imp['gain'].max():.2f} "
              f"round_surv={round_surv}/{len(features)}")
        for row in importance[~importance["feature"].isin(random_features)].itertuples(index=False):
            survives = row.gain > gain_threshold and (not zero_importance_drop or row.split > 0)
            if survives:
                survival[row.feature] += 1
            rows.append(
                {
                    "round": round_index,
                    "feature": row.feature,
                    "split": float(row.split),
                    "gain": float(row.gain),
                    "random_gain_threshold": gain_threshold,
                    "random_split_threshold": split_threshold,
                    "valid_auc": auc,
                    "survives": survives,
                }
            )

    min_survival = math.ceil(rounds * min_survival_rate)
    kept = [feature for feature in features if survival[feature] >= min_survival]
    return kept, pd.DataFrame(rows)


def d04_null_importance(parts: DatasetParts, features: list[str], cfg: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    step_cfg = cfg["d04_null_importance"]
    if not step_cfg.get("enabled", True):
        return features, pd.DataFrame()

    max_features = int(step_cfg.get("max_features_for_null_importance", len(features)))
    working_features = features[:max_features]
    real_rounds = int(step_cfg.get("real_rounds", 3))
    null_rounds = int(step_cfg.get("null_rounds", 20))
    null_percentile = float(step_cfg.get("null_percentile", 75))
    score_threshold = float(step_cfg.get("score_threshold", 1.0))
    seed = int(cfg["random_seed"])
    real_gains = {feature: [] for feature in working_features}
    null_gains = {feature: [] for feature in working_features}

    for round_index in range(real_rounds):
        model, _ = train_lgbm(parts, working_features, cfg, seed=seed + 200 + round_index)
        importance = model_importance(model, working_features)
        for row in importance.itertuples(index=False):
            real_gains[row.feature].append(float(row.gain))

    rng = np.random.default_rng(seed + 300)
    for round_index in range(null_rounds):
        shuffled_parts = DatasetParts(
            parts.train_x,
            pd.Series(rng.permutation(parts.train_y.to_numpy())),
            parts.valid_x,
            parts.valid_y,
        )
        model, _ = train_lgbm(shuffled_parts, working_features, cfg, seed=seed + 300 + round_index)
        importance = model_importance(model, working_features)
        for row in importance.itertuples(index=False):
            null_gains[row.feature].append(float(row.gain))

    rows = []
    kept = []
    eps = 1e-12
    for feature in working_features:
        real_mean = float(np.mean(real_gains[feature])) if real_gains[feature] else 0.0
        null_cut = float(np.percentile(null_gains[feature], null_percentile)) if null_gains[feature] else 0.0
        score = real_mean / (null_cut + eps)
        keep = real_mean > 0 and score >= score_threshold
        if keep:
            kept.append(feature)
        rows.append(
            {
                "feature": feature,
                "real_gain_mean": real_mean,
                "null_gain_percentile": null_cut,
                "null_percentile": null_percentile,
                "null_importance_score": score,
                "survives": keep,
            }
        )
    return kept, pd.DataFrame(rows).sort_values(["survives", "null_importance_score"], ascending=[False, False])


def d05_top_importance(parts: DatasetParts, features: list[str], cfg: dict[str, Any]) -> tuple[list[str], pd.DataFrame, float]:
    step_cfg = cfg["d05_baseline_importance"]
    keep_top_n = int(step_cfg.get("keep_top_n", cfg.get("target_feature_count", 500)))
    if not step_cfg.get("enabled", True):
        return features[:keep_top_n], pd.DataFrame(), float("nan")

    model, auc = train_lgbm(parts, features, cfg, seed=int(cfg["random_seed"]) + 500)
    importance = model_importance(model, features).sort_values("gain", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    kept = importance.head(keep_top_n)["feature"].tolist()
    return kept, importance, auc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_feature_list(path: Path, features: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(features) + "\n", encoding="utf-8")


def display_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config_path = resolve_project_path(project_dir, args.config)
    cfg = load_yaml(config_path)["feature_refine"]
    output_dir = resolve_project_path(project_dir, cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_features = load_feature_list(project_dir, cfg)
    sql = build_sampling_sql(cfg, initial_features)

    dp_cache_cfg = cfg.get("dp_feather", {})
    dataset_id = dp_cache_cfg.get("dataset_id", "feature_refine_wide_sample")
    description = dp_cache_cfg.get(
        "description",
        "D01/D02 wide-table sample for D03-D05 feature refinement.",
    )
    feather_path, metadata_path = default_dataset_paths(
        project_dir,
        dataset_id=dataset_id,
        data_dir=dp_cache_cfg.get("data_dir", "data/local/dp_feather"),
        metadata_dir=dp_cache_cfg.get("metadata_dir", "data/profile/dp_feather_datasets"),
    )
    if args.dry_run_sql:
        write_dataset_metadata(
            project_dir=project_dir,
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
        return 0

    raw_df = load_or_fetch_dp_feather(
        project_dir=project_dir,
        sql=sql,
        dataset_id=dataset_id,
        description=description,
        feather_path=feather_path,
        metadata_path=metadata_path,
        refresh=args.refresh_dp_cache,
        sql_approved=args.sql_approved,
    )
    x, available_features, preprocess_stats = coerce_feature_frame(raw_df, initial_features, cfg)
    parts = make_dataset_parts(raw_df, x, cfg)
    print(f"[STAGE] raw_rows={len(raw_df)} initial_feat={len(initial_features)} "
          f"available={len(available_features)} train={len(parts.train_x)} valid={len(parts.valid_x)}")

    corr_features, corr_drops = global_corr_select(parts.train_x.loc[:, available_features], parts.train_y, cfg)
    print(f"[STAGE] after_global_corr: {len(corr_features)} (dropped {len(corr_drops)})")
    parts_corr = DatasetParts(parts.train_x.loc[:, corr_features], parts.train_y, parts.valid_x.loc[:, corr_features], parts.valid_y)
    d03_features, d03_detail = d03_random_importance(parts_corr, corr_features, cfg)
    print(f"[STAGE] after_d03: {len(d03_features)} (dropped {len(corr_features) - len(d03_features)})")
    if len(d03_features) == 0:
        print("[FATAL] D03 eliminated all features, aborting", file=sys.stderr)
        return 1
    parts_d03 = DatasetParts(parts_corr.train_x.loc[:, d03_features], parts_corr.train_y, parts_corr.valid_x.loc[:, d03_features], parts_corr.valid_y)
    d04_features, d04_detail = d04_null_importance(parts_d03, d03_features, cfg)
    parts_d04 = DatasetParts(parts_d03.train_x.loc[:, d04_features], parts_d03.train_y, parts_d03.valid_x.loc[:, d04_features], parts_d03.valid_y)
    final_features, d05_importance, d05_auc = d05_top_importance(parts_d04, d04_features, cfg)

    preprocess_stats.to_csv(output_dir / "preprocess_feature_stats.csv", index=False, encoding="utf-8-sig")
    corr_drops.to_csv(output_dir / "d00_global_corr_drops.csv", index=False, encoding="utf-8-sig")
    d03_detail.to_csv(output_dir / "d03_random_importance_detail.csv", index=False, encoding="utf-8-sig")
    d04_detail.to_csv(output_dir / "d04_null_importance_detail.csv", index=False, encoding="utf-8-sig")
    d05_importance.to_csv(output_dir / "d05_baseline_importance.csv", index=False, encoding="utf-8-sig")
    write_feature_list(output_dir / "final_500_features.txt", final_features)
    with (output_dir / "sample.pkl").open("wb") as handle:
        pickle.dump({"raw_shape": raw_df.shape, "features": final_features}, handle)
    write_json(
        output_dir / "stage_summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "wide_table": cfg["input"]["wide_table"],
            "feather_path": display_path(feather_path, project_dir),
            "raw_rows": int(len(raw_df)),
            "total_rows": int(len(raw_df)),
            "train_samples": int(len(parts.train_x)),
            "valid_samples": int(len(parts.valid_x)),
            "initial_features": len(initial_features),
            "available_features": len(available_features),
            "after_global_corr": len(corr_features),
            "after_d03_random_importance": len(d03_features),
            "after_d04_null_importance": len(d04_features),
            "final_features": len(final_features),
            "d05_valid_auc": d05_auc,
            "sampling_where": cfg["sampling"].get("where"),
        },
    )
    manifest = write_manifest(
        project_dir,
        "refine_wide_features",
        inputs=[
            config_path,
            resolve_project_path(project_dir, cfg["input"]["feature_map"]),
        ],
        outputs=[
            output_dir / "stage_summary.json",
            output_dir / "final_500_features.txt",
            output_dir / "d05_baseline_importance.csv",
        ],
    )
    print(f"output: {output_dir}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
