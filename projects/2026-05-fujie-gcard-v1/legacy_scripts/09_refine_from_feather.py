#!/usr/bin/env python3
"""Feature refinement (D03+) using draft feather data instead of DP wide table.

Reuses the same importance-refinement logic from jingying_agent.feature_refine,
but reads data from local feather file with rand_flag0 sampling.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from jingying_agent.config import load_yaml
from jingying_agent.feature_refine import (
    DatasetParts,
    d03_random_importance,
    d04_null_importance,
    d05_top_importance,
    fill_for_model,
    global_corr_select,
    univariate_auc_scores,
)

FEATHER_PATH = "/root/notebook/draft/十分之一观察样本.feather"
OUTPUT_DIR_NAME = "runs/feature_refine_feather"
# Features flagged for potential data leakage (e.g. future-dependent). NOT auto-dropped.
LEAKAGE_FLAGGED_FEATURES = [
    "unpaid_principal_future_light_add_heavy",
]


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def load_feature_list(project_dir: Path, cfg: dict[str, Any]) -> list[str]:
    """Load feature names from feature map CSV, excluding base/id/label/split columns."""
    feature_map_path = resolve_project_path(project_dir, cfg["input"]["feature_map"])
    with feature_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    features = [row["output_feature"] for row in rows if row.get("output_feature")]
    base_columns = set(cfg["input"].get("base_columns", []))
    id_columns = set(cfg["input"].get("id_columns", []))
    label_column = cfg["input"]["label_column"]
    split_column = cfg["input"]["split_column"]
    exclude = base_columns | id_columns | {label_column, split_column}
    return [f for f in features if f not in exclude]


def load_feather_sample(
    cfg: dict[str, Any],
    features: list[str],
    rand_flag_threshold: float = 0.02,
    max_rows: int | None = None,
):
    """Load sampled data from feather file.

    Strategy:
    1. Read base columns to determine sample mask
    2. Read feature columns in batches, filter & coerce each batch
    3. Concatenate results
    """
    input_cfg = cfg["input"]
    base_columns = list(dict.fromkeys(input_cfg["base_columns"]))
    label_col = input_cfg["label_column"]
    split_col = input_cfg["split_column"]
    preprocessing = cfg.get("preprocessing", {})
    sentinels = preprocessing.get("missing_sentinels", [-999, -998])
    min_non_null = float(preprocessing.get("min_non_null_rate", 0.01))
    drop_constant = bool(preprocessing.get("drop_constant", True))

    # Step 1: Read base columns to determine sample
    print("[FEATHER] Reading base columns for sampling...")
    t0 = time.time()
    read_cols = list(dict.fromkeys(base_columns + [label_col, split_col]))
    base = pd.read_feather(FEATHER_PATH, columns=read_cols)
    print(f"[FEATHER] base={base.shape} in {time.time()-t0:.1f}s")

    # Build sample mask
    train_mask = base[split_col].str.startswith(input_cfg["train_value"].rstrip("-OOS"))
    valid_mask = base[split_col].str.startswith(input_cfg["valid_value"].rstrip("-OOS"))
    sample_mask = (
        (base["rand_flag0"] < rand_flag_threshold)
        & base[label_col].isin([0, 1])
        & (train_mask | valid_mask)
    )
    n_sample = sample_mask.sum()
    print(f"[FEATHER] rand_flag0 < {rand_flag_threshold}: {n_sample}/{len(base)} rows")

    if n_sample == 0:
        raise RuntimeError(f"No rows match criteria (rand_flag0 < {rand_flag_threshold})")

    # Extract y and split for return
    y_series = base.loc[sample_mask, label_col].copy()
    split_series = base.loc[sample_mask, split_col].copy()
    sample_idx_full = base.index[sample_mask].tolist()

    if max_rows and len(sample_idx_full) > max_rows:
        rng = np.random.default_rng(42)
        keep_positions = sorted(rng.choice(len(sample_idx_full), size=max_rows, replace=False))
        sample_idx = [sample_idx_full[i] for i in keep_positions]
        # Also filter y and split to match
        y_series = y_series.iloc[keep_positions]
        split_series = split_series.iloc[keep_positions]
        print(f"[FEATHER] Subsample from {len(sample_idx_full)} to {max_rows} rows")
    else:
        sample_idx = sample_idx_full

    # Step 2: Find available features in feather
    all_columns = pd.read_feather(FEATHER_PATH, columns=None).columns
    available_features = [f for f in features if f in all_columns]
    missing = len(features) - len(available_features)
    if missing:
        print(f"[FEATHER] {missing}/{len(features)} features not in feather")

    # Step 3: Read features in batches, coerce, and keep
    BATCH_SIZE = 400
    n_batches = (len(available_features) + BATCH_SIZE - 1) // BATCH_SIZE
    kept_features = []
    drop_counts = {}
    x_parts = []

    print(f"[FEATHER] Reading {len(available_features)} features in {n_batches} batches...")
    t_total = time.time()
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(available_features))
        batch_features = available_features[start:end]
        batch_cols = list(dict.fromkeys(base_columns + [label_col, split_col] + batch_features))

        t0 = time.time()
        df_batch = pd.read_feather(FEATHER_PATH, columns=batch_cols)
        # Filter to sampled rows
        df_batch = df_batch.iloc[sample_idx].copy()
        load_t = time.time() - t0

        t0 = time.time()
        for feat in batch_features:
            s = pd.to_numeric(df_batch[feat], errors="coerce")
            if sentinels:
                s = s.replace(sentinels, np.nan)
            s = s.replace([np.inf, -np.inf], np.nan)
            nn_rate = s.notna().mean()
            nuniq = s.nunique(dropna=True)
            if nn_rate < min_non_null:
                drop_counts["low_non_null_rate"] = drop_counts.get("low_non_null_rate", 0) + 1
            elif drop_constant and nuniq <= 1:
                drop_counts["constant"] = drop_counts.get("constant", 0) + 1
            else:
                kept_features.append(feat)
                x_parts.append(pd.Series(s.values.astype(np.float32), name=feat))
        coerce_t = time.time() - t0
        print(f"  batch {batch_idx+1}/{n_batches}: {len(batch_features)} cols, "
              f"load={load_t:.1f}s coerce={coerce_t:.1f}s, kept_so_far={len(kept_features)}")

    print(f"[FEATHER] Total load+coerce time: {time.time()-t_total:.1f}s")
    print(f"[COERCE] Kept {len(kept_features)}/{len(available_features)}, drops={drop_counts}")

    # Build feature DataFrame
    x = pd.concat(x_parts, axis=1)
    return x, kept_features, y_series, split_series


def make_dataset_parts(x: pd.DataFrame, y: pd.Series, split_series: pd.Series, cfg: dict[str, Any]):
    """Adapted split logic for feather data which has DEV-OOS, OOT-OOS etc."""
    train_val = cfg["input"]["train_value"]
    valid_val = cfg["input"]["valid_value"]

    # Use numpy arrays to avoid index alignment issues
    split_arr = split_series.values.astype(str)
    y_arr = y.values.astype(int)
    train_mask = np.array([s.startswith(train_val.rstrip("-OOS"))
                           for s in split_arr]) & np.isin(y_arr, [0, 1])
    valid_mask = np.array([s.startswith(valid_val.rstrip("-OOS"))
                           for s in split_arr]) & np.isin(y_arr, [0, 1])

    if not train_mask.any() or not valid_mask.any():
        raise RuntimeError(f"No train/valid samples: train={train_mask.sum()}, valid={valid_mask.sum()}")

    return DatasetParts(
        train_x=x.iloc[train_mask].reset_index(drop=True),
        train_y=pd.Series(y_arr[train_mask]).reset_index(drop=True),
        valid_x=x.iloc[valid_mask].reset_index(drop=True),
        valid_y=pd.Series(y_arr[valid_mask]).reset_index(drop=True),
    )


def main() -> int:
    project_dir = PROJECT_DIR
    config_path = project_dir / "configs" / "refine_features.yaml"
    cfg = load_yaml(config_path)["feature_refine"]
    output_dir = project_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load feature list from feature map
    print("=" * 60)
    print("Step 0: Load feature list from feature map")
    print("=" * 60)
    initial_features = load_feature_list(project_dir, cfg)
    print(f"Feature map total: {len(initial_features)} features")

    # Load feather data with sampling
    print("\n" + "=" * 60)
    print("Step 1: Load feather data")
    print("=" * 60)
    total_time = time.time()
    sampling = cfg.get("sampling", {})
    max_rows = sampling.get("max_rows", 50000)
    import re
    rand_threshold = 0.01
    where_clause = sampling.get("where", "")
    match = re.search(r"rand_flag0\s*<\s*([\d.]+)", where_clause)
    if match:
        rand_threshold = float(match.group(1))

    x, available_features, y, split_series = load_feather_sample(
        cfg, initial_features, rand_flag_threshold=rand_threshold, max_rows=max_rows
    )
    parts = make_dataset_parts(x[available_features], y, split_series, cfg)

    print(f"[STAGE] raw_rows={len(x)} initial_feat={len(initial_features)} "
          f"available={len(available_features)} train={len(parts.train_x)} valid={len(parts.valid_x)}")

    # Global correlation de-duplication (CHUNKED - avoids full NxN matrix)
    print("\n" + "=" * 60)
    print("Step 2: Global correlation de-duplication (chunked)")
    print("=" * 60)
    t0 = time.time()
    threshold = float(cfg["global_corr"]["threshold"])
    if not cfg["global_corr"].get("enabled", True):
        corr_features, corr_drops = list(parts.train_x.columns), pd.DataFrame()
    else:
        # 1. Univariate AUC sort
        scores = univariate_auc_scores(parts.train_x, parts.train_y)
        sorted_features = scores.index.tolist()
        print(f"[GLOBAL_CORR] Univariate AUC done, {len(sorted_features)} features, {time.time()-t0:.1f}s")

        # 2. Pre-standardize all features (z-score) for fast dot-product correlation
        X_raw = parts.train_x[sorted_features].fillna(0).values.astype(np.float64)
        X_mean = X_raw.mean(axis=0)
        X_std = X_raw.std(axis=0, ddof=0)
        X_std[X_std == 0] = 1.0
        X_std = (X_raw - X_mean) / X_std  # (n_samples, n_features)
        # Normalize to unit vectors for cosine similarity (= correlation for centered data)
        X_norms = np.sqrt((X_std ** 2).sum(axis=0))
        X_norms[X_norms == 0] = 1.0
        X_std = X_std / X_norms
        n = X_std.shape[0]
        print(f"[GLOBAL_CORR] Standardized {X_std.shape}, {time.time()-t0:.1f}s")

        # 3. Chunked greedy: dot product X_std[:, feat_idx] @ X_std[:, kept_indices]
        kept_indices = []
        drops = []
        CHUNK = 500
        for start in range(0, len(sorted_features), CHUNK):
            end = min(start + CHUNK, len(sorted_features))
            chunk_indices = list(range(start, end))
            for feat_i, feat_idx in enumerate(chunk_indices):
                feat_name = sorted_features[feat_idx]
                if kept_indices:
                    # Correlations = X_std[:, feat_idx] @ X_std[:, kept_indices]
                    # Shape: (1, n_kept) → use (n_kept,) @ (n_kept, n)
                    kept_vec = X_std[:, kept_indices].T  # (n_kept, n)
                    feat_vec = X_std[:, feat_idx]          # (n,)
                    corrs = np.abs(kept_vec @ feat_vec)     # (n_kept,)
                    max_corr_idx = np.argmax(corrs)
                    max_corr = corrs[max_corr_idx]
                    if max_corr >= threshold:
                        best_feat = sorted_features[kept_indices[max_corr_idx]]
                        drops.append({
                            "feature": feat_name, "drop_reason": "global_corr",
                            "kept_feature": best_feat, "corr": float(max_corr),
                            "feature_score": float(scores[feat_name]),
                            "kept_score": float(scores[best_feat]),
                        })
                        continue
                kept_indices.append(feat_idx)
            print(f"  chunk {start//CHUNK+1}: {start+1}-{end}/{len(sorted_features)}, "
                  f"kept={len(kept_indices)}, {time.time()-t0:.1f}s")

        corr_features = [sorted_features[i] for i in kept_indices]
        corr_drops = pd.DataFrame(drops)
    print(f"[STAGE] after_global_corr: {len(corr_features)} (dropped {len(corr_drops)}) "
          f"time={time.time()-t0:.1f}s")
    parts_corr = DatasetParts(
        parts.train_x.loc[:, corr_features], parts.train_y,
        parts.valid_x.loc[:, corr_features], parts.valid_y,
    )

    # D03: Random importance
    print("\n" + "=" * 60)
    print("Step 3: D03 Random importance filtering")
    print("=" * 60)
    t0 = time.time()
    d03_features, d03_detail = d03_random_importance(parts_corr, corr_features, cfg)
    print(f"[STAGE] after_d03: {len(d03_features)} (dropped {len(corr_features) - len(d03_features)}) "
          f"time={time.time()-t0:.1f}s")
    if len(d03_features) == 0:
        print("[FATAL] D03 eliminated all features, aborting", file=sys.stderr)
        return 1
    parts_d03 = DatasetParts(
        parts_corr.train_x.loc[:, d03_features], parts_corr.train_y,
        parts_corr.valid_x.loc[:, d03_features], parts_corr.valid_y,
    )

    # D04: Null importance
    print("\n" + "=" * 60)
    print("Step 4: D04 Null importance filtering")
    print("=" * 60)
    t0 = time.time()
    d04_features, d04_detail = d04_null_importance(parts_d03, d03_features, cfg)
    d04_passed = d04_detail[d04_detail["survives"] == True] if len(d04_detail) > 0 else pd.DataFrame()
    print(f"[STAGE] after_d04: {len(d04_features)} (dropped {len(d03_features) - len(d04_features)}) "
          f"time={time.time()-t0:.1f}s")
    parts_d04 = DatasetParts(
        parts_d03.train_x.loc[:, d04_features], parts_d03.train_y,
        parts_d03.valid_x.loc[:, d04_features], parts_d03.valid_y,
    )

    # D05: Baseline importance top-N
    print("\n" + "=" * 60)
    print("Step 5: D05 Baseline importance top-N")
    print("=" * 60)
    t0 = time.time()
    final_features, d05_importance, d05_auc = d05_top_importance(parts_d04, d04_features, cfg)
    print(f"[STAGE] final_features: {len(final_features)}, D05 valid AUC={d05_auc:.4f}, "
          f"time={time.time()-t0:.1f}s")

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving outputs")
    print("=" * 60)
    corr_drops.to_csv(output_dir / "d00_global_corr_drops.csv", index=False, encoding="utf-8-sig")
    d03_detail.to_csv(output_dir / "d03_random_importance_detail.csv", index=False, encoding="utf-8-sig")
    d04_detail.to_csv(output_dir / "d04_null_importance_detail.csv", index=False, encoding="utf-8-sig")
    d05_importance.to_csv(output_dir / "d05_baseline_importance.csv", index=False, encoding="utf-8-sig")
    (output_dir / "final_500_features.txt").write_text("\n".join(final_features) + "\n", encoding="utf-8")

    # Write leakage-flagged final features (with annotations) for modeling input
    leakage_in_final = [f for f in LEAKAGE_FLAGGED_FEATURES if f in final_features]
    final_txt_lines = [f"# LEAKAGE-WARN: {f} - contains indicators that may depend on future information; review before production use" for f in leakage_in_final]
    final_txt_lines += final_features
    (output_dir / "final_features.txt").write_text("\n".join(final_txt_lines) + "\n", encoding="utf-8")

    # Summary
    summary = {
        "source": "feather",
        "feather_path": FEATHER_PATH,
        "total_rows": len(x),
        "initial_features": len(initial_features),
        "available_features": len(available_features),
        "after_global_corr": len(corr_features),
        "after_d03_random_importance": len(d03_features),
        "after_d04_null_importance": len(d04_features),
        "final_features": len(final_features),
        "d05_valid_auc": d05_auc,
        "rand_flag_threshold": rand_threshold,
        "train_samples": len(parts.train_x),
        "valid_samples": len(parts.valid_x),
        "config": str(config_path),
        "leakage_flagged_features": LEAKAGE_FLAGGED_FEATURES,
        "leakage_flagged_in_final": leakage_in_final,
    }
    with (output_dir / "stage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nOutput dir: {output_dir}")
    print(f"Final features: {len(final_features)}")
    print(f"D05 valid AUC: {d05_auc:.4f}")
    print(f"Total time: {time.time() - total_time:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
