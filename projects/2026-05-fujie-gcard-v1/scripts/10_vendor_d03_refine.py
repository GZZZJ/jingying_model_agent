#!/usr/bin/env python3
"""按 feature-select-v2 的 D03 算法重新筛选，替代原来的「比随机噪声强」策略。

vendor D03 算法:
  1. 5 轮 bagging，每轮采样 50% 训练数据
  2. 注入一个随机列 (1-10 整数)
  3. 训练 LightGBM，三步剔除:
     - random_drop: importance < random_col 的 importance
     - zero_drop: importance == 0
     - thresholds_drop: 按 gain 降序排列，剔除累计 gain 占比 > 0.95 的尾部特征
  4. 5 轮 bagging 取剔除特征的并集（即必须在所有轮都存活）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from jingying_agent.config import load_yaml

# ============================================================================
# Vendor-style D03: select_by_importance + bagging
# ============================================================================

def select_by_importance(
    X_train, y_train, model_features, random_col,
    params_dict, num_boost_round=1000,
    thresholds=0.95, weight=1.0,
    importance_type_list=None,
):
    """单轮重要性筛选：按随机数重要性 + 累计阈值剔除。

    Returns:
        all_drop_features: dict with keys 'random', 'zero', 'thresholds'
    """
    if importance_type_list is None:
        importance_type_list = ['split', 'gain']

    dataset = lgb.Dataset(
        data=X_train[model_features + [random_col]],
        label=y_train.values,
        categorical_feature='auto',
    )

    model = lgb.train(
        params={**params_dict, 'metric': ['auc']},
        train_set=dataset,
        valid_sets=[dataset],
        valid_names=['INS'],
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    imp_df = pd.DataFrame({
        'fea': model_features + [random_col],
        'split': model.feature_importance(importance_type='split'),
        'gain': model.feature_importance(importance_type='gain'),
    })

    # 1. random_drop: importance < random_col importance * weight AND > 0
    # 2. zero_drop: importance == 0
    random_drop = []
    zero_drop = []
    for imp_type in importance_type_list:
        random_imp = imp_df[imp_df.fea == random_col][imp_type].iloc[0]
        random_drop += list(
            imp_df[(imp_df[imp_type] < random_imp * weight) & (imp_df[imp_type] > 0)].fea
        )
        zero_drop += list(imp_df[imp_df[imp_type] == 0].fea)

    random_drop = list(set(random_drop) - {random_col})
    zero_drop = list(set(zero_drop) - {random_col})

    # 3. thresholds_drop: 剔除累计重要性占比超过 thresholds 的尾部特征
    imp_df_clean = imp_df[~imp_df.fea.isin(random_drop + zero_drop + [random_col])].copy()
    thresholds_drop = []
    if thresholds is not None and len(imp_df_clean) > 0:
        for imp_type in importance_type_list:
            sorted_df = imp_df_clean.sort_values(by=imp_type, ascending=False)
            total = sorted_df[imp_type].sum()
            if total > 0:
                cumsum_ratio = sorted_df[imp_type].cumsum() / total
                thresholds_drop += list(sorted_df[cumsum_ratio > thresholds].fea)
    thresholds_drop = list(set(thresholds_drop))

    all_drop = {
        'random': random_drop,
        'zero': zero_drop,
        'thresholds': thresholds_drop,
    }
    print(f"  drop_info: random={len(random_drop)}, zero={len(zero_drop)}, "
          f"thresholds={len(thresholds_drop)}, total_dropped={len(set(random_drop + zero_drop + thresholds_drop))}")
    return all_drop


def gen_data_iter(X_train, y_train, round_num=5, bagging_fraction=0.5, random_seed=0):
    """生成 bagging 采样迭代器。"""
    rng = np.random.default_rng(random_seed)
    n_total = len(X_train)
    n_sample = max(1, int(n_total * bagging_fraction))
    for i in range(round_num):
        indices = rng.choice(n_total, size=n_sample, replace=False)
        yield X_train.iloc[indices], y_train.iloc[indices]


def vendor_d03_select(parts, model_features, cfg):
    """feature-select-v2 风格的 D03 筛选。

    Args:
        parts: DatasetParts (train_x, train_y, valid_x, valid_y)
        model_features: 输入特征列表
        cfg: refine_features.yaml 的 feature_refine 部分

    Returns:
        kept_features: 保留的特征列表
        detail: 每轮详情
    """
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
    }
    num_boost_round = 400  # lighter than vendor default 1000 for speed

    # Vendor defaults
    bagging_round = 5
    bagging_fraction = 0.5
    thresholds = 0.95
    weight = 1.0  # random importance weight
    random_seed = int(cfg.get("random_seed", 0))

    # Fill NaN with median (same as fill_for_model)
    train_x = parts.train_x[model_features].copy()
    medians = train_x.median(numeric_only=True).fillna(0)
    train_x = train_x.fillna(medians).fillna(0)
    train_y = parts.train_y

    random_col = '__vendor_random__'
    rng = np.random.default_rng(random_seed)

    data_iter = gen_data_iter(train_x, train_y, round_num=bagging_round,
                               bagging_fraction=bagging_fraction, random_seed=random_seed)

    all_drops_per_round = []
    for bag_idx, (X_sample, y_sample) in enumerate(data_iter):
        X_sample = X_sample.copy()
        X_sample[random_col] = rng.integers(1, 11, len(X_sample))

        print(f"\n[D03-vendor] bagging round {bag_idx + 1}/{bagging_round}, "
              f"n_samples={len(X_sample)}, n_features={len(model_features)}")

        drop_result = select_by_importance(
            X_train=X_sample, y_train=y_sample,
            model_features=model_features,
            random_col=random_col,
            params_dict=params,
            num_boost_round=num_boost_round,
            thresholds=thresholds,
            weight=weight,
            importance_type_list=['split', 'gain'],
        )
        all_drops_per_round.append(drop_result)

    # 取剔除特征的并集
    all_drop = set()
    for drop_result in all_drops_per_round:
        all_drop.update(drop_result['random'])
        all_drop.update(drop_result['zero'])
        all_drop.update(drop_result['thresholds'])

    kept = [f for f in model_features if f not in all_drop]

    # Build detail DataFrame
    detail_rows = []
    for bag_idx, drop_result in enumerate(all_drops_per_round):
        all_round_dropped = set(drop_result['random'] + drop_result['zero'] + drop_result['thresholds'])
        for feat in model_features:
            detail_rows.append({
                'round': bag_idx,
                'feature': feat,
                'dropped': feat in all_round_dropped,
                'drop_type': ('random' if feat in drop_result['random'] else
                               'zero' if feat in drop_result['zero'] else
                               'thresholds' if feat in drop_result['thresholds'] else
                               'survived'),
            })

    return kept, pd.DataFrame(detail_rows)


# ============================================================================
# Main: reuse 09_refine_from_feather data loading, swap D03, re-run D04+D05
# ============================================================================

def main() -> int:
    project_dir = PROJECT_DIR
    config_path = project_dir / "configs" / "refine_features.yaml"
    cfg = load_yaml(config_path)["feature_refine"]
    output_dir = project_dir / "runs/feature_refine_feather_vendor_d03"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import data loading + global_corr from 09 script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "refine_feather", project_dir / "scripts" / "09_refine_from_feather.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Import D04/D05 from 08 script (same as 09 does)
    refine_spec = importlib.util.spec_from_file_location(
        "refine_wide", project_dir / "scripts" / "08_refine_wide_features.py"
    )
    refine = importlib.util.module_from_spec(refine_spec)
    refine_spec.loader.exec_module(refine)

    DatasetParts = refine.DatasetParts

    total_time = time.time()

    # Step 0: Load feature list
    print("=" * 60)
    print("Step 0: Load feature list")
    print("=" * 60)
    initial_features = mod.load_feature_list(project_dir, cfg)
    print(f"Feature map total: {len(initial_features)} features")

    # Step 1: Load feather data (reuse 09's function)
    print("\n" + "=" * 60)
    print("Step 1: Load feather data")
    print("=" * 60)
    import re
    sampling = cfg.get("sampling", {})
    max_rows = sampling.get("max_rows", 50000)
    rand_threshold = 0.01
    match = re.search(r"rand_flag0\s*<\s*([\d.]+)", sampling.get("where", ""))
    if match:
        rand_threshold = float(match.group(1))

    x, available_features, y, split_series = mod.load_feather_sample(
        cfg, initial_features, rand_flag_threshold=rand_threshold, max_rows=max_rows
    )
    parts = mod.make_dataset_parts(x[available_features], y, split_series, cfg)
    print(f"[STAGE] rows={len(x)} available={len(available_features)} "
          f"train={len(parts.train_x)} valid={len(parts.valid_x)}")

    # Step 2: Global correlation (reuse 09's chunked implementation)
    print("\n" + "=" * 60)
    print("Step 2: Global correlation de-duplication (chunked)")
    print("=" * 60)
    t0 = time.time()
    threshold = float(cfg["global_corr"]["threshold"])
    scores = refine.univariate_auc_scores(parts.train_x, parts.train_y)
    sorted_features = scores.index.tolist()
    X_raw = parts.train_x[sorted_features].fillna(0).values.astype(np.float64)
    X_mean = X_raw.mean(axis=0)
    X_stdv = X_raw.std(axis=0, ddof=0)
    X_stdv[X_stdv == 0] = 1.0
    X_std = (X_raw - X_mean) / X_stdv
    X_norms = np.sqrt((X_std ** 2).sum(axis=0))
    X_norms[X_norms == 0] = 1.0
    X_std = X_std / X_norms

    kept_indices, drops = [], []
    for start in range(0, len(sorted_features), 500):
        end = min(start + 500, len(sorted_features))
        for feat_idx in range(start, end):
            feat_name = sorted_features[feat_idx]
            if kept_indices:
                kept_vec = X_std[:, kept_indices].T
                feat_vec = X_std[:, feat_idx]
                max_corr = np.abs(kept_vec @ feat_vec).max()
                if max_corr >= threshold:
                    best_i = kept_indices[int(np.argmax(np.abs(kept_vec @ feat_vec)))]
                    drops.append({"feature": feat_name, "drop_reason": "global_corr",
                                   "kept_feature": sorted_features[best_i], "corr": float(max_corr)})
                    continue
            kept_indices.append(feat_idx)
    corr_features = [sorted_features[i] for i in kept_indices]
    print(f"[STAGE] after_global_corr: {len(corr_features)} (dropped {len(drops)}) "
          f"time={time.time()-t0:.1f}s")

    parts_corr = DatasetParts(
        parts.train_x.loc[:, corr_features], parts.train_y,
        parts.valid_x.loc[:, corr_features], parts.valid_y,
    )

    # Step 3: Vendor-style D03
    print("\n" + "=" * 60)
    print("Step 3: D03 随机重要性筛选 (vendor-style: 累计gain 95%尾部剔除)")
    print("=" * 60)
    t0 = time.time()
    d03_features, d03_detail = vendor_d03_select(parts_corr, corr_features, cfg)
    print(f"[STAGE] after_vendor_d03: {len(d03_features)} (dropped {len(corr_features) - len(d03_features)}) "
          f"time={time.time()-t0:.1f}s")
    if len(d03_features) == 0:
        print("[FATAL] D03 eliminated all features", file=sys.stderr)
        return 1

    parts_d03 = DatasetParts(
        parts_corr.train_x.loc[:, d03_features], parts_corr.train_y,
        parts_corr.valid_x.loc[:, d03_features], parts_corr.valid_y,
    )

    # Step 4: D04 Null importance
    print("\n" + "=" * 60)
    print("Step 4: D04 Null importance filtering")
    print("=" * 60)
    t0 = time.time()
    d04_features, d04_detail = refine.d04_null_importance(parts_d03, d03_features, cfg)
    print(f"[STAGE] after_d04: {len(d04_features)} (dropped {len(d03_features) - len(d04_features)}) "
          f"time={time.time()-t0:.1f}s")

    parts_d04 = DatasetParts(
        parts_d03.train_x.loc[:, d04_features], parts_d03.train_y,
        parts_d03.valid_x.loc[:, d04_features], parts_d03.valid_y,
    )

    # Step 5: D05 Baseline importance Top-N
    print("\n" + "=" * 60)
    print("Step 5: D05 Baseline importance top-N")
    print("=" * 60)
    t0 = time.time()
    final_features, d05_importance, d05_auc = refine.d05_top_importance(parts_d04, d04_features, cfg)
    print(f"[STAGE] final_features: {len(final_features)}, D05 valid AUC={d05_auc:.4f}, "
          f"time={time.time()-t0:.1f}s")

    # Save
    print("\n" + "=" * 60)
    print("Saving outputs")
    print("=" * 60)
    d03_detail.to_csv(output_dir / "d03_vendor_random_importance_detail.csv", index=False, encoding="utf-8-sig")
    d04_detail.to_csv(output_dir / "d04_null_importance_detail.csv", index=False, encoding="utf-8-sig")
    d05_importance.to_csv(output_dir / "d05_baseline_importance.csv", index=False, encoding="utf-8-sig")
    (output_dir / "final_500_features.txt").write_text("\n".join(final_features) + "\n", encoding="utf-8")

    summary = {
        "source": "feather + vendor-style D03",
        "d03_algorithm": "feature-select-v2: 5-round bagging, random+zero+thresholds(0.95) drops, union",
        "total_rows": len(x),
        "initial_features": len(initial_features),
        "available_features": len(available_features),
        "after_global_corr": len(corr_features),
        "after_vendor_d03": len(d03_features),
        "after_d04": len(d04_features),
        "final_features": len(final_features),
        "d05_valid_auc": d05_auc,
        "train_samples": len(parts.train_x),
        "valid_samples": len(parts.valid_x),
        "total_time_seconds": time.time() - total_time,
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
