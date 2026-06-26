"""End-to-end score evaluation from a prediction table."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from risk_model_workbench.evaluation.decile_lift import compute_decile_lift
from risk_model_workbench.evaluation.metrics import auc_score, ks_score
from risk_model_workbench.evaluation.stability import compute_score_psi


def evaluate_scores_from_feather(
    *,
    scores_feather: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    progress: Any | None = None,
) -> dict[str, Any]:
    """Evaluate model and champion scores from a feather score table."""
    scores_path = Path(scores_feather)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eval_cfg = config["evaluation"]
    if progress:
        progress.emit(step="read_scores", message=f"正在读取评分数据：{scores_path}", percent=8)
    df = pd.read_feather(scores_path)
    if progress:
        progress.emit(
            step="read_scores_done",
            message=f"评分数据读取完成：{len(df)} 行 {len(df.columns)} 列",
            percent=18,
            metrics={"rows": int(len(df)), "columns": int(len(df.columns))},
        )

    label_col = eval_cfg["label_column"]
    split_col = eval_cfg["split_column"]
    time_col = eval_cfg.get("time_column", "mdl_dte")
    requested_metrics = config.get("metrics") or eval_cfg.get("metrics") or []
    missing_requirements: list[dict[str, Any]] = []
    for required in [label_col, split_col]:
        if required not in df.columns:
            raise ValueError(f"required evaluation column missing: {required}")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    for column in ["prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    desired_scores = eval_cfg.get("score_columns", ["model_score"])
    score_cols = [column for column in desired_scores if column in df.columns]
    if not score_cols:
        raise ValueError(f"none of the configured score columns are present: {desired_scores}")
    missing_scores = [column for column in desired_scores if column not in df.columns]
    for column in missing_scores:
        missing_requirements.append({"kind": "score_column", "column": column, "reason": "missing from score table"})
    for column in score_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if time_col in df.columns:
        df["mdl_month"] = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)

    splits = sorted(df[split_col].dropna().unique().tolist())
    overall_rows: list[dict[str, Any]] = []
    if progress:
        progress.emit(step="overall_metrics", message=f"开始计算整体指标：{len(splits)} 个样本分组，{len(score_cols)} 个分数字段", percent=25)
    for split_val in splits:
        mask = df[split_col] == split_val
        row = _base_slice_row(df, mask, label_col, split_col, split_val)
        _add_score_metrics(row, df, mask, label_col, score_cols)
        overall_rows.append(row)
    pd.DataFrame(overall_rows).to_csv(output_path / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    if progress:
        progress.emit(step="overall_metrics_done", message=f"整体指标完成：{len(overall_rows)} 行", percent=38, metrics={"rows": len(overall_rows)})

    monthly_rows: list[dict[str, Any]] = []
    if progress:
        progress.emit(step="monthly_metrics", message="开始计算月度指标", percent=42)
    if "mdl_month" in df.columns:
        for split_val in splits:
            split_mask = df[split_col] == split_val
            for month in sorted(df.loc[split_mask, "mdl_month"].dropna().unique()):
                mask = split_mask & (df["mdl_month"] == month)
                if int(mask.sum()) < 50:
                    continue
                row = _base_slice_row(df, mask, label_col, "mdl_month", month)
                row[split_col] = split_val
                _add_score_metrics(row, df, mask, label_col, score_cols)
                monthly_rows.append(row)
    pd.DataFrame(monthly_rows).to_csv(output_path / "monthly_metrics.csv", index=False, encoding="utf-8-sig")
    if progress:
        progress.emit(step="monthly_metrics_done", message=f"月度指标完成：{len(monthly_rows)} 行", percent=52, metrics={"rows": len(monthly_rows)})

    if progress:
        progress.emit(step="segment_metrics", message="开始计算分群指标", percent=56)
    segment_rows = _segment_metrics(df, label_col, split_col, score_cols, splits)
    pd.DataFrame(segment_rows).to_csv(output_path / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    dimension_columns = list(
        dict.fromkeys(
            [
                *[column for column in eval_cfg.get("segment_columns", []) if column],
                *[column for column in eval_cfg.get("comparison_dimensions", []) if column],
                *[column for column in eval_cfg.get("risk_profile_dimensions", []) if column],
            ]
        )
    )
    dimension_rows, missing_dimension_rows = _generic_dimension_metrics(df, dimension_columns, label_col, split_col, score_cols, splits)
    missing_requirements.extend(missing_dimension_rows)
    pd.DataFrame(dimension_rows).to_csv(output_path / "dimension_metrics.csv", index=False, encoding="utf-8-sig")
    if progress:
        progress.emit(step="segment_metrics_done", message=f"分群指标完成：{len(segment_rows)} 行", percent=64, metrics={"rows": len(segment_rows)})

    if progress:
        progress.emit(step="decile_outputs", message="开始生成 decile lift 结果", percent=68)
    _write_decile_outputs(df, output_path, score_cols, label_col)
    ranking_rows = _ranking_inversion_rows(output_path, score_cols)
    pd.DataFrame(ranking_rows).to_csv(output_path / "ranking_inversion.csv", index=False, encoding="utf-8-sig")
    _write_cross_gain_outputs(df, output_path, score_cols, label_col)
    if progress:
        progress.emit(step="intent_risk_outputs", message="开始生成意愿资产交叉观察结果", percent=76)
    intent_segments: dict[str, Any] = {}
    if "blue_customer_flag" in df.columns:
        intent_segments = {
            "e2e3": df["blue_customer_flag"].isin(["E2", "E3"]),
            "b2": df["blue_customer_flag"] == "B2",
        }
    _write_intent_risk_outputs(df, output_path, label_col, segment_filters=intent_segments)
    business_risk_rows = _business_risk_rows(df, label_col, score_cols, eval_cfg.get("risk_profile_dimensions", []))
    pd.DataFrame(business_risk_rows).to_csv(output_path / "business_risk.csv", index=False, encoding="utf-8-sig")
    if progress:
        progress.emit(step="psi_outputs", message="开始生成 PSI 稳定性结果", percent=82)
    _write_psi_outputs(df, output_path, score_cols, time_col)
    benchmark_rows = _benchmark_uplift(df, label_col, split_col, score_cols, splits)
    pd.DataFrame(benchmark_rows).to_csv(output_path / "benchmark_uplift.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(_feature_gain_rows(benchmark_rows)).to_csv(output_path / "feature_gain_summary.csv", index=False, encoding="utf-8-sig")
    (output_path / "missing_evaluation_requirements.json").write_text(
        json.dumps({"missing": missing_requirements}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress.emit(step="benchmark_done", message=f"冠军挑战者对比指标完成：{len(benchmark_rows)} 行", percent=90, metrics={"rows": len(benchmark_rows)})

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scores_source": str(scores_path),
        "score_columns_evaluated": score_cols,
        "requested_metrics": requested_metrics,
        "label_column": label_col,
        "split_column": split_col,
        "splits_evaluated": splits,
        "n_total_samples": len(df),
        "n_score_columns": len(score_cols),
        "missing_requirements": missing_requirements,
        "dimension_columns_evaluated": [column for column in dimension_columns if column in df.columns],
        "overall_metrics": {
            score: {
                split_val: {
                    "auc": next((row.get(f"{score}_auc") for row in overall_rows if row[split_col] == split_val), None),
                    "ks": next((row.get(f"{score}_ks") for row in overall_rows if row[split_col] == split_val), None),
                }
                for split_val in splits
            }
            for score in score_cols
        },
        "output_dir": str(output_path),
    }
    (output_path / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if progress:
        progress.emit(
            step="write_summary",
            status="done",
            message=f"模型评估完成：评估 {len(score_cols)} 个分数字段，总样本 {len(df)} 行",
            percent=100,
            metrics={"score_columns": score_cols, "rows": int(len(df)), "output_dir": str(output_path)},
        )
    return summary


def _base_slice_row(df: pd.DataFrame, mask, label_col: str, key_name: str, key_value: Any) -> dict[str, Any]:
    n = int(mask.sum())
    positive = int(df.loc[mask, label_col].sum())
    return {key_name: key_value, "n_samples": n, "positive": positive, "bad_rate": round(positive / n, 6) if n else 0}


def _add_score_metrics(row: dict[str, Any], df: pd.DataFrame, mask, label_col: str, score_cols: list[str]) -> None:
    for score_col in score_cols:
        row[f"{score_col}_auc"] = auc_score(df.loc[mask, label_col], df.loc[mask, score_col])
        row[f"{score_col}_ks"] = ks_score(df.loc[mask, label_col], df.loc[mask, score_col])


def _segment_metrics(df: pd.DataFrame, label_col: str, split_col: str, score_cols: list[str], splits: list[Any]) -> list[dict[str, Any]]:
    segment_defs = {
        "全客群": None,
        "老户次新": "blue_customer_flag in ['E2', 'E3']",
        "老户": "blue_customer_flag == 'E3'",
        "次新": "blue_customer_flag == 'E2'",
        "流失户": "blue_customer_flag == 'B2'",
    }
    rows: list[dict[str, Any]] = []
    for segment, expression in segment_defs.items():
        if expression is None:
            segment_mask = pd.Series(True, index=df.index)
        elif "blue_customer_flag" not in df.columns:
            continue
        else:
            segment_mask = df.eval(expression)
        for split_val in splits:
            mask = segment_mask & (df[split_col] == split_val)
            if int(mask.sum()) < 50:
                continue
            row = _base_slice_row(df, mask, label_col, split_col, split_val)
            row["segment"] = segment
            _add_score_metrics(row, df, mask, label_col, score_cols)
            if row.get("model_score_ks") is not None:
                for score_col in score_cols:
                    ref = row.get(f"{score_col}_ks")
                    if score_col != "model_score" and ref is not None:
                        row[f"ks_uplift_vs_{score_col}"] = round(row["model_score_ks"] - ref, 6)
            rows.append(row)
    return rows


def _write_decile_outputs(df: pd.DataFrame, output_path: Path, score_cols: list[str], label_col: str) -> None:
    for score_col in score_cols:
        dec_all = compute_decile_lift(df, score_col, label_col)
        if len(dec_all):
            dec_all.to_csv(output_path / f"decile_lift_all_{score_col}.csv", index=False, encoding="utf-8-sig")
            if score_col == "model_score":
                dec_all.to_csv(output_path / "decile_lift_all.csv", index=False, encoding="utf-8-sig")
        if "blue_customer_flag" not in df.columns:
            continue
        for segment_name, segment_mask in {
            "e2e3": df["blue_customer_flag"].isin(["E2", "E3"]),
            "b2": df["blue_customer_flag"] == "B2",
        }.items():
            decile = compute_decile_lift(df[segment_mask], score_col, label_col)
            if len(decile):
                decile.to_csv(output_path / f"decile_lift_{segment_name}_{score_col}.csv", index=False, encoding="utf-8-sig")
                if score_col == "model_score":
                    decile.to_csv(output_path / f"decile_lift_{segment_name}.csv", index=False, encoding="utf-8-sig")


def _generic_dimension_metrics(
    df: pd.DataFrame,
    dimensions: list[str],
    label_col: str,
    split_col: str,
    score_cols: list[str],
    splits: list[Any],
    *,
    min_samples: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for dimension in dimensions:
        if dimension not in df.columns:
            missing.append({"kind": "dimension", "column": dimension, "reason": "missing from score table"})
            continue
        for value in sorted(df[dimension].dropna().astype(str).unique()):
            dimension_mask = df[dimension].astype(str) == value
            for split_val in splits:
                mask = dimension_mask & (df[split_col] == split_val)
                if int(mask.sum()) < min_samples:
                    continue
                row = _base_slice_row(df, mask, label_col, split_col, split_val)
                row["dimension"] = dimension
                row["dimension_value"] = value
                _add_score_metrics(row, df, mask, label_col, score_cols)
                rows.append(row)
    return rows, missing


def _ranking_inversion_rows(output_path: Path, score_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        path = output_path / f"decile_lift_all_{score_col}.csv"
        if not path.exists():
            continue
        decile = pd.read_csv(path)
        if "bad_rate" not in decile.columns:
            continue
        rates = pd.to_numeric(decile["bad_rate"], errors="coerce").dropna().tolist()
        inversions = sum(1 for left, right in zip(rates, rates[1:]) if right < left)
        rows.append(
            {
                "score_column": score_col,
                "decile_count": len(rates),
                "inversion_count": int(inversions),
                "is_monotonic_non_decreasing": inversions == 0,
            }
        )
    return rows


def _write_cross_gain_outputs(df: pd.DataFrame, output_path: Path, score_cols: list[str], label_col: str, bins: int = 10) -> None:
    if "model_score" not in score_cols:
        return
    valid_model = df["model_score"].notna()
    if int(valid_model.sum()) < 20:
        return
    model_bins = pd.qcut(df.loc[valid_model, "model_score"], bins, labels=False, duplicates="drop")
    for champion in [score for score in score_cols if score != "model_score"]:
        valid = valid_model & df[champion].notna()
        if int(valid.sum()) < 20:
            continue
        temp = df.loc[valid, [label_col, champion]].copy()
        temp["model_bin"] = pd.qcut(df.loc[valid, "model_score"], bins, labels=False, duplicates="drop")
        temp["champion_bin"] = pd.qcut(temp[champion], bins, labels=False, duplicates="drop")
        rows = []
        for (model_bin, champion_bin), group in temp.groupby(["model_bin", "champion_bin"], dropna=True):
            rows.append(
                {
                    "model_bin": int(model_bin) + 1,
                    "champion": champion,
                    "champion_bin": int(champion_bin) + 1,
                    "n_samples": int(len(group)),
                    "bad": int(group[label_col].sum()),
                    "bad_rate": float(group[label_col].mean()) if len(group) else 0.0,
                }
            )
        if rows:
            pd.DataFrame(rows).to_csv(output_path / f"cross_gain_matrix_{champion}.csv", index=False, encoding="utf-8-sig")


def _write_intent_risk_outputs(
    df: pd.DataFrame,
    output_path: Path,
    label_col: str,
    segment_filters: dict[str, Any] | None = None,
) -> None:
    """Write intent x zc_level risk matrices for the cohort and each segment.

    Always writes the cohort files (``intent_zc_*.csv``). When ``segment_filters``
    is provided (e.g. {"e2e3": mask, "b2": mask}) it also writes per-segment files
    (``intent_zc_*_<segment>.csv``) so segmented intent matrices are produced, not
    flagged as missing by the report.
    """
    if "zc_level" not in df.columns or "model_score" not in df.columns:
        return
    _write_intent_tables(_compute_intent_risk_tables(df, label_col), output_path, suffix="")
    if segment_filters:
        for seg_name, seg_mask in segment_filters.items():
            seg_df = df[seg_mask]
            if int(seg_df["model_score"].notna().sum()) < 30:
                continue
            _write_intent_tables(_compute_intent_risk_tables(seg_df, label_col), output_path, suffix=f"_{seg_name}")


def _compute_intent_risk_tables(df_slice: pd.DataFrame, label_col: str) -> dict[str, pd.DataFrame]:
    """Bin model_score into intent levels and build the four intent x zc tables."""
    empty = {
        "distribution": pd.DataFrame(),
        "ftr_rate": pd.DataFrame(),
        "amount_risk": pd.DataFrame(),
        "headcount": pd.DataFrame(),
    }
    valid = df_slice["model_score"].notna()
    if int(valid.sum()) < 30:
        return empty
    bins = pd.qcut(df_slice.loc[valid, "model_score"], 3, labels=False, duplicates="drop")
    frame = df_slice.copy()
    frame["intent_level"] = "中意愿"
    if bins.nunique() == 3:
        frame.loc[valid, "intent_level"] = ["低意愿" if b == 0 else "中意愿" if b == 1 else "高意愿" for b in bins]
    elif bins.nunique() == 2:
        frame.loc[valid, "intent_level"] = ["低意愿" if b == 0 else "高意愿" for b in bins]
    total = max(len(frame), 1)
    rows = []
    for intent in ["低意愿", "中意愿", "高意愿"]:
        for zc_val in sorted(frame["zc_level"].dropna().unique()):
            mask = (frame["intent_level"] == intent) & (frame["zc_level"] == zc_val)
            n = int(mask.sum())
            if n:
                positive = int(frame.loc[mask, label_col].sum())
                rows.append({"intent_level": intent, "zc_level": str(zc_val), "n_samples": n, "pct": round(n / total, 6), "bad": positive, "bad_rate": round(positive / n, 6)})
    distribution = pd.DataFrame(rows)
    ftr_rate = (
        distribution.pivot_table(index="intent_level", columns="zc_level", values="bad_rate", aggfunc="first")
        if len(distribution)
        else pd.DataFrame()
    )
    amount_risk = pd.DataFrame()
    headcount = pd.DataFrame()
    if "prc_amt_xz_30d_3m" in frame.columns and "ovd_amt_xz_30d_3m" in frame.columns:
        risk_rows = []
        head_rows = []
        for intent in ["低意愿", "中意愿", "高意愿"]:
            mask = frame["intent_level"] == intent
            n = int(mask.sum())
            if n:
                principal = float(frame.loc[mask, "prc_amt_xz_30d_3m"].fillna(0).sum())
                overdue = float(frame.loc[mask, "ovd_amt_xz_30d_3m"].fillna(0).sum())
                head_risk = int((frame.loc[mask, "ovd_amt_xz_30d_3m"].fillna(0) > 0).sum())
                risk_rows.append({"intent_level": intent, "n_samples": n, "total_principal": principal, "total_overdue": overdue, "amount_overdue_rate": overdue / principal if principal > 0 else 0, "head_risk_count": head_risk, "head_risk_rate": round(head_risk / n, 6)})
            for zc_val in sorted(frame["zc_level"].dropna().unique()):
                zc_mask = mask & (frame["zc_level"] == zc_val)
                zc_n = int(zc_mask.sum())
                if zc_n:
                    head_risk = int((frame.loc[zc_mask, "ovd_amt_xz_30d_3m"].fillna(0) > 0).sum())
                    head_rows.append({"intent_level": intent, "zc_level": str(zc_val), "n_samples": zc_n, "head_risk_count": head_risk, "head_risk_rate": round(head_risk / zc_n, 6)})
        amount_risk = pd.DataFrame(risk_rows)
        headcount = pd.DataFrame(head_rows)
    return {"distribution": distribution, "ftr_rate": ftr_rate, "amount_risk": amount_risk, "headcount": headcount}


def _write_intent_tables(tables: dict[str, pd.DataFrame], output_path: Path, suffix: str) -> None:
    if len(tables["distribution"]):
        tables["distribution"].to_csv(output_path / f"intent_zc_distribution{suffix}.csv", index=False, encoding="utf-8-sig")
        tables["ftr_rate"].to_csv(output_path / f"intent_zc_ftr_rate{suffix}.csv", encoding="utf-8-sig")
    if len(tables["amount_risk"]):
        tables["amount_risk"].to_csv(output_path / f"intent_zc_amount_risk{suffix}.csv", index=False, encoding="utf-8-sig")
    if len(tables["headcount"]):
        tables["headcount"].to_csv(output_path / f"intent_zc_headcount_risk{suffix}.csv", index=False, encoding="utf-8-sig")


def _business_risk_rows(df: pd.DataFrame, label_col: str, score_cols: list[str], dimensions: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    amount_cols = [column for column in ["prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m"] if column in df.columns]
    group_cols = [column for column in dimensions if column in df.columns]
    if not amount_cols and not group_cols:
        return rows
    for score_col in score_cols:
        if score_col not in df.columns:
            continue
        valid = df[score_col].notna()
        if int(valid.sum()) < 20:
            continue
        temp = df.loc[valid].copy()
        temp["score_band"] = pd.qcut(temp[score_col], 5, labels=False, duplicates="drop")
        by_cols = ["score_band"] + group_cols[:2]
        for keys, group in temp.groupby(by_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"score_column": score_col, "score_band": int(keys[0]) + 1 if pd.notna(keys[0]) else None, "n_samples": int(len(group)), "bad": int(group[label_col].sum()), "bad_rate": float(group[label_col].mean()) if len(group) else 0.0}
            for idx, column in enumerate(group_cols[:2], start=1):
                row[column] = str(keys[idx])
            for amount_col in amount_cols:
                row[f"{amount_col}_sum"] = float(pd.to_numeric(group[amount_col], errors="coerce").fillna(0).sum())
            rows.append(row)
    return rows


def _write_psi_outputs(df: pd.DataFrame, output_path: Path, score_cols: list[str], time_col: str) -> None:
    if time_col not in df.columns:
        return
    rows = []
    bin_rows = []
    for score_col in score_cols:
        psi_df, bin_df = compute_score_psi(df, score_col, time_col)
        if len(psi_df):
            psi_df["score_column"] = score_col
            rows.append(psi_df)
        if len(bin_df):
            bin_df["score_column"] = score_col
            bin_rows.append(bin_df)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(output_path / "score_psi_by_month.csv", index=False, encoding="utf-8-sig")
    if bin_rows:
        # Per-bin score distribution + PSI component by month (stability bin detail).
        pd.concat(bin_rows, ignore_index=True).to_csv(output_path / "score_psi_bin_detail.csv", index=False, encoding="utf-8-sig")


def _benchmark_uplift(df: pd.DataFrame, label_col: str, split_col: str, score_cols: list[str], splits: list[Any]) -> list[dict[str, Any]]:
    if "model_score" not in df.columns:
        return []
    rows = []
    for split_val in splits:
        mask = df[split_col] == split_val
        if int(mask.sum()) < 50:
            continue
        row = {"final_flag": split_val, "n_samples": int(mask.sum()), "model_score_auc": auc_score(df.loc[mask, label_col], df.loc[mask, "model_score"]), "model_score_ks": ks_score(df.loc[mask, label_col], df.loc[mask, "model_score"])}
        for ref in [score for score in score_cols if score != "model_score"]:
            row[f"{ref}_auc"] = auc_score(df.loc[mask, label_col], df.loc[mask, ref])
            row[f"{ref}_ks"] = ks_score(df.loc[mask, label_col], df.loc[mask, ref])
            if row["model_score_auc"] is not None and row[f"{ref}_auc"] is not None:
                row[f"auc_uplift_vs_{ref}"] = round(row["model_score_auc"] - row[f"{ref}_auc"], 6)
            if row["model_score_ks"] is not None and row[f"{ref}_ks"] is not None:
                row[f"ks_uplift_vs_{ref}"] = round(row["model_score_ks"] - row[f"{ref}_ks"], 6)
        rows.append(row)
    return rows


def _feature_gain_rows(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        split_val = row.get("final_flag")
        for key, value in row.items():
            if key.startswith("auc_uplift_vs_") or key.startswith("ks_uplift_vs_"):
                rows.append(
                    {
                        "split": split_val,
                        "metric": "auc" if key.startswith("auc_") else "ks",
                        "baseline_score": key.split("_vs_", 1)[-1],
                        "uplift": value,
                    }
                )
    return rows
