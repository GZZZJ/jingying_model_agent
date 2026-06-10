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


def evaluate_scores_from_feather(*, scores_feather: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate model and champion scores from a feather score table."""
    scores_path = Path(scores_feather)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eval_cfg = config["evaluation"]
    df = pd.read_feather(scores_path)

    label_col = eval_cfg["label_column"]
    split_col = eval_cfg["split_column"]
    time_col = eval_cfg.get("time_column", "mdl_dte")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    for column in ["prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    desired_scores = eval_cfg.get("score_columns", ["model_score"])
    score_cols = [column for column in desired_scores if column in df.columns]
    for column in score_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if time_col in df.columns:
        df["mdl_month"] = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)

    splits = sorted(df[split_col].dropna().unique().tolist())
    overall_rows: list[dict[str, Any]] = []
    for split_val in splits:
        mask = df[split_col] == split_val
        row = _base_slice_row(df, mask, label_col, split_col, split_val)
        _add_score_metrics(row, df, mask, label_col, score_cols)
        overall_rows.append(row)
    pd.DataFrame(overall_rows).to_csv(output_path / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    monthly_rows: list[dict[str, Any]] = []
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

    segment_rows = _segment_metrics(df, label_col, split_col, score_cols, splits)
    pd.DataFrame(segment_rows).to_csv(output_path / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    _write_decile_outputs(df, output_path, score_cols, label_col)
    _write_intent_risk_outputs(df, output_path, label_col)
    _write_psi_outputs(df, output_path, score_cols, time_col)
    benchmark_rows = _benchmark_uplift(df, label_col, split_col, score_cols, splits)
    pd.DataFrame(benchmark_rows).to_csv(output_path / "benchmark_uplift.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scores_source": str(scores_path),
        "score_columns_evaluated": score_cols,
        "label_column": label_col,
        "split_column": split_col,
        "splits_evaluated": splits,
        "n_total_samples": len(df),
        "n_score_columns": len(score_cols),
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


def _write_intent_risk_outputs(df: pd.DataFrame, output_path: Path, label_col: str) -> None:
    if "zc_level" not in df.columns or "model_score" not in df.columns:
        return
    valid = df["model_score"].notna()
    if int(valid.sum()) < 30:
        return
    bins = pd.qcut(df.loc[valid, "model_score"], 3, labels=False, duplicates="drop")
    df["intent_level"] = "中意愿"
    if bins.nunique() == 3:
        df.loc[valid, "intent_level"] = ["低意愿" if b == 0 else "中意愿" if b == 1 else "高意愿" for b in bins]
    elif bins.nunique() == 2:
        df.loc[valid, "intent_level"] = ["低意愿" if b == 0 else "高意愿" for b in bins]
    rows = []
    for intent in ["低意愿", "中意愿", "高意愿"]:
        for zc_val in sorted(df["zc_level"].dropna().unique()):
            mask = (df["intent_level"] == intent) & (df["zc_level"] == zc_val)
            n = int(mask.sum())
            if n:
                positive = int(df.loc[mask, label_col].sum())
                rows.append({"intent_level": intent, "zc_level": str(zc_val), "n_samples": n, "pct": round(n / len(df), 6), "bad": positive, "bad_rate": round(positive / n, 6)})
    intent_zc = pd.DataFrame(rows)
    if len(intent_zc):
        intent_zc.to_csv(output_path / "intent_zc_distribution.csv", index=False, encoding="utf-8-sig")
        intent_zc.pivot_table(index="intent_level", columns="zc_level", values="bad_rate", aggfunc="first").to_csv(
            output_path / "intent_zc_ftr_rate.csv", encoding="utf-8-sig"
        )
    if "prc_amt_xz_30d_3m" not in df.columns or "ovd_amt_xz_30d_3m" not in df.columns:
        return
    risk_rows = []
    head_rows = []
    for intent in ["低意愿", "中意愿", "高意愿"]:
        mask = df["intent_level"] == intent
        n = int(mask.sum())
        if n:
            principal = float(df.loc[mask, "prc_amt_xz_30d_3m"].fillna(0).sum())
            overdue = float(df.loc[mask, "ovd_amt_xz_30d_3m"].fillna(0).sum())
            head_risk = int((df.loc[mask, "ovd_amt_xz_30d_3m"].fillna(0) > 0).sum())
            risk_rows.append({"intent_level": intent, "n_samples": n, "total_principal": principal, "total_overdue": overdue, "amount_overdue_rate": overdue / principal if principal > 0 else 0, "head_risk_count": head_risk, "head_risk_rate": round(head_risk / n, 6)})
        for zc_val in sorted(df["zc_level"].dropna().unique()):
            zc_mask = mask & (df["zc_level"] == zc_val)
            zc_n = int(zc_mask.sum())
            if zc_n:
                head_risk = int((df.loc[zc_mask, "ovd_amt_xz_30d_3m"].fillna(0) > 0).sum())
                head_rows.append({"intent_level": intent, "zc_level": str(zc_val), "n_samples": zc_n, "head_risk_count": head_risk, "head_risk_rate": round(head_risk / zc_n, 6)})
    pd.DataFrame(risk_rows).to_csv(output_path / "intent_zc_amount_risk.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(head_rows).to_csv(output_path / "intent_zc_headcount_risk.csv", index=False, encoding="utf-8-sig")


def _write_psi_outputs(df: pd.DataFrame, output_path: Path, score_cols: list[str], time_col: str) -> None:
    if time_col not in df.columns:
        return
    rows = []
    for score_col in score_cols:
        psi_df = compute_score_psi(df, score_col, time_col)
        if len(psi_df):
            psi_df["score_column"] = score_col
            rows.append(psi_df)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(output_path / "score_psi_by_month.csv", index=False, encoding="utf-8-sig")


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
