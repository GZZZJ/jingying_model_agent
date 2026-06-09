#!/usr/bin/env python3
"""Evaluation script for 复借G卡 model scores.

Computes overall, monthly, segment, decile-lift, intent-zc cross,
risk observation, and stability (PSI) metrics.

Usage:
    python3 scripts/04_evaluate.py \
      --scores-feather runs/model_scores/scores_all_splits.feather \
      --output-dir runs/model_eval \
      --config configs/evaluate.yaml
"""

from __future__ import annotations

import argparse
import json
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


# ── helper: safe AUC / KS ────────────────────────────────────────


def _auc(y_true, y_score) -> float | None:
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(ys) & np.isin(yt, [0, 1])
    if mask.sum() < 2 or len(np.unique(yt[mask])) < 2:
        return None
    return float(roc_auc_score(yt[mask], ys[mask]))


def _ks(y_true, y_score) -> float | None:
    from scipy.stats import ks_2samp

    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(ys) & np.isin(yt, [0, 1])
    if mask.sum() < 2 or len(np.unique(yt[mask])) < 2:
        return None
    pos = ys[mask][yt[mask] == 1]
    neg = ys[mask][yt[mask] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(ks_2samp(pos, neg).statistic)


def _psi(expected, actual) -> float | None:
    """PSI between two distributions (percentages)."""
    eps = 1e-10
    e = np.asarray(expected, dtype=float) + eps
    a = np.asarray(actual, dtype=float) + eps
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


# ── helper: decile lift ──────────────────────────────────────────


def compute_decile_lift(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
) -> pd.DataFrame:
    """Compute decile lift table for a given score."""
    sub = df[[score_col, label_col]].dropna(subset=[score_col]).copy()
    if len(sub) < 20:
        return pd.DataFrame()

    # 10 equal-frequency bins (higher score = higher risk)
    sub["decile"] = pd.qcut(sub[score_col], 10, labels=False, duplicates="drop")
    n_bins = sub["decile"].nunique()
    if n_bins < 2:
        return pd.DataFrame()

    bad_rate_total = sub[label_col].mean()

    rows = []
    for d in sorted(sub["decile"].unique(), reverse=True):
        group = sub[sub["decile"] == d]
        n = len(group)
        bad = int(group[label_col].sum())
        br = group[label_col].mean()
        cum_bad = int(sub[sub["decile"] >= d][label_col].sum())
        cum_n = int((sub["decile"] >= d).sum())
        remaining_bad = int(sub[sub["decile"] < d][label_col].sum())
        remaining_n = int((sub["decile"] < d).sum())

        rows.append({
            "decile": int(d) + 1,
            "n_samples": n,
            "pct": n / len(sub),
            "bad": bad,
            "bad_rate": br,
            "cum_bad": cum_bad,
            "cum_bad_rate": cum_bad / cum_n if cum_n > 0 else 0,
            "cum_lift": (cum_bad / cum_n) / bad_rate_total if bad_rate_total > 0 and cum_n > 0 else 0,
            "remaining_bad": remaining_bad,
            "remaining_bad_rate": remaining_bad / remaining_n if remaining_n > 0 else 0,
            "remaining_lift": (remaining_bad / remaining_n) / bad_rate_total if bad_rate_total > 0 and remaining_n > 0 else 0,
        })

    return pd.DataFrame(rows)


# ── helper: monthly PSI ──────────────────────────────────────────


def compute_score_psi(
    df: pd.DataFrame,
    score_col: str,
    time_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI of score distribution by month.

    Uses the first month as the expected distribution.
    """
    sub = df[[score_col, time_col]].dropna(subset=[score_col]).copy()
    if len(sub) < 20:
        return pd.DataFrame()

    # Convert time_col to month
    if sub[time_col].dtype == "object":
        sub["_month"] = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")
    else:
        sub["_month"] = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")

    months = sorted(sub["_month"].dropna().unique())
    if len(months) < 2:
        return pd.DataFrame()

    # Bin all data together
    sub["_bin"] = pd.qcut(sub[score_col], n_bins, labels=False, duplicates="drop")

    # Expected: first month
    base = sub[sub["_month"] == months[0]]
    base_dist = base["_bin"].value_counts(normalize=True).sort_index()

    rows = []
    for m in months:
        current = sub[sub["_month"] == m]
        current_dist = current["_bin"].value_counts(normalize=True).sort_index()

        # Align distributions
        all_bins = sorted(set(base_dist.index) | set(current_dist.index))
        e_dist = [base_dist.get(b, 0) for b in all_bins]
        a_dist = [current_dist.get(b, 0) for b in all_bins]

        rows.append({
            "month": str(m),
            "psi": _psi(e_dist, a_dist),
            "n_samples": len(current),
        })

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Evaluate 复借G卡 model scores")
    parser.add_argument("--scores-feather", required=True, help="Path to scores_all_splits.feather")
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--config", default="configs/evaluate.yaml", help="Evaluation config yaml")
    args = parser.parse_args()

    scores_path = project_dir / args.scores_feather if not Path(args.scores_feather).is_absolute() else Path(args.scores_feather)
    output_dir = project_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    config_path = project_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)

    cfg = load_yaml(config_path)
    eval_cfg = cfg["evaluation"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load scores ───────────────────────────────────────────
    print(f"[LOAD] Reading {scores_path} ...")
    t0 = time.time()
    df = pd.read_feather(scores_path)
    print(f"[LOAD] {df.shape} in {time.time() - t0:.1f}s")

    label_col = eval_cfg["label_column"]
    split_col = eval_cfg["split_column"]
    time_col = eval_cfg.get("time_column", "mdl_dte")

    # Ensure label is numeric
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    # Ensure risk columns are numeric
    for rc in ["prc_amt_xz_30d_3m", "ovd_amt_xz_30d_3m"]:
        if rc in df.columns:
            df[rc] = pd.to_numeric(df[rc], errors="coerce")

    # Determine available score columns and coerce to numeric
    desired_scores = eval_cfg.get("score_columns", ["model_score", "gcard_v2", "gcard_v4", "gcard_v5", "gcard_v6"])
    score_cols = [s for s in desired_scores if s in df.columns]
    for sc in score_cols:
        df[sc] = pd.to_numeric(df[sc], errors="coerce")
    print(f"[SCORE] Available score columns: {score_cols}")

    # Parse mdl_dte to month if available
    if time_col in df.columns:
        if df[time_col].dtype == "object":
            df["mdl_month"] = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)
        else:
            df["mdl_month"] = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)

    # ── 2. Overall metrics ───────────────────────────────────────
    print("[EVAL] Computing overall metrics ...")
    overall_rows = []
    splits = df[split_col].dropna().unique().tolist()

    for split_val in sorted(splits):
        mask = df[split_col] == split_val
        n = int(mask.sum())
        pos = int(df.loc[mask, label_col].sum())
        br = pos / n if n > 0 else 0.0

        row: dict[str, Any] = {
            "final_flag": split_val, "n_samples": n, "positive": pos, "bad_rate": round(br, 6),
        }
        for sc in score_cols:
            if sc in df.columns:
                row[f"{sc}_auc"] = _auc(df.loc[mask, label_col], df.loc[mask, sc])
                row[f"{sc}_ks"] = _ks(df.loc[mask, label_col], df.loc[mask, sc])
        overall_rows.append(row)

    overall_metrics = pd.DataFrame(overall_rows)
    overall_metrics.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    # ── 3. Monthly metrics ───────────────────────────────────────
    print("[EVAL] Computing monthly metrics ...")
    monthly_rows = []
    if "mdl_month" in df.columns:
        for split_val in sorted(splits):
            split_mask = df[split_col] == split_val
            for month in sorted(df.loc[split_mask, "mdl_month"].dropna().unique()):
                mask = split_mask & (df["mdl_month"] == month)
                n = int(mask.sum())
                if n < 50:
                    continue
                pos = int(df.loc[mask, label_col].sum())
                br = pos / n

                row: dict[str, Any] = {
                    "mdl_month": month, "final_flag": split_val,
                    "n_samples": n, "positive": pos, "bad_rate": round(br, 6),
                }
                for sc in score_cols:
                    if sc in df.columns:
                        row[f"{sc}_auc"] = _auc(df.loc[mask, label_col], df.loc[mask, sc])
                        row[f"{sc}_ks"] = _ks(df.loc[mask, label_col], df.loc[mask, sc])
                monthly_rows.append(row)

    monthly_metrics = pd.DataFrame(monthly_rows)
    monthly_metrics.to_csv(output_dir / "monthly_metrics.csv", index=False, encoding="utf-8-sig")

    # ── 4. Segment metrics ───────────────────────────────────────
    print("[EVAL] Computing segment metrics ...")
    segment_defs = {
        "全客群": None,
        "老户次新": "blue_customer_flag in ['E2', 'E3']",
        "老户": "blue_customer_flag == 'E3'",
        "次新": "blue_customer_flag == 'E2'",
        "流失户": "blue_customer_flag == 'B2'",
    }

    segment_rows = []
    for seg_name, seg_filter in segment_defs.items():
        if seg_filter is None:
            seg_mask = pd.Series(True, index=df.index)
        else:
            seg_mask = df.eval(seg_filter)

        for split_val in sorted(splits):
            mask = seg_mask & (df[split_col] == split_val)
            n = int(mask.sum())
            if n < 50:
                continue
            pos = int(df.loc[mask, label_col].sum())
            br = pos / n

            row: dict[str, Any] = {
                "segment": seg_name, "final_flag": split_val,
                "n_samples": n, "positive": pos, "bad_rate": round(br, 6),
            }
            for sc in score_cols:
                if sc in df.columns:
                    row[f"{sc}_auc"] = _auc(df.loc[mask, label_col], df.loc[mask, sc])
                    row[f"{sc}_ks"] = _ks(df.loc[mask, label_col], df.loc[mask, sc])

            # KS uplift relative to historical scores
            if "model_score_ks" in row and row["model_score_ks"] is not None:
                for sc in score_cols:
                    if sc != "model_score" and f"{sc}_ks" in row and row[f"{sc}_ks"] is not None:
                        row[f"ks_uplift_vs_{sc}"] = round(row["model_score_ks"] - row[f"{sc}_ks"], 6)
            segment_rows.append(row)

    segment_metrics = pd.DataFrame(segment_rows)
    segment_metrics.to_csv(output_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")

    # ── 5. Decile lift tables ────────────────────────────────────
    print("[EVAL] Computing decile lift ...")

    for sc in score_cols:
        # All
        dec_all = compute_decile_lift(df, sc, label_col)
        if len(dec_all) > 0:
            dec_all.to_csv(output_dir / f"decile_lift_all_{sc}.csv", index=False, encoding="utf-8-sig")

        # E2+E3
        if "blue_customer_flag" in df.columns:
            mask_e2e3 = df["blue_customer_flag"].isin(["E2", "E3"])
            dec_e2e3 = compute_decile_lift(df[mask_e2e3], sc, label_col)
            if len(dec_e2e3) > 0:
                dec_e2e3.to_csv(output_dir / f"decile_lift_e2e3_{sc}.csv", index=False, encoding="utf-8-sig")

            # B2
            mask_b2 = df["blue_customer_flag"] == "B2"
            dec_b2 = compute_decile_lift(df[mask_b2], sc, label_col)
            if len(dec_b2) > 0:
                dec_b2.to_csv(output_dir / f"decile_lift_b2_{sc}.csv", index=False, encoding="utf-8-sig")

    # Also save with the exact names from the plan (for model_score)
    for sc in score_cols:
        if sc == "model_score":
            dec_all = compute_decile_lift(df, sc, label_col)
            if len(dec_all) > 0:
                dec_all.to_csv(output_dir / "decile_lift_all.csv", index=False, encoding="utf-8-sig")

            if "blue_customer_flag" in df.columns:
                mask_e2e3 = df["blue_customer_flag"].isin(["E2", "E3"])
                dec_e2e3 = compute_decile_lift(df[mask_e2e3], sc, label_col)
                if len(dec_e2e3) > 0:
                    dec_e2e3.to_csv(output_dir / "decile_lift_e2e3.csv", index=False, encoding="utf-8-sig")

                mask_b2 = df["blue_customer_flag"] == "B2"
                dec_b2 = compute_decile_lift(df[mask_b2], sc, label_col)
                if len(dec_b2) > 0:
                    dec_b2.to_csv(output_dir / "decile_lift_b2.csv", index=False, encoding="utf-8-sig")

    # ── 6. Intent x zc_level cross ────────────────────────────────
    print("[EVAL] Computing intent x zc_level cross ...")
    if "zc_level" in df.columns and "model_score" in df.columns:
        # Cut model_score into low/mid/high (equal-frequency thirds)
        valid = df["model_score"].notna()
        if valid.sum() >= 30:
            score_valid = df.loc[valid, "model_score"]
            try:
                bins = pd.qcut(score_valid, 3, labels=False, duplicates="drop")
                # Map bin index to labels
                bin_count = bins.max() + 1
                labels_map = {0: "低意愿", 1: "中意愿", 2: "高意愿"}
                if bin_count == 3:
                    label_list = [labels_map[b] for b in bins]
                elif bin_count == 2:
                    label_list = ["低意愿" if b == 0 else "高意愿" for b in bins]
                else:
                    label_list = ["中意愿"] * len(bins)
                df["intent_level"] = "中意愿"  # default
                df.loc[valid, "intent_level"] = label_list

                intent_zc_rows = []
                for intent in ["低意愿", "中意愿", "高意愿"]:
                    if (df["intent_level"] == intent).sum() == 0:
                        continue
                    for zc_val in sorted(df["zc_level"].dropna().unique()):
                        mask = (df["intent_level"] == intent) & (df["zc_level"] == zc_val)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        pos = int(df.loc[mask, label_col].sum())
                        intent_zc_rows.append({
                            "intent_level": intent,
                            "zc_level": str(zc_val),
                            "n_samples": n,
                            "pct": round(n / len(df), 6),
                            "bad": pos,
                            "bad_rate": round(pos / n, 6) if n > 0 else 0,
                        })

                intent_zc = pd.DataFrame(intent_zc_rows)
                intent_zc.to_csv(output_dir / "intent_zc_distribution.csv", index=False, encoding="utf-8-sig")

                # Intent x zc bad rate pivot
                if len(intent_zc) > 0:
                    br_pivot = intent_zc.pivot_table(
                        index="intent_level", columns="zc_level", values="bad_rate", aggfunc="first",
                    )
                    br_pivot.to_csv(output_dir / "intent_zc_ftr_rate.csv", encoding="utf-8-sig")
            except Exception as e:
                print(f"  [WARN] intent-zc cross failed: {e}")

    # ── 7. Risk observation ──────────────────────────────────────
    print("[EVAL] Computing risk observation ...")
    risk_amount_col = "prc_amt_xz_30d_3m"
    risk_ovd_col = "ovd_amt_xz_30d_3m"

    if risk_amount_col in df.columns and risk_ovd_col in df.columns:
        if "intent_level" in df.columns:
            risk_rows = []
            for intent in ["低意愿", "中意愿", "高意愿"]:
                if (df["intent_level"] == intent).sum() == 0:
                    continue
                mask = df["intent_level"] == intent
                n = int(mask.sum())
                total_prc = float(df.loc[mask, risk_amount_col].fillna(0).sum())
                total_ovd = float(df.loc[mask, risk_ovd_col].fillna(0).sum())
                head_risk = int((df.loc[mask, risk_ovd_col].fillna(0) > 0).sum())
                head_risk_rate = head_risk / n if n > 0 else 0

                risk_rows.append({
                    "intent_level": intent,
                    "n_samples": n,
                    "total_principal": total_prc,
                    "total_overdue": total_ovd,
                    "amount_overdue_rate": total_ovd / total_prc if total_prc > 0 else 0,
                    "head_risk_count": head_risk,
                    "head_risk_rate": round(head_risk_rate, 6),
                })

            risk_amount = pd.DataFrame(risk_rows)
            risk_amount.to_csv(output_dir / "intent_zc_amount_risk.csv", index=False, encoding="utf-8-sig")

            # Headcount risk
            head_risk_rows = []
            for intent in ["低意愿", "中意愿", "高意愿"]:
                if (df["intent_level"] == intent).sum() == 0:
                    continue
                if "zc_level" in df.columns:
                    for zc_val in sorted(df["zc_level"].dropna().unique()):
                        mask = (df["intent_level"] == intent) & (df["zc_level"] == zc_val)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        head_risk = int((df.loc[mask, risk_ovd_col].fillna(0) > 0).sum())
                        head_risk_rows.append({
                            "intent_level": intent,
                            "zc_level": str(zc_val),
                            "n_samples": n,
                            "head_risk_count": head_risk,
                            "head_risk_rate": round(head_risk / n, 6),
                        })

            head_risk_df = pd.DataFrame(head_risk_rows)
            if len(head_risk_df) > 0:
                head_risk_df.to_csv(output_dir / "intent_zc_headcount_risk.csv", index=False, encoding="utf-8-sig")

    # ── 8. Stability / PSI ────────────────────────────────────────
    print("[EVAL] Computing score PSI by month ...")
    if "mdl_month" in df.columns:
        psi_rows = []
        for sc in score_cols:
            psi_df = compute_score_psi(df, sc, "mdl_dte" if "mdl_dte" in df.columns else time_col)
            if len(psi_df) > 0:
                psi_df["score_column"] = sc
                psi_rows.append(psi_df)

        if psi_rows:
            psi_all = pd.concat(psi_rows, ignore_index=True)
            psi_all.to_csv(output_dir / "score_psi_by_month.csv", index=False, encoding="utf-8-sig")

    # ── 9. Benchmark uplift ──────────────────────────────────────
    print("[EVAL] Computing benchmark uplift ...")
    if "model_score" in df.columns:
        benchmark_rows = []
        ref_scores = [s for s in score_cols if s != "model_score"]

        for split_val in sorted(splits):
            mask = df[split_col] == split_val
            n = int(mask.sum())
            if n < 50:
                continue

            model_auc = _auc(df.loc[mask, label_col], df.loc[mask, "model_score"])
            model_ks = _ks(df.loc[mask, label_col], df.loc[mask, "model_score"])

            row: dict[str, Any] = {"final_flag": split_val, "n_samples": n, "model_score_auc": model_auc, "model_score_ks": model_ks}
            for ref in ref_scores:
                if ref in df.columns:
                    ref_auc = _auc(df.loc[mask, label_col], df.loc[mask, ref])
                    ref_ks = _ks(df.loc[mask, label_col], df.loc[mask, ref])
                    row[f"{ref}_auc"] = ref_auc
                    row[f"{ref}_ks"] = ref_ks
                    if model_auc is not None and ref_auc is not None:
                        row[f"auc_uplift_vs_{ref}"] = round(model_auc - ref_auc, 6)
                    if model_ks is not None and ref_ks is not None:
                        row[f"ks_uplift_vs_{ref}"] = round(model_ks - ref_ks, 6)
            benchmark_rows.append(row)

        benchmark = pd.DataFrame(benchmark_rows)
        benchmark.to_csv(output_dir / "benchmark_uplift.csv", index=False, encoding="utf-8-sig")

    # ── 10. Evaluation summary ────────────────────────────────────
    print("[EVAL] Writing evaluation summary ...")

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
            sc: {
                split_val: {
                    "auc": next((r.get(f"{sc}_auc") for r in overall_rows if r["final_flag"] == split_val), None),
                    "ks": next((r.get(f"{sc}_ks") for r in overall_rows if r["final_flag"] == split_val), None),
                }
                for split_val in splits
            }
            for sc in score_cols
        },
        "output_dir": str(output_dir),
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Evaluation complete: {output_dir}")
    print(f"  Score columns: {score_cols}")
    print(f"  Splits: {splits}")
    print("=" * 60)


if __name__ == "__main__":
    main()
