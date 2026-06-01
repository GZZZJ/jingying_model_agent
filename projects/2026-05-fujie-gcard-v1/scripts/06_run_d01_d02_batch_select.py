#!/usr/bin/env python3
"""Run per-table D01 TOAD and D02 PSI screening for the backtracked features.

This script is intentionally not wired into the default project command yet.
It follows the current feature backtracking contract:

- source tables already contain sample columns plus one feature group
- use a 1/10 sample with rand_flag0 < 0.1
- keep only DEV and OOT rows
- run D01 on DEV rows
- run D02 comparing OOT against DEV

Example:
    python3 projects/2026-05-fujie-gcard-v1/scripts/06_run_d01_d02_batch_select.py

Portable usage:
    python3 06_run_d01_d02_batch_select.py \
      --feature-select-code-dir /path/to/feature-select-v2/scripts/code \
      --feature-columns /path/to/feature_columns.csv \
      --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COL = "ftr_30d_ord_flag"
SPLIT_COL = "final_flag"
DEV_VALUE = "DEV"
OOT_VALUE = "OOT"

D01_THRESHOLDS = {
    "empty": 0.95,
    "corr": 0.80,
    "iv": 0.005,
}
D02_PSI_THRESHOLD = 0.10

DEFAULT_ROUND_NUM = 500
DEFAULT_RANDOM_SEED = 0


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_DIR = SCRIPT_PATH.parents[1] if len(SCRIPT_PATH.parents) >= 2 else Path.cwd()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D01/D02 feature screening by feature table.")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR), help="Project workspace directory.")
    parser.add_argument(
        "--feature-select-code-dir",
        default=os.environ.get("FEATURE_SELECT_V2_CODE_DIR"),
        help=(
            "Path to feature-select-v2/scripts/code. If omitted, the script searches common paths "
            "under project-dir, cwd, and the script directory."
        ),
    )
    parser.add_argument(
        "--feature-columns",
        default="data/profile/feature_metadata/feature_columns.csv",
        help="Feature metadata CSV, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/d01_d02_batch_select",
        help="Output directory, relative to project-dir unless absolute.",
    )
    parser.add_argument("--table", action="append", help="Only run the specified full table name. Repeatable.")
    parser.add_argument("--max-tables", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--round-num", type=int, default=DEFAULT_ROUND_NUM, help="D01 feature batch size.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for feature order.")
    parser.add_argument("--force", action="store_true", help="Overwrite table checkpoints if they already exist.")
    parser.add_argument(
        "--use-native",
        action="store_true",
        help="Force feature-select-v2 native selector instead of toad. Default auto-detects toad.",
    )
    return parser.parse_args()


def find_feature_select_code_dir(project_dir: Path, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            project_dir / "vendor/feature-select-v2/scripts/code",
            project_dir.parent.parent / "vendor/feature-select-v2/scripts/code",
            Path.cwd() / "vendor/feature-select-v2/scripts/code",
            SCRIPT_PATH.parent / "feature-select-v2/scripts/code",
            SCRIPT_PATH.parent.parent / "feature-select-v2/scripts/code",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "utils" / "feature_select.py").exists():
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Cannot locate feature-select-v2 scripts/code. Pass --feature-select-code-dir or set "
        f"FEATURE_SELECT_V2_CODE_DIR.\nSearched:\n{searched}"
    )


def load_feature_select_functions(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir / "utils"))
    from utils.feature_select import batch_psi, d01_preselect_by_toad

    return batch_psi, d01_preselect_by_toad


def resolve_project_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def load_feature_map(feature_columns_path: Path) -> dict[str, list[str]]:
    table_features: dict[str, list[str]] = {}
    with feature_columns_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            table = row["full_table_name"]
            feature = row["feature_name"]
            table_features.setdefault(table, []).append(feature)
    return table_features


def table_slug(table_name: str) -> str:
    return table_name.replace(".", "__dot__")


def build_sample_sql(table_name: str) -> str:
    return f"""
select *
from {table_name}
where rand_flag0 < 0.1
and {SPLIT_COL} in ('{DEV_VALUE}', '{OOT_VALUE}')
"""


def coerce_features(df: pd.DataFrame, feature_list: list[str]) -> list[str]:
    available = [feature for feature in feature_list if feature in df.columns]
    for feature in available:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].replace([np.inf, -np.inf, -999, -998], np.nan)
    return available


def shuffle_features(feature_list: list[str], random_seed: int) -> list[str]:
    features = feature_list.copy()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(features)
    return features


def get_d01_remain(round_select_result: dict) -> list[str]:
    if not round_select_result:
        return []
    max_round = max(round_select_result)
    return [feature for group_result in round_select_result[max_round] for feature in group_result[2]]


def get_d01_drop_counts(round_select_result: dict, total_features: int, remain_features: list[str]) -> dict[str, int]:
    empty_drop = []
    corr_drop = []
    iv_drop = []
    for group_results in round_select_result.values():
        for select_result in group_results:
            empty_drop.extend(select_result[0].get("empty", []))
            corr_drop.extend(select_result[0].get("corr", []))
            iv_drop.extend(select_result[0].get("iv", []))
    return {
        "d01_empty_drop": len(set(empty_drop)),
        "d01_corr_drop": len(set(corr_drop)),
        "d01_iv_drop": len(set(iv_drop)),
        "d01_total_drop": total_features - len(remain_features),
    }


def run_d01(
    df: pd.DataFrame,
    feature_list: list[str],
    round_num: int,
    use_native: bool | None,
    d01_preselect_by_toad_func,
) -> tuple[dict, list[str]]:
    dev_df = df[(df[SPLIT_COL] == DEV_VALUE) & (df[TARGET_COL].isin([0, 1]))].copy()
    dev_df[TARGET_COL] = dev_df[TARGET_COL].astype(int)
    select_result = d01_preselect_by_toad_func(
        df=dev_df.loc[:, feature_list + [TARGET_COL]],
        target_col=TARGET_COL,
        feature_list=feature_list,
        preselect_condition=D01_THRESHOLDS,
        round_num=round_num,
        use_native=use_native,
        max_round=None,
    )
    return select_result, get_d01_remain(select_result)


def run_d02(df: pd.DataFrame, remain_features: list[str], batch_psi_func) -> tuple[dict, dict[str, float], list[str]]:
    if not remain_features:
        return {}, {}, []
    dev_df = df[df[SPLIT_COL] == DEV_VALUE].copy()
    oot_df = df[df[SPLIT_COL] == OOT_VALUE].copy()
    if dev_df.empty or oot_df.empty:
        raise RuntimeError(f"D02 requires both {DEV_VALUE} and {OOT_VALUE} rows.")
    data_iter = iter(
        [
            ("base_DEV", dev_df.loc[:, remain_features]),
            ("exp_OOT", oot_df.loc[:, remain_features]),
        ]
    )
    psi_result = batch_psi_func(data_iter, remain_features, method="quantile", num_nbins=10)
    feature_max_psi = {feature: max(psi_info.values()) for feature, psi_info in psi_result[2].items()}
    psi_drop_features = [feature for feature, psi in feature_max_psi.items() if psi > D02_PSI_THRESHOLD]
    return psi_result, feature_max_psi, psi_drop_features


def write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "table",
        "input_features",
        "sample_rows",
        "dev_rows",
        "oot_rows",
        "d01_remain",
        "d01_empty_drop",
        "d01_corr_drop",
        "d01_iv_drop",
        "d01_total_drop",
        "d02_psi_drop",
        "final_remain",
        "elapsed_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    feature_select_code_dir = find_feature_select_code_dir(project_dir, args.feature_select_code_dir)
    batch_psi_func, d01_preselect_by_toad_func = load_feature_select_functions(feature_select_code_dir)

    from tmlpatch.database import TMLSQLClient

    feature_columns_path = resolve_project_path(project_dir, args.feature_columns)
    output_dir = resolve_project_path(project_dir, args.output_dir)
    cache_dir = output_dir / "cache"
    sql_dir = output_dir / "sql"
    results_dir = output_dir / "results"

    table_features = load_feature_map(feature_columns_path)
    if args.table:
        requested = set(args.table)
        table_features = {table: features for table, features in table_features.items() if table in requested}
    if args.max_tables is not None:
        table_features = dict(list(table_features.items())[: args.max_tables])

    run_meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_dir": str(project_dir),
        "feature_columns": str(feature_columns_path),
        "feature_select_code_dir": str(feature_select_code_dir),
        "thresholds": {
            "d01": D01_THRESHOLDS,
            "d02_psi": D02_PSI_THRESHOLD,
        },
        "sampling": {
            "where": f"rand_flag0 < 0.1 and {SPLIT_COL} in ('{DEV_VALUE}', '{OOT_VALUE}')",
            "d01": f"{SPLIT_COL} = '{DEV_VALUE}'",
            "d02": f"{DEV_VALUE} vs {OOT_VALUE}",
        },
    }
    write_json(output_dir / "run_config.json", run_meta)

    summary_rows: list[dict] = []
    final_remain_by_table: dict[str, list[str]] = {}

    client = TMLSQLClient()
    try:
        total_tables = len(table_features)
        for table_index, (table_name, feature_list) in enumerate(table_features.items(), start=1):
            start_time = time.time()
            slug = table_slug(table_name)
            checkpoint_path = cache_dir / f"{slug}.pkl"
            if checkpoint_path.exists() and not args.force:
                with checkpoint_path.open("rb") as handle:
                    checkpoint = pickle.load(handle)
                summary_rows.append(checkpoint["summary"])
                final_remain_by_table[table_name] = checkpoint["final_remain_features"]
                print(f"[SKIP] ({table_index}/{total_tables}) {table_name}")
                continue

            print(f"[INFO] ({table_index}/{total_tables}) {table_name}, features={len(feature_list)}", flush=True)
            sample_sql = build_sample_sql(table_name)
            (sql_dir / f"{slug}.sql").parent.mkdir(parents=True, exist_ok=True)
            (sql_dir / f"{slug}.sql").write_text(sample_sql.strip() + "\n", encoding="utf-8")

            sample = client.sql(sample_sql).to_pandas()
            available_features = coerce_features(sample, feature_list)
            if not available_features:
                raise RuntimeError(f"No feature columns available in table: {table_name}")
            available_features = shuffle_features(available_features, args.random_seed + table_index)

            d01_result, d01_remain = run_d01(
                sample,
                available_features,
                round_num=args.round_num,
                use_native=True if args.use_native else None,
                d01_preselect_by_toad_func=d01_preselect_by_toad_func,
            )
            psi_result, feature_max_psi, psi_drop_features = run_d02(sample, d01_remain, batch_psi_func)
            final_remain = [feature for feature in d01_remain if feature not in set(psi_drop_features)]

            elapsed = round(time.time() - start_time, 3)
            drop_counts = get_d01_drop_counts(d01_result, len(available_features), d01_remain)
            summary = {
                "table": table_name,
                "input_features": len(available_features),
                "sample_rows": int(len(sample)),
                "dev_rows": int((sample[SPLIT_COL] == DEV_VALUE).sum()),
                "oot_rows": int((sample[SPLIT_COL] == OOT_VALUE).sum()),
                "d01_remain": len(d01_remain),
                **drop_counts,
                "d02_psi_drop": len(psi_drop_features),
                "final_remain": len(final_remain),
                "elapsed_seconds": elapsed,
            }

            checkpoint = {
                "summary": summary,
                "d01_result": d01_result,
                "d01_remain_features": d01_remain,
                "d02_psi_result": psi_result,
                "d02_feature_max_psi": feature_max_psi,
                "d02_psi_drop_features": psi_drop_features,
                "final_remain_features": final_remain,
            }
            write_pickle(checkpoint_path, checkpoint)
            summary_rows.append(summary)
            final_remain_by_table[table_name] = final_remain
            print(
                f"[OK] {table_name}: input={len(available_features)}, "
                f"d01_remain={len(d01_remain)}, final={len(final_remain)}, elapsed={elapsed}s",
                flush=True,
            )
            del sample
            gc.collect()
    finally:
        client.stop()

    write_summary_csv(results_dir / "d01_d02_table_summary.csv", summary_rows)
    write_json(results_dir / "d01_d02_final_remain_features.json", final_remain_by_table)
    write_json(
        results_dir / "d01_d02_run_summary.json",
        {
            "tables": len(summary_rows),
            "input_features": sum(int(row["input_features"]) for row in summary_rows),
            "d01_remain": sum(int(row["d01_remain"]) for row in summary_rows),
            "d02_psi_drop": sum(int(row["d02_psi_drop"]) for row in summary_rows),
            "final_remain": sum(int(row["final_remain"]) for row in summary_rows),
        },
    )
    print(f"[DONE] output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
