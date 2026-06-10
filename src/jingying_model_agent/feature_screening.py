"""Feature screening process summaries for model project reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jingying_model_agent.config import load_yaml


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_feature_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_top_features(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "rank": int(float(row.get("rank") or len(rows) + 1)),
                    "feature": row.get("feature", ""),
                    "gain": float(row.get("gain") or 0),
                    "split": int(float(row.get("split") or 0)),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def build_feature_screening_summary(project_dir: str | Path) -> dict[str, Any]:
    """Build a source-backed summary of the current completed screening flow."""
    project_path = Path(project_dir).resolve()
    project_config = load_yaml(project_path / "project.yml" if (project_path / "project.yml").exists() else project_path / "project.yaml")
    feature_select_config = load_yaml(project_path / "configs" / "feature_select.yaml").get("feature_select", {})
    refine_config = load_yaml(project_path / "configs" / "refine_features.yaml")["feature_refine"]

    d01_d02 = _read_json(project_path / "runs" / "d01_d02_batch_select" / "results" / "d01_d02_run_summary.json")
    preferred_refine_dir = project_path / "runs" / "feature_refine_feather"
    configured_refine_dir = project_path / refine_config.get("output_dir", "runs/feature_refine_wide")
    refine_dir = preferred_refine_dir if (preferred_refine_dir / "stage_summary.json").exists() else configured_refine_dir
    refine_summary_path = refine_dir / "stage_summary.json"
    refine_summary = _read_json(refine_summary_path)
    final_features_path = refine_dir / "final_500_features.txt"
    d05_importance_path = refine_dir / "d05_baseline_importance.csv"

    final_count = _read_feature_count(final_features_path)
    thresholds = feature_select_config.get("thresholds", {})
    d01_d02_config = feature_select_config.get("d01_d02", {})
    train_value = d01_d02_config.get("train_value", "DEV")
    valid_value = d01_d02_config.get("valid_value", "OOT")
    d01_thresholds = (
        f"缺失率 < {float(thresholds.get('empty', 0.95)):.2f}，"
        f"相关性 < {float(thresholds.get('corr', 0.80)):.2f}，"
        f"IV >= {float(thresholds.get('iv', 0.005)):.3f}"
    )
    d02_threshold = f"{train_value} vs {valid_value}，PSI <= {float(thresholds.get('psi', 0.10)):.2f}"
    global_corr = refine_config["global_corr"]
    d03 = refine_config["d03_random_importance"]
    d04 = refine_config["d04_null_importance"]
    d05 = refine_config["d05_baseline_importance"]

    initial_feature_count = int(d01_d02["input_features"])
    initial_table_count = int(d01_d02.get("tables", 0)) or len(feature_select_config.get("bigtable", {}).get("tables", []))
    screening_rows = [
        {
            "step": "初始",
            "method": f"初始候选变量总数：{initial_table_count}张特征表，共{initial_feature_count:,}个字段级候选变量",
            "remaining_features": initial_feature_count,
            "source": "runs/d01_d02_batch_select/results/d01_d02_run_summary.json",
        },
        {
            "step": 1,
            "method": f"分表基础预筛：{d01_thresholds}",
            "remaining_features": int(d01_d02["d01_remain"]),
            "source": "runs/d01_d02_batch_select/results/d01_d02_run_summary.json",
        },
        {
            "step": 2,
            "method": f"稳定性筛选：{d02_threshold}",
            "remaining_features": int(d01_d02["final_remain"]),
            "source": "runs/d01_d02_batch_select/results/d01_d02_run_summary.json",
        },
        {
            "step": 3,
            "method": (
                f"Feather观察样本可用特征：{int(refine_summary.get('total_rows', refine_summary.get('raw_rows', 0))):,}行，"
                "过滤缺失率过低和常量字段"
            ),
            "remaining_features": int(refine_summary["available_features"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 4,
            "method": f"全局相关性去重：相关性阈值 {float(global_corr['threshold']):.2f}，按单变量AUC保留更强特征",
            "remaining_features": int(refine_summary["after_global_corr"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 5,
            "method": (
                "随机噪声重要性筛选："
                f"{int(d03['rounds'])}轮，{int(d03['random_feature_count'])}个随机噪声特征，"
                f"存活率 >= {float(d03['min_survival_rate']):.2f}"
            ),
            "remaining_features": int(refine_summary["after_d03_random_importance"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 6,
            "method": (
                "空标签重要性筛选："
                f"{int(d04['null_rounds'])}轮空标签，空标签重要性{int(d04['null_percentile'])}分位，"
                f"score >= {float(d04['score_threshold']):.2f}"
            ),
            "remaining_features": int(refine_summary["after_d04_null_importance"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 7,
            "method": (
                "基线模型重要性筛选：LightGBM gain importance，"
                f"保留前{int(d05['keep_top_n'])}个，valid AUC={float(refine_summary['d05_valid_auc']):.4f}"
            ),
            "remaining_features": final_count,
            "source": str(final_features_path.relative_to(project_path)),
        },
    ]

    return {
        "project": {
            "name": project_config["project"]["name"],
            "display_name": project_config["project"]["display_name"],
            "scenario": project_config["project"]["scenario"],
        },
        "basis": {
            "primary_refine_source": refine_dir.name,
            "feather_path": refine_summary.get("feather_path", ""),
            "initial_feature_count": initial_feature_count,
            "sample_rows": int(refine_summary.get("total_rows", refine_summary.get("raw_rows", 0))),
            "train_samples": int(refine_summary.get("train_samples", 0)),
            "valid_samples": int(refine_summary.get("valid_samples", 0)),
            "d05_valid_auc": float(refine_summary["d05_valid_auc"]),
            "stage_summary": str(refine_summary_path.relative_to(project_path)),
        },
        "feature_select_v2_alignment": {
            "status": "concept_aligned_not_exact_reimplementation",
            "summary": (
                f"当前{project_config['project']['display_name']} Feather 主线借鉴 feature-select-v2 的随机重要性、"
                "Null Importance、Top Importance 思路，但阈值和局部实现为项目内自定义。"
            ),
            "local_reference_paths": [
                "my-skills/develop/feature-select-v2",
                "vendor/feature-select-v2",
            ],
            "current_project_config": "configs/refine_features.yaml",
        },
        "screening_rows": screening_rows,
        "top_features": _read_top_features(d05_importance_path),
    }


def write_feature_screening_summary(project_dir: str | Path, output_path: str | Path) -> Path:
    project_path = Path(project_dir).resolve()
    path = Path(output_path)
    resolved = path if path.is_absolute() else project_path / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(build_feature_screening_summary(project_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved
