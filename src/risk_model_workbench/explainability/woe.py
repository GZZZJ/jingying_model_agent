"""WOE artifact generation for top model features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "feature",
    "rank",
    "gain",
    "split_importance",
    "bin_order",
    "bin_label",
    "lower_bound",
    "upper_bound",
    "is_missing_bin",
    "split_value",
    "good",
    "bad",
    "total",
    "bad_rate",
    "pop_pct",
    "woe",
    "iv_component",
    "status",
    "skip_reason",
]


@dataclass(frozen=True)
class WoeGenerationResult:
    """Paths written by top-feature WOE generation."""

    output_dir: Path
    summary_path: Path
    image_paths: list[Path]


def sanitize_feature_filename(feature: str) -> str:
    """Return a filename-safe feature name."""
    safe = re.sub(r'[\\/*?:"<>|]', "_", str(feature))
    safe = safe.strip().strip(".")
    return safe or "feature"


def select_top_features(importance: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Select Top N features by descending gain and add one-based rank."""
    required = {"feature", "gain"}
    missing = required - set(importance.columns)
    if missing:
        raise ValueError(f"feature importance missing columns: {sorted(missing)}")

    selected = importance.copy()
    selected["gain"] = pd.to_numeric(selected["gain"], errors="coerce").fillna(0.0)
    selected = selected.sort_values("gain", ascending=False).head(top_n).reset_index(drop=True)
    selected.insert(0, "rank", range(1, len(selected) + 1))
    if "split" not in selected.columns:
        selected["split"] = np.nan
    return selected


def compute_woe_table(
    df: pd.DataFrame,
    *,
    feature: str,
    rank: int,
    gain: float,
    split_importance: Any,
    label_col: str,
    split_col: str,
    base_split_value: Any = "DEV",
    n_bins: int = 10,
    missing_values: Iterable[Any] | None = None,
    smooth: float = 0.5,
) -> pd.DataFrame:
    """Compute a long-format WOE table for one feature."""
    missing_columns = [column for column in [feature, label_col, split_col] if column not in df.columns]
    if missing_columns:
        return _skip_frame(feature, rank, gain, split_importance, f"missing_columns:{','.join(missing_columns)}")

    data = df[[feature, label_col, split_col]].copy()
    data["_feature_value"] = _coerce_feature_values(data[feature], missing_values)
    data["_is_missing"] = data["_feature_value"].isna()
    data[label_col] = pd.to_numeric(data[label_col], errors="coerce")
    data = data[data[label_col].isin([0, 1])].copy()
    if data.empty:
        return _skip_frame(feature, rank, gain, split_importance, "no_binary_label_rows")

    base_mask = data[split_col] == base_split_value
    if not bool(base_mask.any()):
        return _skip_frame(feature, rank, gain, split_importance, "base_split_empty")

    bins = _build_numeric_bins(data.loc[base_mask & ~data["_is_missing"], "_feature_value"], n_bins)
    if bins is None or len(bins) < 3:
        return _skip_frame(feature, rank, gain, split_importance, "too_few_bins")

    data["_bin_order"] = np.nan
    data["_bin_label"] = pd.Series(index=data.index, dtype="object")
    data["_lower_bound"] = np.nan
    data["_upper_bound"] = np.nan

    if bool(data["_is_missing"].any()):
        data.loc[data["_is_missing"], "_bin_order"] = 0
        data.loc[data["_is_missing"], "_bin_label"] = "Missing"
        numeric_start = 1
    else:
        numeric_start = 0

    numeric_mask = ~data["_is_missing"]
    cut = pd.cut(data.loc[numeric_mask, "_feature_value"], bins=bins, labels=False, include_lowest=True, right=True)
    for raw_bin_idx, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        order = numeric_start + raw_bin_idx
        mask = numeric_mask & (cut == raw_bin_idx)
        data.loc[mask, "_bin_order"] = order
        data.loc[mask, "_bin_label"] = _format_bin_label(lower, upper)
        data.loc[mask, "_lower_bound"] = lower
        data.loc[mask, "_upper_bound"] = upper

    data = data[data["_bin_order"].notna()].copy()
    if data.empty:
        return _skip_frame(feature, rank, gain, split_importance, "no_rows_after_binning")

    split_values = _ordered_splits(data[split_col].dropna().unique().tolist(), base_split_value)
    bin_specs = (
        data[["_bin_order", "_bin_label", "_lower_bound", "_upper_bound"]]
        .drop_duplicates()
        .sort_values("_bin_order")
        .to_dict("records")
    )
    n_effective_bins = len(bin_specs)
    rows: list[dict[str, Any]] = []
    for split_value in split_values:
        split_data = data[data[split_col] == split_value]
        total_good = int((split_data[label_col] == 0).sum())
        total_bad = int((split_data[label_col] == 1).sum())
        good_denom = total_good + smooth * n_effective_bins
        bad_denom = total_bad + smooth * n_effective_bins
        split_total = len(split_data)
        for spec in bin_specs:
            group = split_data[split_data["_bin_order"] == spec["_bin_order"]]
            good = int((group[label_col] == 0).sum())
            bad = int((group[label_col] == 1).sum())
            total = good + bad
            good_dist = (good + smooth) / good_denom if good_denom else np.nan
            bad_dist = (bad + smooth) / bad_denom if bad_denom else np.nan
            woe = float(np.log(bad_dist / good_dist)) if good_dist > 0 and bad_dist > 0 else np.nan
            rows.append(
                {
                    "feature": feature,
                    "rank": int(rank),
                    "gain": float(gain),
                    "split_importance": split_importance,
                    "bin_order": int(spec["_bin_order"]),
                    "bin_label": spec["_bin_label"],
                    "lower_bound": spec["_lower_bound"],
                    "upper_bound": spec["_upper_bound"],
                    "is_missing_bin": bool(spec["_bin_label"] == "Missing"),
                    "split_value": split_value,
                    "good": good,
                    "bad": bad,
                    "total": total,
                    "bad_rate": bad / total if total else np.nan,
                    "pop_pct": total / split_total if split_total else 0.0,
                    "woe": woe,
                    "iv_component": (bad_dist - good_dist) * woe if not np.isnan(woe) else np.nan,
                    "status": "ok",
                    "skip_reason": "",
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def generate_top_feature_woe(
    df: pd.DataFrame,
    importance: pd.DataFrame,
    *,
    output_dir: str | Path,
    label_col: str,
    split_col: str,
    top_n: int = 20,
    n_bins: int = 10,
    base_split_value: Any = "DEV",
    missing_values: Iterable[Any] | None = None,
) -> WoeGenerationResult:
    """Write Top feature WOE summary and PNG charts."""
    output_path = Path(output_dir)
    image_dir = output_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    top_features = select_top_features(importance, top_n=top_n)
    summary_frames: list[pd.DataFrame] = []
    image_paths: list[Path] = []
    for item in top_features.to_dict("records"):
        table = compute_woe_table(
            df,
            feature=str(item["feature"]),
            rank=int(item["rank"]),
            gain=float(item.get("gain", 0.0)),
            split_importance=item.get("split"),
            label_col=label_col,
            split_col=split_col,
            base_split_value=base_split_value,
            n_bins=n_bins,
            missing_values=missing_values,
        )
        summary_frames.append(table)
        if not table.empty and table["status"].eq("ok").any():
            image_path = image_dir / f"{int(item['rank']):03d}_{sanitize_feature_filename(str(item['feature']))}_WOE.png"
            try:
                plot_woe_chart(table, output_path=image_path, base_split_value=base_split_value)
            except ModuleNotFoundError as exc:
                if exc.name != "matplotlib":
                    raise
            else:
                image_paths.append(image_path)

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary_path = output_path / f"woe_top{top_n}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return WoeGenerationResult(output_dir=output_path, summary_path=summary_path, image_paths=image_paths)


def plot_woe_chart(table: pd.DataFrame, *, output_path: str | Path, base_split_value: Any = "DEV") -> Path:
    """Render one WOE chart from a long-format WOE table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = table[table["status"] == "ok"].copy()
    feature = str(ok["feature"].iloc[0])
    rank = int(ok["rank"].iloc[0])
    gain = float(ok["gain"].iloc[0])
    total_iv = ok.groupby("bin_label", as_index=False)["iv_component"].first()["iv_component"].sum()
    base = ok[ok["split_value"] == base_split_value]
    missing_rate = float(base.loc[base["is_missing_bin"], "pop_pct"].sum()) if not base.empty else 0.0

    bins = ok[["bin_order", "bin_label"]].drop_duplicates().sort_values("bin_order")
    x_pos = np.arange(len(bins))
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx()
    colors = ["dodgerblue", "darkorange", "forestgreen", "crimson", "purple", "sienna", "gold", "teal"]
    split_values = _ordered_splits(ok["split_value"].dropna().unique().tolist(), base_split_value)
    bar_width = 0.8 / max(len(split_values), 1)

    for idx, split_value in enumerate(split_values):
        subset = ok[ok["split_value"] == split_value].sort_values("bin_order")
        color = colors[idx % len(colors)]
        offset = bar_width * (idx - (len(split_values) - 1) / 2)
        ax2.bar(x_pos + offset, subset["pop_pct"], width=bar_width, color=color, alpha=0.35, label=f"{split_value} Pop %")
        ax1.plot(x_pos, subset["woe"], marker="o", linestyle="-", color=color, label=f"{split_value} WOE")

    ax1.set_title(f"Top {rank} WOE: {feature}\nGain={gain:.3f}  IV={total_iv:.4f}  Missing({base_split_value})={missing_rate:.1%}", fontsize=14)
    ax1.set_ylabel("WOE = ln(bad_dist / good_dist)")
    ax2.set_ylabel("Population Percentage")
    ax1.set_xlabel("Feature Bins")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bins["bin_label"], rotation=45, ha="right")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _coerce_feature_values(series: pd.Series, missing_values: Iterable[Any] | None) -> pd.Series:
    result = series.copy()
    if missing_values is not None:
        result = result.replace(list(missing_values), np.nan)
    return pd.to_numeric(result, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _build_numeric_bins(series: pd.Series, n_bins: int) -> np.ndarray | None:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.nunique() < 2:
        return None
    try:
        bins = pd.qcut(valid, q=n_bins, retbins=True, duplicates="drop")[1]
    except ValueError:
        try:
            bins = pd.cut(valid, bins=n_bins, retbins=True, duplicates="drop")[1]
        except ValueError:
            return None
    bins = np.unique(np.asarray(bins, dtype=float))
    if len(bins) < 3:
        return None
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def _format_bin_label(lower: float, upper: float) -> str:
    lower_text = "-inf" if np.isneginf(lower) else f"{lower:.6g}"
    upper_text = "inf" if np.isposinf(upper) else f"{upper:.6g}"
    return f"({lower_text}, {upper_text}]"


def _ordered_splits(values: list[Any], base_split_value: Any) -> list[Any]:
    preferred = [base_split_value, "OOT", "DEV-OOS", "OOT-OOS"]
    ordered = [value for value in preferred if value in values]
    ordered.extend(sorted([value for value in values if value not in ordered], key=str))
    return ordered


def _skip_frame(feature: str, rank: int, gain: float, split_importance: Any, reason: str) -> pd.DataFrame:
    row = {column: np.nan for column in SUMMARY_COLUMNS}
    row.update(
        {
            "feature": feature,
            "rank": int(rank),
            "gain": float(gain),
            "split_importance": split_importance,
            "status": "skipped",
            "skip_reason": reason,
        }
    )
    return pd.DataFrame([row], columns=SUMMARY_COLUMNS)
