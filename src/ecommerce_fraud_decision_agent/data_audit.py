from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA_ARTIFACTS_DIR, DATA_RAW_DIR


@dataclass(frozen=True)
class SegmentStats:
    name: str
    rows: int
    fraud_rows: int
    fraud_rate: float
    purchase_start: str
    purchase_end: str
    fast_purchase_rows: int
    fast_purchase_fraud_rate: float | None


def load_fraud_data(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    data = pd.read_csv(raw_dir / "Fraud_Data.csv")
    data["signup_time"] = pd.to_datetime(data["signup_time"])
    data["purchase_time"] = pd.to_datetime(data["purchase_time"])
    data["signup_to_purchase_seconds"] = (
        data["purchase_time"] - data["signup_time"]
    ).dt.total_seconds()
    data["is_fast_purchase_le_1s"] = data["signup_to_purchase_seconds"] <= 1
    return data


def load_ip_country(raw_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "IpAddress_to_Country.csv")


def map_country(data: pd.DataFrame, ip_country: pd.DataFrame) -> pd.Series:
    ranges = ip_country.sort_values("lower_bound_ip_address").reset_index(drop=True)
    lower = ranges["lower_bound_ip_address"].to_numpy()
    upper = ranges["upper_bound_ip_address"].to_numpy()
    countries = ranges["country"].to_numpy()

    ips = data["ip_address"].to_numpy()
    idx = np.searchsorted(lower, ips, side="right") - 1
    valid_idx = idx >= 0
    clipped = np.clip(idx, 0, len(upper) - 1)
    matched = valid_idx & (ips <= upper[clipped])

    out = np.full(len(data), "UNKNOWN", dtype=object)
    out[matched] = countries[clipped[matched]]
    return pd.Series(out, index=data.index, name="country")


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _segment_stats(name: str, segment: pd.DataFrame) -> SegmentStats:
    fast = segment[segment["is_fast_purchase_le_1s"]]
    return SegmentStats(
        name=name,
        rows=int(len(segment)),
        fraud_rows=int(segment["class"].sum()),
        fraud_rate=float(segment["class"].mean()),
        purchase_start=str(segment["purchase_time"].min()),
        purchase_end=str(segment["purchase_time"].max()),
        fast_purchase_rows=int(len(fast)),
        fast_purchase_fraud_rate=None if fast.empty else float(fast["class"].mean()),
    )


def time_split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = data.sort_values("purchase_time").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return {
        "train": ordered.iloc[:train_end].copy(),
        "valid": ordered.iloc[train_end:valid_end].copy(),
        "test": ordered.iloc[valid_end:].copy(),
    }


def anomaly_relationship_table(data: pd.DataFrame) -> pd.DataFrame:
    base_rate = data["class"].mean()
    rows: list[dict[str, object]] = []

    checks = {
        "FAST_PURCHASE_LE_1S": data["signup_to_purchase_seconds"] <= 1,
        "FAST_PURCHASE_LE_2MIN": data["signup_to_purchase_seconds"] <= 120,
        "FAST_PURCHASE_LE_15MIN": data["signup_to_purchase_seconds"] <= 900,
        "REPEATED_DEVICE_FULL_DATA_EDA": data["device_id"].duplicated(keep=False),
        "REPEATED_IP_FULL_DATA_EDA": data["ip_address"].duplicated(keep=False),
        "UNMAPPED_IP_COUNTRY": data["country"].eq("UNKNOWN"),
    }

    for signal, mask in checks.items():
        subset = data[mask]
        fraud_rate = _rate(subset["class"].sum(), len(subset))
        rows.append(
            {
                "signal": signal,
                "rows": int(mask.sum()),
                "row_share": float(mask.mean()),
                "fraud_rows": int(subset["class"].sum()),
                "fraud_rate": fraud_rate,
                "baseline_fraud_rate": float(base_rate),
                "fraud_lift_vs_baseline": None if fraud_rate is None else fraud_rate / base_rate,
                "usage": (
                    "EDA/correlation evidence only; full-data repeated ID signals must not enter production model"
                    if "FULL_DATA_EDA" in signal
                    else "Candidate rule or monitoring evidence"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_data_quality_report(raw_dir: Path = DATA_RAW_DIR) -> dict[str, object]:
    data = load_fraud_data(raw_dir)
    ip_country = load_ip_country(raw_dir)
    data["country"] = map_country(data, ip_country)
    splits = time_split(data)
    anomaly_table = anomaly_relationship_table(data)

    fast = data[data["is_fast_purchase_le_1s"]]
    report: dict[str, object] = {
        "dataset": {
            "rows": int(len(data)),
            "columns": list(data.columns),
            "fraud_rows": int(data["class"].sum()),
            "legit_rows": int((data["class"] == 0).sum()),
            "fraud_rate": float(data["class"].mean()),
            "purchase_time_min": str(data["purchase_time"].min()),
            "purchase_time_max": str(data["purchase_time"].max()),
            "signup_time_min": str(data["signup_time"].min()),
            "signup_time_max": str(data["signup_time"].max()),
            "unmapped_country_rows": int(data["country"].eq("UNKNOWN").sum()),
            "unmapped_country_rate": float(data["country"].eq("UNKNOWN").mean()),
        },
        "fast_purchase_observation": {
            "definition": "signup_to_purchase_seconds <= 1",
            "business_hypothesis": (
                "A purchase completed within one second after signup is operationally abnormal "
                "and may indicate automation, scripted abuse, or synthetic-data generation."
            ),
            "relationship_to_label": {
                "rows": int(len(fast)),
                "fraud_rows": int(fast["class"].sum()),
                "fraud_rate": None if fast.empty else float(fast["class"].mean()),
                "baseline_fraud_rate": float(data["class"].mean()),
                "lift_vs_baseline": None
                if fast.empty
                else float(fast["class"].mean() / data["class"].mean()),
            },
            "interpretation": (
                "Treat this as a distributional anomaly that is strongly associated with fraud in this dataset. "
                "It can support a rule baseline and AI explanation, but should be monitored and validated "
                "before being generalized to production."
            ),
        },
        "time_splits": [asdict(_segment_stats(name, split)) for name, split in splits.items()],
        "anomaly_relationships": anomaly_table.to_dict(orient="records"),
    }
    return report


def write_data_audit_outputs(output_dir: Path = DATA_ARTIFACTS_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_data_quality_report()
    anomaly_table = pd.DataFrame(report["anomaly_relationships"])

    report_path = output_dir / "data_quality_report.json"
    anomaly_path = output_dir / "anomaly_relationship_table.csv"
    markdown_path = output_dir / "synthetic_and_anomaly_audit.md"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    anomaly_table.to_csv(anomaly_path, index=False)
    markdown_path.write_text(render_audit_markdown(report), encoding="utf-8")
    return {
        "report": report_path,
        "anomaly_table": anomaly_path,
        "markdown": markdown_path,
    }


def render_audit_markdown(report: dict[str, object]) -> str:
    dataset = report["dataset"]
    fast = report["fast_purchase_observation"]
    lines = [
        "# Data Quality And Anomaly Audit",
        "",
        "## Dataset Snapshot",
        "",
        f"- Rows: {dataset['rows']:,}",
        f"- Fraud rows: {dataset['fraud_rows']:,}",
        f"- Fraud rate: {dataset['fraud_rate']:.4f}",
        f"- Purchase period: {dataset['purchase_time_min']} to {dataset['purchase_time_max']}",
        f"- Unmapped country rows: {dataset['unmapped_country_rows']:,} ({dataset['unmapped_country_rate']:.2%})",
        "",
        "## Fast Purchase Observation",
        "",
        f"- Definition: `{fast['definition']}`",
        f"- Business hypothesis: {fast['business_hypothesis']}",
        f"- Rows: {fast['relationship_to_label']['rows']:,}",
        f"- Fraud rate among these rows: {fast['relationship_to_label']['fraud_rate']:.4f}",
        f"- Baseline fraud rate: {fast['relationship_to_label']['baseline_fraud_rate']:.4f}",
        f"- Lift vs baseline: {fast['relationship_to_label']['lift_vs_baseline']:.2f}x",
        "",
        "Interpretation:",
        "",
        fast["interpretation"],
        "",
        "## Time Split Drift",
        "",
        "| Split | Rows | Fraud Rate | Fast Purchase Rows | Fast Purchase Fraud Rate | Period |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in report["time_splits"]:
        fast_rate = split["fast_purchase_fraud_rate"]
        fast_rate_text = "NA" if fast_rate is None else f"{fast_rate:.4f}"
        lines.append(
            f"| {split['name']} | {split['rows']:,} | {split['fraud_rate']:.4f} | "
            f"{split['fast_purchase_rows']:,} | {fast_rate_text} | "
            f"{split['purchase_start']} to {split['purchase_end']} |"
        )
    lines.extend(
        [
            "",
            "## Modeling Implication",
            "",
            "- The `<=1s` behavior should enter the project first as an anomaly-correlation finding.",
            "- It can become a rules-only baseline and reason code, but the ML model should also be evaluated without relying solely on this signal.",
            "- Full-data repeated device/IP indicators are useful for EDA but must be rebuilt as historical as-of features before modeling.",
        ]
    )
    return "\n".join(lines) + "\n"

