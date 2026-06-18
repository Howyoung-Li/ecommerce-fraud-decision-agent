import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS_DATA_DIR = ROOT / "docs" / "data"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_json(ROOT / "artifacts" / "modeling" / "model_metrics.json")
    data_quality = load_json(ROOT / "artifacts" / "data" / "data_quality_report.json")
    cases = load_json(ROOT / "artifacts" / "agent" / "generated_case_judgments.json")
    harness = load_json(ROOT / "artifacts" / "agent" / "eval_harness_report.json")
    topk = pd.read_csv(ROOT / "artifacts" / "policy" / "topk_review_strategy.csv")

    xgb_test_topk = topk[
        topk["model_name"].eq("xgboost") & topk["evaluation_split"].eq("test")
    ].to_dict(orient="records")

    bundle = {
        "project": {
            "name": "E-commerce Fraud Decision Agent",
            "positioning": "AI-first smart risk decision simulator with leakage-aware OOT validation.",
        },
        "data_quality": {
            "rows": data_quality["dataset"]["rows"],
            "fraud_rate": data_quality["dataset"]["fraud_rate"],
            "purchase_time_min": data_quality["dataset"]["purchase_time_min"],
            "purchase_time_max": data_quality["dataset"]["purchase_time_max"],
            "fast_purchase": data_quality["fast_purchase_observation"],
            "time_splits": data_quality["time_splits"],
        },
        "modeling": {
            "metrics": metrics,
            "xgboost_test_topk": xgb_test_topk,
            "leakage_position": (
                "Full-data repeated IP/device signals are used only for EDA. "
                "The model uses as-of historical counts and OOT validation."
            ),
        },
        "agent": {
            "cases": cases,
            "harness": harness,
        },
    }

    out = DOCS_DATA_DIR / "workbench.json"
    out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"workbench_data: {out}")


if __name__ == "__main__":
    main()

