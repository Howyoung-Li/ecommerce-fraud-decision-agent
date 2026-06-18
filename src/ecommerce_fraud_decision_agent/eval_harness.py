from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import AGENT_ARTIFACTS_DIR
from .smart_risk import generate_transaction, judge_transaction


@dataclass(frozen=True)
class HarnessCase:
    name: str
    scenario: str
    seed: int
    expected_reason_codes: set[str]
    allowed_decisions: set[str]
    require_model_score: bool = True


HARNESS_CASES = [
    HarnessCase(
        name="normal model-elevated case",
        scenario="normal_new_user",
        seed=7,
        expected_reason_codes={"MODEL_SCORE_ELEVATED"},
        allowed_decisions={"step_up_verification", "manual_review"},
    ),
    HarnessCase(
        name="fast purchase suspicious case",
        scenario="fast_purchase_suspicious",
        seed=11,
        expected_reason_codes={"FAST_PURCHASE_AFTER_SIGNUP"},
        allowed_decisions={"step_up_verification", "manual_review"},
    ),
    HarnessCase(
        name="one-second anomaly case",
        scenario="synthetic_fast_anomaly",
        seed=13,
        expected_reason_codes={"FAST_PURCHASE_ANOMALY"},
        allowed_decisions={"manual_review", "decline"},
    ),
    HarnessCase(
        name="reused device abuse case",
        scenario="reused_device_abuse",
        seed=17,
        expected_reason_codes={"DEVICE_REUSE"},
        allowed_decisions={"step_up_verification", "manual_review", "decline"},
    ),
    HarnessCase(
        name="reused ip attack case",
        scenario="reused_ip_attack",
        seed=23,
        expected_reason_codes={"IP_REUSE"},
        allowed_decisions={"step_up_verification", "manual_review", "decline"},
    ),
    HarnessCase(
        name="borderline mixed signal case",
        scenario="borderline_mixed_signal",
        seed=19,
        expected_reason_codes={"DEVICE_REUSE_WEAK"},
        allowed_decisions={"approve", "step_up_verification"},
    ),
]


def _check_judgment(case: HarnessCase, judgment: dict[str, object]) -> list[str]:
    failures: list[str] = []
    reason_codes = {reason["code"] for reason in judgment["reason_codes"]}
    citation_codes = {citation["code"] for citation in judgment["policy_citations"]}
    trace_steps = {step["step"] for step in judgment["trace"]}

    missing_codes = case.expected_reason_codes - reason_codes
    if missing_codes:
        failures.append(f"missing reason codes: {sorted(missing_codes)}")

    if judgment["decision"] not in case.allowed_decisions:
        failures.append(
            f"decision {judgment['decision']} not in {sorted(case.allowed_decisions)}"
        )

    if "SYNTHETIC_DATA_CAVEAT" not in citation_codes:
        failures.append("missing synthetic data caveat citation")

    for citation in judgment["policy_citations"]:
        for field in ["source_name", "source_path", "section", "retrieval_score"]:
            if field not in citation:
                failures.append(f"citation {citation['code']} missing {field}")

    citation_by_code = {citation["code"]: citation for citation in judgment["policy_citations"]}
    expected_sections = {
        "SYNTHETIC_DATA_CAVEAT": {"Synthetic Data Caveat"},
        "FAST_PURCHASE_ANOMALY": {"Fast Purchase Behavior", "Fast Purchase Anomaly"},
        "DEVICE_REUSE": {"Device And IP Reuse"},
        "IP_REUSE": {"Device And IP Reuse"},
        "HIGH_PURCHASE_VALUE": {"Purchase Value And Night Activity"},
        "NIGHT_PURCHASE": {"Purchase Value And Night Activity"},
        "MODEL_SCORE_ELEVATED": {"Model Score Usage", "Interpretation"},
    }
    for code in citation_codes:
        expected = expected_sections.get(code)
        actual = citation_by_code[code].get("section")
        if expected and actual not in expected:
            failures.append(
                f"citation {code} expected section {sorted(expected)}, got {actual}"
            )

    if not judgment.get("limitations"):
        failures.append("missing limitation text")

    if case.require_model_score and judgment.get("model_score") is None:
        failures.append("missing trained model score")

    if not (0 <= judgment["final_risk_score"] <= 1):
        failures.append("final_risk_score outside [0, 1]")

    required_trace = {
        "generate_transaction",
        "build_reason_codes",
        "estimate_rule_score",
        "score_with_trained_model",
        "blend_scores",
        "retrieve_policy_citations",
        "compose_ai_risk_summary",
    }
    missing_trace = required_trace - trace_steps
    if missing_trace:
        failures.append(f"missing trace steps: {sorted(missing_trace)}")

    for reason in judgment["reason_codes"]:
        evidence = reason.get("evidence", "")
        if "=" not in evidence:
            failures.append(f"reason {reason['code']} lacks value evidence")

    return failures


def run_harness() -> dict[str, object]:
    results = []
    pass_count = 0
    for case in HARNESS_CASES:
        judgment = judge_transaction(generate_transaction(case.scenario, case.seed))
        failures = _check_judgment(case, judgment)
        passed = not failures
        pass_count += int(passed)
        results.append(
            {
                "name": case.name,
                "scenario": case.scenario,
                "seed": case.seed,
                "passed": passed,
                "failures": failures,
                "decision": judgment["decision"],
                "risk_level": judgment["risk_level"],
                "reason_codes": [reason["code"] for reason in judgment["reason_codes"]],
                "model_score": judgment["model_score"],
                "final_risk_score": judgment["final_risk_score"],
            }
        )

    return {
        "total": len(results),
        "passed": pass_count,
        "failed": len(results) - pass_count,
        "pass_rate": pass_count / len(results) if results else 0,
        "results": results,
    }


def write_harness_report(output_dir: Path = AGENT_ARTIFACTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_harness()
    path = output_dir / "eval_harness_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
