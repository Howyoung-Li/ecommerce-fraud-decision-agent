from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from .data_audit import build_data_quality_report
from .evidence_store import retrieve_evidence, write_evidence_store
from .features import feature_columns
from .paths import AGENT_ARTIFACTS_DIR, MODELING_ARTIFACTS_DIR


Decision = Literal["approve", "manual_review", "step_up_verification", "decline"]
RiskLevel = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class GeneratedTransaction:
    case_id: str
    scenario: str
    signup_to_purchase_seconds: int
    purchase_value: int
    device_seen_user_count_hist: int
    ip_seen_user_count_hist: int
    source: str
    browser: str
    country: str
    purchase_hour: int
    is_generated_case: bool = True


@dataclass(frozen=True)
class ReasonCode:
    code: str
    evidence: str
    source: str
    severity: Literal["low", "medium", "high", "critical"]


SCENARIOS: dict[str, dict[str, object]] = {
    "normal_new_user": {
        "signup_seconds": (2 * 24 * 3600, 45 * 24 * 3600),
        "purchase_value": (15, 90),
        "device_count": (1, 1),
        "ip_count": (1, 1),
        "hours": [10, 11, 13, 15, 20],
    },
    "fast_purchase_suspicious": {
        "signup_seconds": (2, 120),
        "purchase_value": (50, 250),
        "device_count": (1, 3),
        "ip_count": (1, 3),
        "hours": [0, 1, 2, 23],
    },
    "synthetic_fast_anomaly": {
        "signup_seconds": (1, 1),
        "purchase_value": (10, 180),
        "device_count": (1, 4),
        "ip_count": (1, 4),
        "hours": [0, 2, 4, 22],
    },
    "reused_device_abuse": {
        "signup_seconds": (300, 7200),
        "purchase_value": (70, 350),
        "device_count": (5, 14),
        "ip_count": (1, 5),
        "hours": [1, 2, 3, 21, 23],
    },
    "reused_ip_attack": {
        "signup_seconds": (300, 7200),
        "purchase_value": (30, 260),
        "device_count": (1, 4),
        "ip_count": (5, 16),
        "hours": [0, 1, 2, 22, 23],
    },
    "borderline_mixed_signal": {
        "signup_seconds": (900, 4 * 3600),
        "purchase_value": (80, 180),
        "device_count": (2, 4),
        "ip_count": (1, 3),
        "hours": [8, 12, 18, 23],
    },
}


POLICY_CITATIONS = {
    "FAST_PURCHASE_ANOMALY": (
        "Very short signup-to-purchase intervals are abnormal for first-purchase risk review. "
        "In the dataset audit, <=1 second purchases are strongly associated with fraud and should be reviewed as an anomaly signal."
    ),
    "DEVICE_REUSE": (
        "Multiple historical users sharing the same device can indicate account farming or scripted abuse. "
        "Use as-of historical device counts rather than full-data leakage features."
    ),
    "IP_REUSE": (
        "Multiple historical users sharing the same IP can indicate coordinated abuse. "
        "Use as-of historical IP counts and combine this signal with transaction context."
    ),
    "HIGH_PURCHASE_VALUE": (
        "Higher purchase value increases exposure and should raise review priority when combined with fast purchase or reuse signals."
    ),
    "NIGHT_PURCHASE": (
        "Night-time purchases are not fraud by themselves, but can increase review priority when other risk signals co-occur."
    ),
    "SYNTHETIC_DATA_CAVEAT": (
        "This demo uses public data with synthetic-like anomalies. The AI judgment is a controlled risk simulation, not a real-user fraud decision."
    ),
    "MODEL_SCORE_ELEVATED": (
        "An elevated model score can increase review priority, but should be interpreted with reason codes, policy context, and OOT validation caveats."
    ),
}

EVIDENCE_QUERIES = {
    "SYNTHETIC_DATA_CAVEAT": "synthetic data caveat generated case judgments controlled simulations real-user fraud decisions",
    "FAST_PURCHASE_ANOMALY": "fast purchase <=1 second anomaly fraud label relationship",
    "DEVICE_REUSE": "device reuse as-of historical counts account farming leakage",
    "IP_REUSE": "ip reuse as-of historical counts coordinated abuse leakage",
    "HIGH_PURCHASE_VALUE": "high purchase value fraud exposure review priority combined risk indicators",
    "NIGHT_PURCHASE": "night-time purchase weak contextual signal other risk indicators co-occur",
    "MODEL_SCORE_ELEVATED": "elevated model score manual review step-up verification not automatic hard decline",
}


def generate_transaction(
    scenario: str | None = None, seed: int | None = None
) -> GeneratedTransaction:
    rng = random.Random(seed)
    scenario_name = scenario or rng.choice(list(SCENARIOS))
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    config = SCENARIOS[scenario_name]
    signup_low, signup_high = config["signup_seconds"]
    value_low, value_high = config["purchase_value"]
    device_low, device_high = config["device_count"]
    ip_low, ip_high = config["ip_count"]

    return GeneratedTransaction(
        case_id=f"sim_{rng.randint(100000, 999999)}",
        scenario=scenario_name,
        signup_to_purchase_seconds=rng.randint(signup_low, signup_high),
        purchase_value=rng.randint(value_low, value_high),
        device_seen_user_count_hist=rng.randint(device_low, device_high),
        ip_seen_user_count_hist=rng.randint(ip_low, ip_high),
        source=rng.choice(["SEO", "Ads", "Direct"]),
        browser=rng.choice(["Chrome", "FireFox", "IE", "Opera", "Safari"]),
        country=rng.choice(["United States", "China", "Japan", "United Kingdom", "UNKNOWN"]),
        purchase_hour=rng.choice(config["hours"]),
    )


def build_reason_codes(case: GeneratedTransaction) -> list[ReasonCode]:
    reasons: list[ReasonCode] = []

    if case.signup_to_purchase_seconds <= 1:
        reasons.append(
            ReasonCode(
                code="FAST_PURCHASE_ANOMALY",
                evidence=(
                    f"signup_to_purchase_seconds={case.signup_to_purchase_seconds}; "
                    "dataset audit shows <=1 second purchases are highly concentrated in fraud labels"
                ),
                source="data_audit + business_anomaly_hypothesis",
                severity="critical",
            )
        )
    elif case.signup_to_purchase_seconds <= 120:
        reasons.append(
            ReasonCode(
                code="FAST_PURCHASE_AFTER_SIGNUP",
                evidence=f"signup_to_purchase_seconds={case.signup_to_purchase_seconds}",
                source="feature_rule",
                severity="high",
            )
        )

    if case.device_seen_user_count_hist >= 5:
        reasons.append(
            ReasonCode(
                code="DEVICE_REUSE",
                evidence=f"device_seen_user_count_hist={case.device_seen_user_count_hist}",
                source="historical_feature",
                severity="high",
            )
        )
    elif case.device_seen_user_count_hist >= 2:
        reasons.append(
            ReasonCode(
                code="DEVICE_REUSE_WEAK",
                evidence=f"device_seen_user_count_hist={case.device_seen_user_count_hist}",
                source="historical_feature",
                severity="medium",
            )
        )

    if case.ip_seen_user_count_hist >= 5:
        reasons.append(
            ReasonCode(
                code="IP_REUSE",
                evidence=f"ip_seen_user_count_hist={case.ip_seen_user_count_hist}",
                source="historical_feature",
                severity="high",
            )
        )
    elif case.ip_seen_user_count_hist >= 2:
        reasons.append(
            ReasonCode(
                code="IP_REUSE_WEAK",
                evidence=f"ip_seen_user_count_hist={case.ip_seen_user_count_hist}",
                source="historical_feature",
                severity="medium",
            )
        )

    if case.purchase_value >= 200:
        reasons.append(
            ReasonCode(
                code="HIGH_PURCHASE_VALUE",
                evidence=f"purchase_value={case.purchase_value}",
                source="transaction_feature",
                severity="medium",
            )
        )

    if case.purchase_hour <= 5 or case.purchase_hour >= 23:
        reasons.append(
            ReasonCode(
                code="NIGHT_PURCHASE",
                evidence=f"purchase_hour={case.purchase_hour}",
                source="transaction_feature",
                severity="low",
            )
        )

    if case.country == "UNKNOWN":
        reasons.append(
            ReasonCode(
                code="UNMAPPED_IP_COUNTRY",
                evidence="country=UNKNOWN after IP mapping",
                source="data_quality_feature",
                severity="low",
            )
        )

    return reasons


def estimate_rule_score(reasons: list[ReasonCode]) -> float:
    weights = {"low": 0.08, "medium": 0.18, "high": 0.30, "critical": 0.70}
    score = sum(weights[reason.severity] for reason in reasons)
    return min(score, 0.99)


def blend_scores(rule_score: float, model_score: float | None) -> float:
    if model_score is None:
        return rule_score
    return min(0.99, 0.55 * rule_score + 0.45 * model_score)


def decide(score: float, reasons: list[ReasonCode]) -> tuple[Decision, RiskLevel]:
    codes = {reason.code for reason in reasons}
    if "FAST_PURCHASE_ANOMALY" in codes and ("DEVICE_REUSE" in codes or "IP_REUSE" in codes):
        return "decline", "critical"
    if score >= 0.70:
        return "manual_review", "high"
    if score >= 0.40:
        return "step_up_verification", "medium"
    return "approve", "low"


def add_model_score_reason(
    reasons: list[ReasonCode], model_score: float | None
) -> list[ReasonCode]:
    if model_score is None or model_score < 0.65:
        return reasons
    return reasons + [
        ReasonCode(
            code="MODEL_SCORE_ELEVATED",
            evidence=f"xgboost_model_score={model_score:.4f}",
            source="trained_model",
            severity="medium",
        )
    ]


def generated_case_feature_frame(case: GeneratedTransaction) -> pd.DataFrame:
    row = {
        "signup_to_purchase_seconds": case.signup_to_purchase_seconds,
        "signup_to_purchase_minutes": case.signup_to_purchase_seconds / 60,
        "purchase_value": case.purchase_value,
        "purchase_value_log": math.log1p(case.purchase_value),
        "purchase_hour": case.purchase_hour,
        "purchase_dayofweek": 0,
        "is_night_purchase": int(case.purchase_hour <= 5 or case.purchase_hour >= 23),
        "is_fast_purchase_le_1s": int(case.signup_to_purchase_seconds <= 1),
        "is_flash_purchase_le_2min": int(case.signup_to_purchase_seconds <= 120),
        "device_seen_count_hist": case.device_seen_user_count_hist,
        "ip_seen_count_hist": case.ip_seen_user_count_hist,
        "country_is_unknown": int(case.country == "UNKNOWN"),
        "source": case.source,
        "browser": case.browser,
        "sex": "unknown",
        "country": case.country,
    }
    frame = pd.DataFrame([row])
    return frame[feature_columns()]


def score_with_trained_model(case: GeneratedTransaction) -> float | None:
    model_path = MODELING_ARTIFACTS_DIR / "xgboost.joblib"
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    frame = generated_case_feature_frame(case)
    return float(model.predict_proba(frame)[:, 1][0])


def judge_transaction(case: GeneratedTransaction) -> dict[str, object]:
    report = build_data_quality_report()
    reasons = build_reason_codes(case)
    model_score = score_with_trained_model(case)
    reasons = add_model_score_reason(reasons, model_score)
    rule_score = estimate_rule_score(reasons)
    final_score = blend_scores(rule_score, model_score)
    decision, risk_level = decide(final_score, reasons)

    citation_codes = ["SYNTHETIC_DATA_CAVEAT"]
    for reason in reasons:
        if reason.code.startswith("FAST_PURCHASE"):
            citation_codes.append("FAST_PURCHASE_ANOMALY")
        elif reason.code.startswith("DEVICE_REUSE"):
            citation_codes.append("DEVICE_REUSE")
        elif reason.code.startswith("IP_REUSE"):
            citation_codes.append("IP_REUSE")
        elif reason.code in POLICY_CITATIONS:
            citation_codes.append(reason.code)
    citation_codes = list(dict.fromkeys(citation_codes))
    citations = retrieve_policy_citations(citation_codes)

    if not reasons:
        summary = "No major risk signals were triggered by the generated transaction features."
    else:
        summary = (
            f"The case is routed to {decision} because {len(reasons)} risk signal(s) were triggered: "
            + ", ".join(reason.code for reason in reasons)
            + "."
        )

    return {
        "case": asdict(case),
        "rule_score": round(rule_score, 4),
        "model_score": None if model_score is None else round(model_score, 4),
        "final_risk_score": round(final_score, 4),
        "decision": decision,
        "risk_level": risk_level,
        "reason_codes": [asdict(reason) for reason in reasons],
        "policy_citations": citations,
        "ai_risk_summary": summary,
        "limitations": (
            "This is an AI-first risk simulation based on public anonymized data and generated case features. "
            "The judgment is evidence-constrained and should not be treated as a real-user fraud decision."
        ),
        "trace": [
            {"step": "generate_transaction", "scenario": case.scenario},
            {"step": "build_reason_codes", "count": len(reasons)},
            {"step": "estimate_rule_score", "score": round(rule_score, 4)},
            {
                "step": "score_with_trained_model",
                "score": None if model_score is None else round(model_score, 4),
            },
            {"step": "blend_scores", "score": round(final_score, 4)},
            {"step": "retrieve_policy_citations", "citation_count": len(citations)},
            {"step": "compose_ai_risk_summary", "decision": decision},
        ],
        "audit_context": {
            "baseline_fraud_rate": report["dataset"]["fraud_rate"],
            "fast_purchase_le_1s_fraud_rate": report["fast_purchase_observation"][
                "relationship_to_label"
            ]["fraud_rate"],
            "fast_purchase_interpretation": report["fast_purchase_observation"][
                "interpretation"
            ],
        },
    }


def retrieve_policy_citations(citation_codes: list[str]) -> list[dict[str, object]]:
    write_evidence_store()
    citations = []
    for code in citation_codes:
        query = EVIDENCE_QUERIES.get(code, code)
        evidence = retrieve_evidence(query, top_k=1)[0]
        citations.append(
            {
                "code": code,
                "text": evidence["text"],
                "source_name": evidence["source_name"],
                "source_path": evidence["source_path"],
                "section": evidence["section"],
                "retrieval_score": round(evidence["score"], 4),
            }
        )
    return citations


def write_demo_cases(output_dir: Path = AGENT_ARTIFACTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        judge_transaction(generate_transaction("normal_new_user", seed=7)),
        judge_transaction(generate_transaction("fast_purchase_suspicious", seed=11)),
        judge_transaction(generate_transaction("synthetic_fast_anomaly", seed=13)),
        judge_transaction(generate_transaction("reused_device_abuse", seed=17)),
        judge_transaction(generate_transaction("reused_ip_attack", seed=23)),
        judge_transaction(generate_transaction("borderline_mixed_signal", seed=19)),
    ]
    path = output_dir / "generated_case_judgments.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
