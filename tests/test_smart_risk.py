import unittest

from ecommerce_fraud_decision_agent.smart_risk import (
    generate_transaction,
    judge_transaction,
)


class SmartRiskTests(unittest.TestCase):
    def test_generated_fast_anomaly_case_has_auditable_reason(self) -> None:
        case = generate_transaction("synthetic_fast_anomaly", seed=13)
        judgment = judge_transaction(case)
        codes = {reason["code"] for reason in judgment["reason_codes"]}
        citation_codes = {citation["code"] for citation in judgment["policy_citations"]}

        self.assertIn("FAST_PURCHASE_ANOMALY", codes)
        self.assertIn("FAST_PURCHASE_ANOMALY", citation_codes)
        self.assertEqual(judgment["audit_context"]["fast_purchase_le_1s_fraud_rate"], 1.0)
        self.assertIn(
            judgment["decision"],
            {
                "manual_review",
                "step_up_verification",
                "decline",
            },
        )

    def test_normal_case_not_auto_declined(self) -> None:
        case = generate_transaction("normal_new_user", seed=7)
        judgment = judge_transaction(case)

        self.assertNotEqual(judgment["decision"], "decline")


if __name__ == "__main__":
    unittest.main()
