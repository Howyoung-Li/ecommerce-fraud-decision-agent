import unittest

from ecommerce_fraud_decision_agent.data_audit import build_data_quality_report


class DataAuditTests(unittest.TestCase):
    def test_fast_purchase_observation_is_label_relationship(self) -> None:
        report = build_data_quality_report()
        observation = report["fast_purchase_observation"]

        self.assertEqual(observation["definition"], "signup_to_purchase_seconds <= 1")
        self.assertEqual(observation["relationship_to_label"]["rows"], 7600)
        self.assertEqual(observation["relationship_to_label"]["fraud_rate"], 1.0)
        self.assertIn("distributional anomaly", observation["interpretation"])

    def test_time_splits_are_present(self) -> None:
        report = build_data_quality_report()
        split_names = {split["name"] for split in report["time_splits"]}

        self.assertEqual(split_names, {"train", "valid", "test"})


if __name__ == "__main__":
    unittest.main()
