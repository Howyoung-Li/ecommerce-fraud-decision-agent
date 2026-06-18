import unittest

import pandas as pd

from ecommerce_fraud_decision_agent.features import add_asof_history_features


class FeatureEngineeringTests(unittest.TestCase):
    def test_asof_history_counts_use_prior_events_only(self) -> None:
        data = pd.DataFrame(
            {
                "user_id": [3, 1, 2, 4],
                "purchase_time": pd.to_datetime(
                    [
                        "2025-01-03 00:00:00",
                        "2025-01-01 00:00:00",
                        "2025-01-02 00:00:00",
                        "2025-01-04 00:00:00",
                    ]
                ),
                "device_id": ["dev_a", "dev_a", "dev_a", "dev_b"],
                "ip_address": [10, 10, 11, 10],
            }
        )

        features = add_asof_history_features(data)

        self.assertEqual(features["device_seen_count_hist"].tolist(), [0, 1, 2, 0])
        self.assertEqual(features["ip_seen_count_hist"].tolist(), [0, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()

