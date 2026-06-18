import unittest

from ecommerce_fraud_decision_agent.eval_harness import run_harness


class EvalHarnessTests(unittest.TestCase):
    def test_harness_passes_all_current_contracts(self) -> None:
        report = run_harness()

        self.assertEqual(report["total"], 6)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

