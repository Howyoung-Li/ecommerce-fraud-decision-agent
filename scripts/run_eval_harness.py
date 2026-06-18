from ecommerce_fraud_decision_agent.eval_harness import write_harness_report


def main() -> None:
    path = write_harness_report()
    print(f"harness_report: {path}")


if __name__ == "__main__":
    main()

