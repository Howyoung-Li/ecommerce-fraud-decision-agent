from ecommerce_fraud_decision_agent.data_audit import write_data_audit_outputs


def main() -> None:
    outputs = write_data_audit_outputs()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

