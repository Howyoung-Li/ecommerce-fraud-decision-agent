from ecommerce_fraud_decision_agent.modeling import train_and_write_outputs


def main() -> None:
    outputs = train_and_write_outputs()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

