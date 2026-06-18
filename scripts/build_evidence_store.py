from ecommerce_fraud_decision_agent.evidence_store import write_evidence_store


def main() -> None:
    path = write_evidence_store()
    print(f"evidence_store: {path}")


if __name__ == "__main__":
    main()

