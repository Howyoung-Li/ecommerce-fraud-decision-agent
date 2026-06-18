import argparse
import json

from ecommerce_fraud_decision_agent.smart_risk import (
    generate_transaction,
    judge_transaction,
    write_demo_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-examples", action="store_true")
    args = parser.parse_args()

    if args.write_examples:
        path = write_demo_cases()
        print(f"examples: {path}")
        return

    case = generate_transaction(args.scenario, args.seed)
    judgment = judge_transaction(case)
    print(json.dumps(judgment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

