# Risk Policy

## Fast Purchase Behavior

Very short signup-to-purchase intervals should be treated as abnormal first-purchase behavior. In this project, the `<=1 second` pattern is first identified as an unusual distributional phenomenon, then validated against the fraud label distribution. It is used as anomaly evidence and a rule-baseline reason code, not as a universal production truth.

## Device And IP Reuse

Multiple historical users sharing the same device or IP can indicate account farming, scripted abuse, or coordinated risk activity. These features must be computed with as-of historical logic before being used in model training or automated decision support.

## Purchase Value And Night Activity

High purchase value increases potential fraud exposure and should raise review priority when combined with fast purchase, device reuse, IP reuse, or elevated model score. Night-time purchase is not a fraud reason by itself, but it can be used as a weak contextual signal when other risk indicators co-occur.

## Model Score Usage

An elevated model score can support manual review or step-up verification, but it should not trigger automatic hard decline by itself. The score must be interpreted together with reason codes, out-of-time validation performance, and policy evidence.

## Synthetic Data Caveat

This project uses public anonymized e-commerce fraud data with synthetic-like anomalies. Generated case judgments are controlled simulations for risk analytics and agent evaluation. They should not be presented as real-user fraud decisions or universal production rules.

## AI Decision Boundary

The AI layer may summarize risk, cite policy, draft review notes, and route cases to tools. It should not independently decide real-user fraud without model, rule, policy, and human-review safeguards.
