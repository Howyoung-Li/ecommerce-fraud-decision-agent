# Modeling Summary

## Feature Governance

The model uses leakage-aware features:

- `signup_to_purchase_seconds`
- `is_fast_purchase_le_1s`
- `is_flash_purchase_le_2min`
- purchase amount and time features
- as-of historical device count
- as-of historical IP count
- source, browser, and country

The model does not use full-data repeated device/IP flags.

The model does not use `sex` as a decision feature. This is intentional because demographic attributes are not needed for the smart risk demo and can create fairness/governance concerns.

See `docs/leakage_tradeoff_and_justification.md` for the detailed rationale behind the leakage and evaluation tradeoff.

## Model Design

Models trained:

- Logistic Regression baseline
- XGBoost risk scoring model

Split design:

- Train: earliest 70% by purchase time
- Validation: next 15%
- Test: latest 15%

This out-of-time design is deliberately stricter than random KFold because fraud behavior and synthetic anomalies shift over time.

## Metrics

| Model | Split | ROC-AUC | PR-AUC | KS | Fraud Rate |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Train | 0.8498 | 0.7427 | 0.6583 | 0.1142 |
| Logistic Regression | Valid | 0.6309 | 0.1292 | 0.2470 | 0.0459 |
| Logistic Regression | Test | 0.6450 | 0.1372 | 0.2947 | 0.0456 |
| XGBoost | Train | 0.8744 | 0.7730 | 0.6634 | 0.1142 |
| XGBoost | Valid | 0.6196 | 0.1344 | 0.2460 | 0.0459 |
| XGBoost | Test | 0.6496 | 0.1469 | 0.2954 | 0.0456 |

## TopK Review Strategy

XGBoost test-set TopK results:

| TopK | Review Volume | Fraud Captured | Precision@K | Recall@K | Lift |
|---:|---:|---:|---:|---:|---:|
| 1% | 227 | 74 | 0.3260 | 0.0716 | 7.15x |
| 3% | 680 | 183 | 0.2691 | 0.1770 | 5.90x |
| 5% | 1,133 | 272 | 0.2401 | 0.2631 | 5.26x |
| 10% | 2,267 | 385 | 0.1698 | 0.3723 | 3.72x |
| 20% | 4,533 | 461 | 0.1017 | 0.4458 | 2.23x |

## Interpretation

The OOT metrics are meaningfully weaker than train metrics. This is a feature, not a failure of the project: it shows that the early synthetic anomaly does not fully generalize into later periods.

The business value is in ranking and triage:

- The model is not strong enough to support automatic hard decline by itself.
- It is useful for prioritizing manual review or step-up verification.
- Top 5% review captures about 26.31% of fraud on the OOT test set with about 5.26x lift over baseline.

## AI Integration

Generated-case judgment now combines:

- Rule score
- XGBoost model score
- Final blended risk score
- Reason codes
- Policy citations
- Limitations about public synthetic-like data
- Trace output

This makes the demo closer to an intelligent risk analyst workflow rather than a pure model benchmark.
