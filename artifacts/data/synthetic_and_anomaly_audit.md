# Data Quality And Anomaly Audit

## Dataset Snapshot

- Rows: 151,112
- Fraud rows: 14,151
- Fraud rate: 0.0936
- Purchase period: 2015-01-01 00:00:44 to 2015-12-16 02:56:05
- Unmapped country rows: 21,966 (14.54%)

## Fast Purchase Observation

- Definition: `signup_to_purchase_seconds <= 1`
- Business hypothesis: A purchase completed within one second after signup is operationally abnormal and may indicate automation, scripted abuse, or synthetic-data generation.
- Rows: 7,600
- Fraud rate among these rows: 1.0000
- Baseline fraud rate: 0.0936
- Lift vs baseline: 10.68x

Interpretation:

Treat this as a distributional anomaly that is strongly associated with fraud in this dataset. It can support a rule baseline and AI explanation, but should be monitored and validated before being generalized to production.

## Time Split Drift

| Split | Rows | Fraud Rate | Fast Purchase Rows | Fast Purchase Fraud Rate | Period |
|---|---:|---:|---:|---:|---|
| train | 105,778 | 0.1142 | 7,600 | 1.0000 | 2015-01-01 00:00:44 to 2015-08-05 11:06:39 |
| valid | 22,667 | 0.0459 | 0 | NA | 2015-08-05 11:07:00 to 2015-09-14 02:11:33 |
| test | 22,667 | 0.0456 | 0 | NA | 2015-09-14 02:14:58 to 2015-12-16 02:56:05 |

## Modeling Implication

- The `<=1s` behavior should enter the project first as an anomaly-correlation finding.
- It can become a rules-only baseline and reason code, but the ML model should also be evaluated without relying solely on this signal.
- Full-data repeated device/IP indicators are useful for EDA but must be rebuilt as historical as-of features before modeling.
