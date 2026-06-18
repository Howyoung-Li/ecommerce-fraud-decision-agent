# Leakage Tradeoff And Justification

## Why The Public Notebook Looks Stronger

The reference notebook achieves much stronger precision and recall because it uses two highly predictive signals:

- `signup_to_purchase_seconds <= 1`
- repeated `device_id` or repeated `ip_address` computed on the full dataset

These signals are useful for exploratory analysis, but they are not equally safe for production-style modeling.

## Fast Purchase Anomaly

`signup_to_purchase_seconds <= 1` is treated as a legitimate anomaly finding.

The reasoning path is:

1. Business hypothesis: completing signup and purchase within one second is operationally abnormal.
2. Distributional finding: the dataset contains a large cluster of these cases.
3. Label relationship: all `<=1s` cases are labeled as fraud in this public dataset.
4. Risk usage: the signal can support anomaly reporting, rule-baseline comparison, reason codes, and monitoring.
5. Caveat: because the signal is concentrated in the early time window and the dataset appears synthetic, it should not be presented as a universal production fraud rule.

This is not considered leakage by itself because the feature is available at transaction time. The concern is generalization, not future information.

## Full-Data Repeated IP/Device Leakage

Full-data repeated ID flags answer this question:

```text
Does this IP/device appear more than once anywhere in the complete dataset?
```

That is not the same as the production-time question:

```text
How many times had this IP/device appeared before this transaction happened?
```

The full-data version can use future transactions to label an earlier transaction as repeated. That creates information leakage because, in a live scoring system, future events are unavailable.

Therefore:

- Full-data repeated IP/device is allowed for EDA and anomaly diagnosis.
- It is not allowed as a main model feature.
- The production-style model uses as-of historical counts instead:
  - `device_seen_count_hist`
  - `ip_seen_count_hist`

## Why Accept Lower OOT Metrics

The project intentionally uses purchase-time train/validation/test splits.

This makes the model evaluation stricter because:

- The `<=1s` anomaly appears in the training period but not in validation/test.
- Fraud rate shifts from about 11.4% in train to about 4.6% in valid/test.
- Future-period behavior is harder to predict than randomly mixed behavior.

The lower OOT metrics are therefore informative. They show that a high random-split score would overstate production readiness.

## Project Decision

The project separates three layers:

1. EDA anomaly layer
   - Can use full-data views to discover suspicious patterns.
   - Must label them as exploratory and non-production.

2. Production-style modeling layer
   - Uses only transaction-time and historical-as-of features.
   - Evaluates on out-of-time validation and test windows.

3. AI risk decision layer
   - Uses model score, rule signals, anomaly audit, policy citations, and limitations.
   - Does not hide weak OOT performance.
   - Routes cases to review or step-up verification rather than claiming fully automated fraud decisions.

## Interview Framing

Strong answer:

> I found that the public notebook's high precision and recall mainly come from one-second purchase anomalies and full-data repeated IP/device flags. The one-second purchase feature is available at transaction time, so I keep it as anomaly evidence and a reason code, but I add a generalization caveat because it is concentrated in the early window. For IP/device reuse, I do not use the full-data repeated flag in the production-style model because it leaks future information. I replace it with as-of historical counts and evaluate with an out-of-time split. The metrics become lower, but they are more honest and closer to how a risk model would behave after deployment.

