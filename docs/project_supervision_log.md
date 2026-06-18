# Project Supervision Log

## 2026-06-16: Project Kickoff

### Completed

- Created formal project folder.
- Copied raw Fraud E-commerce data into `data/raw`.
- Added reproducible data audit pipeline.
- Added anomaly relationship table.
- Added AI-first generated case judgment MVP.
- Added local risk policy document.
- Added unit tests for fast-purchase anomaly logic and generated-case judgment.

### Key Product Decision

The `signup_to_purchase_seconds <= 1` signal is not introduced as an arbitrary hard-coded fraud rule.

It is handled as:

1. Business anomaly hypothesis: completing signup and purchase within one second is operationally abnormal.
2. Distributional observation: the data contains a large cluster of such cases.
3. Label relationship: those cases have a 100% fraud rate in this public dataset.
4. Risk system usage: the signal becomes an anomaly evidence item, reason code, and rule-baseline candidate.
5. Governance caveat: because the dataset has synthetic-like patterns, this signal must be monitored and should not be overgeneralized as a universal production rule.

### Artifacts Produced

- `artifacts/data/data_quality_report.json`
- `artifacts/data/anomaly_relationship_table.csv`
- `artifacts/data/synthetic_and_anomaly_audit.md`
- `artifacts/agent/generated_case_judgments.json`

### Validation

Command:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Result:

```text
Ran 4 tests
OK
```

### Current MVP Behavior

The MVP can generate a transaction scenario, build reason codes, estimate a rule score, cite policy evidence, and return an AI-style structured judgment.

Example scenario:

```bash
PYTHONPATH=src python3 scripts/generate_ai_case_demo.py --scenario synthetic_fast_anomaly --seed 13
```

Expected behavior:

- Flags `FAST_PURCHASE_ANOMALY`
- Cites fast-purchase policy evidence
- Includes synthetic/public-data limitation
- Routes the case to a review-oriented decision

## Next Execution Phase

### Phase 2: Leakage-Aware Feature And Model Layer

Build:

- Time-based train/validation/test split artifacts.
- As-of historical device and IP counts.
- Logistic Regression baseline.
- LightGBM or XGBoost model if dependencies are available.
- TopK review strategy table.

Acceptance criteria:

- No full-data repeated device/IP features in the main model.
- OOT test results reported separately from random or cross-validation results.
- TopK table includes review volume, fraud captured, precision, recall, and baseline lift.

### Phase 3: Agent Harness

Build:

- Evaluation cases for generated transactions.
- Checks for expected reason codes.
- Checks for policy citation presence.
- Checks for numeric consistency with audit artifacts.
- Checks that AI judgment includes limitations for generated/public data.

Acceptance criteria:

- Harness produces a JSON report.
- Failing cases identify which contract broke.

## 2026-06-16: Phase 2 Completed

### Completed

- Added leakage-aware feature pipeline.
- Added as-of historical device/IP counts.
- Documented the leakage tradeoff and justification for not using full-data repeated IP/device flags in the production-style model.
- Removed `sex` from model features for fairness/governance reasons.
- Added Logistic Regression baseline.
- Added XGBoost risk scoring model.
- Added model metrics artifact.
- Added scored transaction artifact.
- Added TopK review strategy artifact.
- Connected generated-case AI judgment to trained XGBoost model score.
- Added regression test for as-of historical features.

### Artifacts Produced

- `artifacts/data/feature_frame.csv`
- `artifacts/modeling/model_metrics.json`
- `artifacts/modeling/scored_transactions.csv`
- `artifacts/modeling/logistic_regression.joblib`
- `artifacts/modeling/xgboost.joblib`
- `artifacts/policy/topk_review_strategy.csv`
- `docs/leakage_tradeoff_and_justification.md`
- `docs/modeling_summary.md`

### Key Modeling Result

Out-of-time test performance is modest:

- XGBoost ROC-AUC: 0.6496
- XGBoost PR-AUC: 0.1469
- XGBoost KS: 0.2954

This confirms the project should not be presented as a high-performance fraud model. It should be presented as a risk decision system that handles data anomalies, leakage control, model scoring, TopK prioritization, and AI-assisted evidence explanation.

### Key Strategy Result

On the OOT test set, reviewing the top 5% scored transactions:

- Review volume: 1,133
- Fraud captured: 272
- Precision@K: 0.2401
- Recall@K: 0.2631
- Lift vs baseline: 5.26x

### Updated AI Behavior

Generated transaction judgments now include:

- `rule_score`
- `model_score`
- `final_risk_score`
- reason codes
- policy citations
- limitations
- trace

If model score is elevated but interpretable rule evidence is weak, the system routes the case to `step_up_verification` rather than silent approval or hard decline.

### Validation

Command:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Result:

```text
Ran 5 tests
OK
```

## Next Execution Phase

### Phase 3: Agent Harness And Local Workbench

Build:

- Eval harness for generated case scenarios.
- Expected reason-code checks.
- Policy citation checks.
- Numeric consistency checks against model and audit artifacts.
- Local HTML/JS workbench for generating cases and viewing AI judgment traces.

Acceptance criteria:

- Harness report is saved as JSON.
- Workbench can display generated transactions, scores, reasons, policy citations, and trace.
- Static demo can be deployed without a live API, with optional local API later.

## 2026-06-16: Phase 3 Completed

### Completed

- Added scenario-based eval harness.
- Added six harness cases covering normal, fast purchase, one-second anomaly, reused device, reused IP, and borderline scenarios.
- Added harness report artifact.
- Added static workbench data bundle.
- Added HTML/CSS/JS workbench.
- Verified the workbench in a local browser.
- Saved a workbench screenshot for quick visual reference.

### Artifacts Produced

- `artifacts/agent/eval_harness_report.json`
- `docs/data/workbench.json`
- `docs/index.html`
- `docs/styles.css`
- `docs/app.js`
- `docs/workbench_screenshot.png`

### Harness Result

```text
6 / 6 passed
pass_rate = 1.0
```

The harness checks:

- expected reason codes
- allowed decisions
- model score presence
- synthetic-data caveat citation
- limitation text
- trace steps
- final risk score range

### Workbench Verification

Local URL:

```text
http://127.0.0.1:4174/
```

Verified behavior:

- Metrics render from `workbench.json`.
- Case list renders six generated scenarios.
- Clicking `synthetic_fast_anomaly` updates transaction facts, scores, reason codes, policy citations, and trace.
- Browser console showed no frontend errors during verification.

## Next Execution Phase

### Phase 4: Lightweight RAG And Evidence Store

Build:

- Convert policy and project docs into small retrievable evidence chunks.
- Add a retrieval tool that returns document id, section, evidence text, and score.
- Replace hard-coded policy citation text with retrieved evidence where possible.
- Extend harness to check retrieval correctness.

Acceptance criteria:

- At least 12 evidence chunks.
- Retrieval works for fast purchase, device/IP reuse, leakage tradeoff, OOT metrics, and AI decision boundary.
- Workbench shows source document and section for policy evidence.

## 2026-06-16: Phase 4 Completed

### Completed

- Added local evidence chunk store.
- Added TF-IDF retrieval tool.
- Built evidence store from policy, leakage-tradeoff, and modeling-summary documents.
- Integrated retrieved evidence into AI policy citations.
- Added source document, source section, and retrieval score to citations.
- Extended harness to check citation metadata and expected source sections.
- Updated workbench citation display.
- Verified workbench in browser after RAG integration.

### Artifacts Produced

- `artifacts/agent/evidence_store.json`
- `docs/rag_evidence_store.md`
- updated `artifacts/agent/generated_case_judgments.json`
- updated `artifacts/agent/eval_harness_report.json`
- updated `docs/data/workbench.json`
- updated `docs/workbench_screenshot.png`

### Harness Result

```text
6 / 6 passed
pass_rate = 1.0
```

### Workbench Verification

Verified that `synthetic_fast_anomaly` displays retrieved citations including:

- `risk_policy.md / Synthetic Data Caveat`
- `leakage_tradeoff_and_justification.md / Fast Purchase Anomaly`
- `risk_policy.md / Device And IP Reuse`
- `risk_policy.md / Purchase Value And Night Activity`
- `risk_policy.md / Model Score Usage`

Browser console showed no frontend errors during verification.

## Next Execution Phase

### Phase 5: Deployment And Resume Integration

Build:

- Create GitHub repository or connect this project to the existing GitHub account.
- Configure GitHub Pages for `docs/`.
- Add concise resume bullet and interview explanation.
- Add README screenshots and run commands.

Acceptance criteria:

- Public demo URL works.
- README has one clear project story.
- Resume wording is specific about AI, RAG, harness, leakage control, and risk strategy.
