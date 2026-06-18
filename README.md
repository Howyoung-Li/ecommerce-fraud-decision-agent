# E-commerce Fraud Decision Agent

AI-first smart risk decision simulator built on public e-commerce fraud data.

The project uses public transaction data as a structured risk environment, then adds an auditable AI risk analyst layer. The AI layer generates or samples transaction scenarios, checks anomaly signals, cites risk policy, and produces structured judgments with limitations.

## Current MVP

- Data quality and anomaly relationship audit
- `<=1s` fast purchase analysis as anomaly-correlation evidence
- Generated transaction scenarios
- Structured AI-style risk judgment
- Policy citations and trace output
- Unit tests for audit and generated-case judgment

## Run

Raw public dataset files are not committed to the repository. Place them under
`data/raw/` before rerunning the full modeling pipeline. The static workbench
uses committed aggregate/demo evidence in `docs/data/workbench.json`, so it can
be viewed directly from GitHub Pages without the raw data.

```bash
PYTHONPATH=src python3 scripts/run_data_audit.py
PYTHONPATH=src python3 scripts/train_risk_model.py
PYTHONPATH=src python3 scripts/run_eval_harness.py
PYTHONPATH=src python3 scripts/generate_ai_case_demo.py --scenario synthetic_fast_anomaly --seed 13
PYTHONPATH=src python3 scripts/generate_ai_case_demo.py --write-examples
python3 scripts/build_workbench.py
PYTHONPATH=src python3 -m unittest discover -s tests
```

Open the static workbench locally:

```bash
python3 -m http.server 4174 --directory docs
```

Then visit `http://127.0.0.1:4174/`.

## Positioning

The `<=1 second` purchase behavior is handled as:

1. Business anomaly hypothesis: completing signup-to-purchase within one second is operationally abnormal.
2. Distributional finding: the dataset contains many such cases.
3. Label relationship: these cases are highly concentrated in fraud labels.
4. Risk usage: the signal becomes a rule-baseline reason code and monitoring item, with caveats about public synthetic-like data.

The AI layer does not make unconstrained fraud decisions. It combines generated case features, anomaly evidence, deterministic reason codes, policy citations, and limitations.

## Current Modeling Result

See `docs/modeling_summary.md`.
See `docs/leakage_tradeoff_and_justification.md` for why the project avoids full-data repeated IP/device features even though they produce much stronger notebook-style metrics.

The OOT test performance is modest, which is expected given the synthetic-like distribution shift. The useful business layer is TopK prioritization:

- XGBoost test ROC-AUC: 0.6496
- XGBoost test PR-AUC: 0.1469
- Top 5% review lift: 5.26x over baseline
- Top 5% review captures 26.31% of OOT fraud

## Project Management

See `docs/project_supervision_log.md` for completed phases, validation results, and next execution criteria.

## Workbench

The static workbench lives in `docs/index.html` and reads `docs/data/workbench.json`.

It displays:

- generated transaction scenarios
- rule score, model score, and final risk score
- decision and risk level
- reason codes
- policy citations
- tool trace
- TopK review strategy
- eval harness results

## RAG Evidence

See `docs/rag_evidence_store.md`.

Policy citations are retrieved from local evidence chunks built from project policy, leakage-tradeoff, and modeling-summary documents. The workbench shows source document, section, retrieval score, and evidence text for each citation.
