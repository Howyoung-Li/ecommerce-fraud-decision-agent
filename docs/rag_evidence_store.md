# RAG Evidence Store

## Purpose

The project uses a lightweight local RAG layer so that policy citations in the AI risk judgment come from retrievable project evidence rather than hard-coded text.

The goal is auditability:

- What source document supports the decision?
- Which section was retrieved?
- What evidence text was used?
- Did the harness verify retrieval behavior?

## Sources

Evidence chunks are built from:

- `policies/risk_policy.md`
- `docs/leakage_tradeoff_and_justification.md`
- `docs/modeling_summary.md`

Current store:

- 18 chunks
- local JSON artifact: `artifacts/agent/evidence_store.json`

## Retrieval Method

The retrieval tool uses local TF-IDF similarity over section title plus body text.

This is intentionally simple:

- No external API dependency
- Reproducible locally
- Easy to inspect
- Enough to demonstrate RAG control and citation grounding

The retrieval output includes:

```json
{
  "chunk_id": "risk_policy:02",
  "source_name": "risk_policy.md",
  "source_path": "policies/risk_policy.md",
  "section": "Device And IP Reuse",
  "text": "...",
  "score": 0.2835
}
```

## Citation Integration

Each generated case judgment maps reason codes to retrieval queries:

- `FAST_PURCHASE_ANOMALY`
- `DEVICE_REUSE`
- `IP_REUSE`
- `HIGH_PURCHASE_VALUE`
- `NIGHT_PURCHASE`
- `MODEL_SCORE_ELEVATED`
- `SYNTHETIC_DATA_CAVEAT`

The workbench displays:

- citation code
- source document
- source section
- retrieval score
- evidence text

## Harness Checks

The eval harness now checks:

- expected reason codes
- allowed decisions
- model score presence
- synthetic-data caveat citation
- citation `source_name`
- citation `source_path`
- citation `section`
- citation `retrieval_score`
- expected citation section for core risk signals
- limitation text
- trace completeness
- final score range

Current result:

```text
6 / 6 passed
pass_rate = 1.0
```

## Interview Framing

> I added a lightweight RAG evidence layer over the project's risk policy, leakage tradeoff note, and modeling summary. The agent does not just generate a reason; it retrieves the supporting source section and exposes the citation in the output. The harness checks that each scenario retrieves the expected type of evidence, such as fast purchase anomaly, as-of IP/device logic, model score usage, and synthetic-data limitations.

