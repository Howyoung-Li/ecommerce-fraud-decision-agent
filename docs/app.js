const state = {
  data: null,
  selectedIndex: 0,
};

const fmt = new Intl.NumberFormat("en-US");
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const num = (value, digits = 3) =>
  value === null || value === undefined ? "--" : Number(value).toFixed(digits);

function label(text) {
  return String(text).replaceAll("_", " ");
}

function setText(id, text) {
  document.getElementById(id).textContent = text;
}

async function loadWorkbench() {
  const response = await fetch("./data/workbench.json");
  state.data = await response.json();
  renderMetrics();
  renderCases();
  renderTopK();
  renderHarness();
  renderSelectedCase();
}

function renderMetrics() {
  const metrics = state.data.modeling.metrics.xgboost.test;
  const top5 = state.data.modeling.xgboost_test_topk.find(
    (row) => Number(row.topk_rate) === 0.05
  );
  const fast = state.data.data_quality.fast_purchase.relationship_to_label;

  setText("rocAuc", num(metrics.roc_auc, 3));
  setText("prAuc", num(metrics.pr_auc, 3));
  setText("topLift", `${num(top5.lift_vs_baseline, 2)}x`);
  setText("fastLift", `${num(fast.lift_vs_baseline, 2)}x`);

  const harness = state.data.agent.harness;
  const status = document.getElementById("harnessStatus");
  status.textContent = `Harness ${harness.passed}/${harness.total}`;
  status.classList.add("status-pass");
}

function renderCases() {
  const list = document.getElementById("caseList");
  list.innerHTML = "";
  state.data.agent.cases.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-button";
    if (index === state.selectedIndex) button.classList.add("active");
    button.dataset.index = String(index);
    button.innerHTML = `
      <strong>${label(item.case.scenario)}</strong>
      <span>${item.decision} · score ${num(item.final_risk_score, 3)}</span>
    `;
    button.addEventListener("click", () => {
      state.selectedIndex = index;
      renderCases();
      renderSelectedCase();
    });
    list.appendChild(button);
  });
}

function renderSelectedCase() {
  const item = state.data.agent.cases[state.selectedIndex];
  setText("scenarioName", label(item.case.scenario));
  setText("caseId", item.case.case_id);

  const badge = document.getElementById("decisionBadge");
  badge.className = `decision-badge ${item.decision}`;
  badge.textContent = label(item.decision);

  setText("ruleScore", num(item.rule_score, 3));
  setText("modelScore", num(item.model_score, 3));
  setText("finalScore", num(item.final_risk_score, 3));
  setText("riskLevel", label(item.risk_level));
  setText("aiSummary", item.ai_risk_summary);
  setText("limitations", item.limitations);

  renderFacts(item.case);
  renderReasons(item.reason_codes);
  renderCitations(item.policy_citations);
  renderTrace(item.trace);
}

function renderFacts(caseData) {
  const facts = [
    ["Signup to purchase", `${fmt.format(caseData.signup_to_purchase_seconds)} sec`],
    ["Purchase value", `$${fmt.format(caseData.purchase_value)}`],
    ["Device seen hist", fmt.format(caseData.device_seen_user_count_hist)],
    ["IP seen hist", fmt.format(caseData.ip_seen_user_count_hist)],
    ["Source", caseData.source],
    ["Browser", caseData.browser],
    ["Country", caseData.country],
    ["Purchase hour", String(caseData.purchase_hour)],
  ];
  document.getElementById("transactionFacts").innerHTML = facts
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
    .join("");
}

function renderReasons(reasons) {
  document.getElementById("reasonCodes").innerHTML = reasons
    .map(
      (reason) => `
        <article class="reason severity-${reason.severity}">
          <strong>${reason.code}</strong>
          <p>${reason.evidence}</p>
          <p>${reason.source}</p>
        </article>
      `
    )
    .join("");
}

function renderCitations(citations) {
  document.getElementById("policyCitations").innerHTML = citations
    .map(
      (citation) => `
        <article class="citation">
          <strong>${citation.code}</strong>
          <p>${citation.source_name} · ${citation.section} · score ${num(citation.retrieval_score, 3)}</p>
          <p>${citation.text}</p>
        </article>
      `
    )
    .join("");
}

function renderTrace(trace) {
  document.getElementById("trace").innerHTML = trace
    .map((step) => {
      const details = Object.entries(step)
        .filter(([key]) => key !== "step")
        .map(([key, value]) => `${key}: ${value}`)
        .join(" · ");
      return `
        <article class="trace-step">
          <strong>${step.step}</strong>
          <p>${details}</p>
        </article>
      `;
    })
    .join("");
}

function renderTopK() {
  document.getElementById("topkTable").innerHTML = state.data.modeling.xgboost_test_topk
    .map(
      (row) => `
        <tr>
          <td>${pct(row.topk_rate, 0)}</td>
          <td>${fmt.format(row.review_volume)}</td>
          <td>${fmt.format(row.fraud_captured)}</td>
          <td>${pct(row.precision_at_k, 1)}</td>
          <td>${pct(row.recall_at_k, 1)}</td>
          <td>${num(row.lift_vs_baseline, 2)}x</td>
        </tr>
      `
    )
    .join("");
}

function renderHarness() {
  document.getElementById("harnessList").innerHTML = state.data.agent.harness.results
    .map(
      (item) => `
        <article class="harness-item ${item.passed ? "pass" : ""}">
          <strong>${item.passed ? "PASS" : "FAIL"} · ${label(item.scenario)}</strong>
          <p>${item.decision} · ${label(item.risk_level)} · ${item.reason_codes.join(", ")}</p>
        </article>
      `
    )
    .join("");
}

document.getElementById("nextCase").addEventListener("click", () => {
  state.selectedIndex = (state.selectedIndex + 1) % state.data.agent.cases.length;
  renderCases();
  renderSelectedCase();
});

loadWorkbench().catch((error) => {
  document.body.innerHTML = `<main class="decision-panel"><h1>Workbench data failed to load</h1><p>${error}</p></main>`;
});
