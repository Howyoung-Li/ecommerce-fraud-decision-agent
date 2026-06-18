const state = {
  data: null,
  selectedIndex: 0,
};

const fmt = new Intl.NumberFormat("en-US");
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const num = (value, digits = 3) =>
  value === null || value === undefined ? "--" : Number(value).toFixed(digits);

const zh = {
  scenario: {
    normal_new_user: "正常新用户",
    fast_purchase_suspicious: "快购可疑交易",
    synthetic_fast_anomaly: "1秒异常购买",
    reused_device_abuse: "设备复用风险",
    reused_ip_attack: "IP复用攻击",
    borderline_mixed_signal: "边界混合信号",
  },
  decision: {
    approve: "通过",
    step_up_verification: "加强验证",
    manual_review: "人工审核",
  },
  risk: {
    low: "低风险",
    medium: "中风险",
    high: "高风险",
  },
  reason: {
    FAST_PURCHASE_AFTER_SIGNUP: "注册后极短时间购买",
    FAST_PURCHASE_ANOMALY: "极短购买异常",
    DEVICE_REUSE: "设备复用",
    DEVICE_REUSE_WEAK: "设备复用轻微信号",
    IP_REUSE: "IP复用",
    IP_REUSE_WEAK: "IP复用轻微信号",
    HIGH_PURCHASE_VALUE: "高金额交易",
    NIGHT_PURCHASE: "夜间交易",
    MODEL_SCORE_ELEVATED: "模型分偏高",
    SYNTHETIC_DATA_CAVEAT: "公开匿名数据限制",
  },
  source: {
    "data_audit + business_anomaly_hypothesis": "数据审计 + 业务异常假设",
    feature_rule: "特征规则",
    historical_feature: "历史特征",
    trained_model: "训练模型",
    transaction_feature: "交易特征",
  },
  citation: {
    SYNTHETIC_DATA_CAVEAT:
      "本项目使用公开匿名电商欺诈数据，部分异常分布具有合成数据特征。生成案例用于风控分析和 Agent 评测，不应被表述为真实用户欺诈判定或可直接上线的通用规则。",
    MODEL_SCORE_ELEVATED:
      "偏高的模型分可以支持人工审核或加强验证，但不应单独触发自动拒绝。需要结合原因码、时间外验证表现和策略证据进行解释。",
    FAST_PURCHASE_ANOMALY:
      "注册后极短时间内完成购买属于业务异常信号。本数据中该信号与欺诈标签高度相关，可作为规则基线、审核提示和监控项，但上线前仍需持续验证。",
    DEVICE_REUSE:
      "同一设备被多个用户历史使用时，可能提示批量注册、账户接管或设备农场风险。该信号需要只使用交易发生前已有的信息构建，避免未来信息泄露。",
    IP_REUSE:
      "同一 IP 被多个用户历史使用时，可能提示代理、攻击流量或批量操作风险。该信号应与设备、交易金额、时间和模型分共同判断。",
    HIGH_PURCHASE_VALUE:
      "高金额交易和异常活跃时间段会提高审核优先级，但需要与其他风险信号共同使用，避免对正常高价值用户造成误伤。",
    NIGHT_PURCHASE:
      "夜间交易可作为弱风险提示，适合用于原因解释和优先级调整，不适合作为单独拒绝依据。",
  },
  section: {
    "Synthetic Data Caveat": "合成数据限制说明",
    "Model Score Usage": "模型分使用边界",
    "Fast Purchase Anomaly": "极短购买异常",
    "Device And IP Reuse": "设备与 IP 复用",
    "Purchase Value And Night Activity": "交易金额与夜间行为",
  },
  trace: {
    generate_transaction: "生成交易案例",
    build_reason_codes: "生成风险原因码",
    estimate_rule_score: "计算规则分",
    score_with_trained_model: "调用模型打分",
    blend_scores: "融合规则分与模型分",
    retrieve_policy_citations: "检索策略证据",
    compose_ai_risk_summary: "生成 AI 风险摘要",
  },
  fields: {
    scenario: "场景",
    count: "数量",
    score: "分数",
    citation_count: "证据数",
    decision: "决策",
  },
};

function label(text, group) {
  const value = String(text);
  if (group && zh[group]?.[value]) return zh[group][value];
  return value.replaceAll("_", " ");
}

function reasonEvidence(reason) {
  const evidence = String(reason.evidence || "");
  return evidence
    .replace("xgboost_model_score=", "XGBoost模型分=")
    .replace("device_seen_user_count_hist=", "历史设备关联用户数=")
    .replace("ip_seen_user_count_hist=", "历史IP关联用户数=")
    .replace("purchase_value=", "交易金额=")
    .replace("purchase_hour=", "交易小时=")
    .replace("signup_to_purchase_seconds=", "注册到购买耗时=");
}

function chineseSummary(item) {
  const reasons = item.reason_codes.map((reason) => label(reason.code, "reason")).join("、");
  return `系统建议“${label(item.decision, "decision")}”，因为触发了 ${item.reason_codes.length} 个风险信号：${reasons}。`;
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
  status.textContent = `评测通过 ${harness.passed}/${harness.total}`;
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
      <strong>${label(item.case.scenario, "scenario")}</strong>
      <span>${label(item.decision, "decision")} · 风险分 ${num(item.final_risk_score, 3)}</span>
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
  setText("scenarioName", label(item.case.scenario, "scenario"));
  setText("caseId", `案例 ${item.case.case_id}`);

  const badge = document.getElementById("decisionBadge");
  badge.className = `decision-badge ${item.decision}`;
  badge.textContent = label(item.decision, "decision");

  setText("ruleScore", num(item.rule_score, 3));
  setText("modelScore", num(item.model_score, 3));
  setText("finalScore", num(item.final_risk_score, 3));
  setText("riskLevel", label(item.risk_level, "risk"));
  setText("aiSummary", chineseSummary(item));
  setText(
    "limitations",
    "说明：本判断基于公开匿名数据和生成案例，用于风控分析、策略解释和 Agent 评测，不代表真实用户欺诈定性。"
  );

  renderFacts(item.case);
  renderReasons(item.reason_codes);
  renderCitations(item.policy_citations);
  renderTrace(item.trace);
}

function renderFacts(caseData) {
  const facts = [
    ["注册到购买耗时", `${fmt.format(caseData.signup_to_purchase_seconds)} 秒`],
    ["交易金额", `$${fmt.format(caseData.purchase_value)}`],
    ["历史设备关联用户数", fmt.format(caseData.device_seen_user_count_hist)],
    ["历史IP关联用户数", fmt.format(caseData.ip_seen_user_count_hist)],
    ["渠道来源", caseData.source],
    ["浏览器", caseData.browser],
    ["国家/地区", caseData.country],
    ["购买小时", String(caseData.purchase_hour)],
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
          <strong>${label(reason.code, "reason")}</strong>
          <p>${reasonEvidence(reason)}</p>
          <p>${label(reason.source, "source")}</p>
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
          <strong>${label(citation.code, "reason") || citation.code}</strong>
          <p>${citation.source_name} · ${label(citation.section, "section")} · 检索分 ${num(citation.retrieval_score, 3)}</p>
          <p>${zh.citation[citation.code] || citation.text}</p>
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
        .map(([key, value]) => {
          const field = zh.fields[key] || key;
          const rendered =
            key === "scenario"
              ? label(value, "scenario")
              : key === "decision"
                ? label(value, "decision")
                : value;
          return `${field}: ${rendered}`;
        })
        .join(" · ");
      return `
        <article class="trace-step">
          <strong>${label(step.step, "trace")}</strong>
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
          <strong>${item.passed ? "通过" : "失败"} · ${label(item.scenario, "scenario")}</strong>
          <p>${label(item.decision, "decision")} · ${label(item.risk_level, "risk")} · ${item.reason_codes.map((code) => label(code, "reason")).join("、")}</p>
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
  document.body.innerHTML = `<main class="decision-panel"><h1>工作台数据加载失败</h1><p>${error}</p></main>`;
});
