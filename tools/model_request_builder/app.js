const form = document.querySelector("#requestForm");
const preview = document.querySelector("#markdownPreview");
const filenameHint = document.querySelector("#filenameHint");
const validationSummary = document.querySelector("#validationSummary");
const statusDot = document.querySelector(".status-dot");
const experimentsEl = document.querySelector("#experiments");
const toast = document.querySelector("#toast");

const STORAGE_KEY = "jingying_model_request_builder";

const defaults = {
  request_id: "2026-06-fujie-gcard-baseline",
  title: "复借 G 卡 baseline 建模需求",
  owner: "辜子骏",
  workflow: "full_modeling",
  objective: "训练候选复借意愿模型，并与历史 GCard 分数进行 champion/challenger 对比。",
  sample_location: "ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample",
  feature_location: "70张候选特征表；详见 configs/feature_tables.txt",
  target_column: "ftr_30d_ord_flag",
  id_columns: "uid, mdl_dte",
  time_column: "mdl_dte",
  period_column: "ds",
  split_column: "final_flag",
  dev_values: "DEV",
  oot_values: "DEV-OOS, OOT, OOT-OOS",
  sample_definition: "可经营、当前未逾期用户、重资产订单；标签为观察日30天内是否发起。",
  feature_sources: "已回溯的70张候选特征表，候选字段约15028个。",
  require_sql_approval: true,
  feature_notes: "真实拉数前必须 dry-run SQL 并获得明确批准；vendor/feature-select-v2/scripts/code/ 视为只读。",
  candidate_targets: "ftr_30d_ord_flag",
  sample_variants: "all, e2e3, b2",
  modeling_notes: "先跑全客群 baseline，再补老户次新、流失户和分客群加权实验。",
  champions: "gcard_v2, gcard_v4, gcard_v5, gcard_v6",
  comparison_dimensions: "split, month, segment, decile",
  risk_profile_dimensions: "blue_customer_flag, zc_level",
  report_outputs: "model_report.md, model_card.md, executive_summary.md",
  extra_notes: "缺失真实训练或评估结果时必须标记 scaffold，不得编造指标。",
  feature_steps: ["metadata", "d01_d02", "refine"],
  metrics: ["auc", "ks", "decile_lift", "ranking_inversion"],
  report_sections: [
    "sample_overview",
    "feature_screening",
    "modeling_plan",
    "top_features",
    "model_performance",
    "champion_comparison",
    "next_action",
  ],
  experiments: [
    { name: "baseline_all", method: "xgboost", segment: "all" },
    { name: "baseline_e2e3", method: "xgboost", segment: "e2e3" },
    { name: "baseline_b2", method: "xgboost", segment: "b2" },
    { name: "weighted_by_segment_v1", method: "xgboost", segment: "all" },
  ],
};

function list(value) {
  return String(value || "")
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function yamlScalar(value) {
  const text = String(value ?? "");
  if (!text) return '""';
  if (/^[A-Za-z0-9_.:/-]+$/.test(text)) return text;
  return JSON.stringify(text);
}

function yamlList(items, indent = 0) {
  const pad = " ".repeat(indent);
  if (!items.length) return `${pad}[]`;
  return items.map((item) => `${pad}- ${yamlScalar(item)}`).join("\n");
}

function multiline(text) {
  const value = String(text || "").trim();
  if (!value) return "待补充。";
  return value;
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 1800);
}

function getProjectName() {
  const selected = document.querySelector("#projectSelect").value;
  const custom = document.querySelector("#customProject").value.trim();
  return custom || selected || "new-model-project";
}

function setCheckboxGroup(name, values) {
  const selected = new Set(values || []);
  document.querySelectorAll(`[data-group="${name}"] input[type="checkbox"]`).forEach((input) => {
    input.checked = selected.has(input.value);
  });
}

function getCheckboxGroup(name) {
  return Array.from(document.querySelectorAll(`[data-group="${name}"] input[type="checkbox"]:checked`)).map((input) => input.value);
}

function field(name) {
  const el = form.elements[name];
  if (!el) return "";
  if (el.type === "checkbox") return el.checked;
  return el.value;
}

function setField(name, value) {
  const el = form.elements[name];
  if (!el) return;
  if (el.type === "checkbox") {
    el.checked = Boolean(value);
  } else {
    el.value = value ?? "";
  }
}

function experimentRow(exp = { name: "", method: "xgboost", segment: "all" }) {
  const row = document.createElement("div");
  row.className = "experiment-row";
  row.innerHTML = `
    <input data-exp="name" placeholder="experiment_id" value="${exp.name || ""}" />
    <select data-exp="method">
      <option value="xgboost">xgboost</option>
      <option value="lightgbm">lightgbm</option>
      <option value="logistic_regression">logistic_regression</option>
      <option value="custom">custom</option>
    </select>
    <input data-exp="segment" placeholder="segment" value="${exp.segment || "all"}" />
    <button type="button" class="remove-experiment" title="删除实验">×</button>
  `;
  row.querySelector('[data-exp="method"]').value = exp.method || "xgboost";
  row.querySelector(".remove-experiment").addEventListener("click", () => {
    row.remove();
    update();
  });
  row.querySelectorAll("input, select").forEach((el) => el.addEventListener("input", update));
  return row;
}

function setExperiments(items) {
  experimentsEl.innerHTML = "";
  (items && items.length ? items : [{ name: "baseline_all", method: "xgboost", segment: "all" }]).forEach((item) => {
    experimentsEl.appendChild(experimentRow(item));
  });
}

function getExperiments() {
  return Array.from(experimentsEl.querySelectorAll(".experiment-row"))
    .map((row) => ({
      name: row.querySelector('[data-exp="name"]').value.trim(),
      method: row.querySelector('[data-exp="method"]').value,
      segment: row.querySelector('[data-exp="segment"]').value.trim() || "all",
    }))
    .filter((item) => item.name);
}

function collectState() {
  return {
    request_id: field("request_id"),
    title: field("title"),
    owner: field("owner"),
    workflow: field("workflow"),
    project: getProjectName(),
    objective: field("objective"),
    sample_location: field("sample_location"),
    feature_location: field("feature_location"),
    target_column: field("target_column"),
    id_columns: field("id_columns"),
    time_column: field("time_column"),
    period_column: field("period_column"),
    split_column: field("split_column"),
    dev_values: field("dev_values"),
    oot_values: field("oot_values"),
    sample_definition: field("sample_definition"),
    feature_steps: getCheckboxGroup("feature_steps"),
    feature_sources: field("feature_sources"),
    require_sql_approval: field("require_sql_approval"),
    feature_notes: field("feature_notes"),
    candidate_targets: field("candidate_targets"),
    sample_variants: field("sample_variants"),
    experiments: getExperiments(),
    modeling_notes: field("modeling_notes"),
    metrics: getCheckboxGroup("metrics"),
    champions: field("champions"),
    comparison_dimensions: field("comparison_dimensions"),
    risk_profile_dimensions: field("risk_profile_dimensions"),
    report_sections: getCheckboxGroup("report_sections"),
    report_outputs: field("report_outputs"),
    extra_notes: field("extra_notes"),
  };
}

function buildMarkdown(state) {
  const lines = [
    "---",
    `request_id: ${yamlScalar(state.request_id)}`,
    `title: ${yamlScalar(state.title)}`,
    `project: ${yamlScalar(state.project)}`,
    `workflow: ${yamlScalar(state.workflow)}`,
    `owner: ${yamlScalar(state.owner)}`,
    "",
    `sample_location: ${yamlScalar(state.sample_location)}`,
    `feature_location: ${yamlScalar(state.feature_location)}`,
    `target_column: ${yamlScalar(state.target_column)}`,
    "id_columns:",
    yamlList(list(state.id_columns), 2),
    `time_column: ${yamlScalar(state.time_column)}`,
    `period_column: ${yamlScalar(state.period_column)}`,
    `split_column: ${yamlScalar(state.split_column)}`,
    "splits:",
    "  dev:",
    "    values:",
    yamlList(list(state.dev_values), 6),
    "  oot_oos:",
    "    values:",
    yamlList(list(state.oot_values), 6),
    "",
    "sample_checks:",
    yamlList(["sample_check_profile", "sample_check_stability"], 2),
    "",
    "feature_selection:",
    "  rounds:",
    yamlList(state.feature_steps, 4),
    `  require_sql_approval: ${state.require_sql_approval ? "true" : "false"}`,
    "",
    "candidate_targets:",
    yamlList(list(state.candidate_targets), 2),
    "sample_variants:",
    yamlList(list(state.sample_variants), 2),
    "experiments:",
    ...state.experiments.flatMap((exp) => [
      `  - name: ${yamlScalar(exp.name)}`,
      `    method: ${yamlScalar(exp.method)}`,
      `    segment: ${yamlScalar(exp.segment)}`,
    ]),
    "",
    "evaluation:",
    "  metrics:",
    yamlList(state.metrics, 4),
    "  champions:",
    yamlList(list(state.champions), 4),
    "  comparison_dimensions:",
    yamlList(list(state.comparison_dimensions), 4),
    "  risk_profile_dimensions:",
    yamlList(list(state.risk_profile_dimensions), 4),
    "",
    "reports:",
    "  sections:",
    yamlList(state.report_sections, 4),
    "  outputs:",
    yamlList(list(state.report_outputs), 4),
    "---",
    "",
    "# 建模目标",
    "",
    multiline(state.objective),
    "",
    "# 样本与切分",
    "",
    multiline(state.sample_definition),
    "",
    "# 特征筛选要求",
    "",
    `候选特征来源：${state.feature_sources || "待补充。"}`,
    "",
    multiline(state.feature_notes),
    "",
    "# 建模实验要求",
    "",
    multiline(state.modeling_notes),
    "",
    "# 评估与报告要求",
    "",
    `重点比较维度：${state.comparison_dimensions || "待补充。"}`,
    "",
    `风险画像维度：${state.risk_profile_dimensions || "待补充。"}`,
    "",
    "# 补充说明",
    "",
    multiline(state.extra_notes),
    "",
  ];
  return lines.join("\n");
}

function validate(state) {
  const missing = [];
  ["request_id", "title", "project", "target_column", "id_columns", "split_column"].forEach((key) => {
    if (!String(state[key] || "").trim()) missing.push(key);
  });
  if (!state.experiments.length) missing.push("experiments");
  if (!state.metrics.length) missing.push("evaluation.metrics");
  if (!state.report_sections.length) missing.push("reports.sections");
  return missing;
}

function update() {
  const state = collectState();
  const markdown = buildMarkdown(state);
  preview.textContent = markdown;
  filenameHint.textContent = `${state.request_id || "model_request"}.md`;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  const missing = validate(state);
  statusDot.classList.toggle("ok", missing.length === 0);
  statusDot.classList.toggle("error", missing.length > 0);
  validationSummary.textContent = missing.length ? `缺少 ${missing.length} 项必填` : "可生成 Markdown";
}

function applyState(state) {
  Object.entries({ ...defaults, ...state }).forEach(([key, value]) => {
    if (["feature_steps", "metrics", "report_sections", "experiments"].includes(key)) return;
    setField(key, value);
  });
  setCheckboxGroup("feature_steps", state.feature_steps || defaults.feature_steps);
  setCheckboxGroup("metrics", state.metrics || defaults.metrics);
  setCheckboxGroup("report_sections", state.report_sections || defaults.report_sections);
  setExperiments(state.experiments || defaults.experiments);
  const custom = state.project && state.project !== "2026-05-fujie-gcard-v1";
  document.querySelector("#projectSelect").value = custom ? "" : "2026-05-fujie-gcard-v1";
  document.querySelector("#customProject").classList.toggle("hidden", !custom);
  document.querySelector("#customProject").value = custom ? state.project : "";
  update();
}

async function copyMarkdown() {
  await navigator.clipboard.writeText(preview.textContent);
  showToast("Markdown 已复制");
}

function downloadMarkdown() {
  const state = collectState();
  const blob = new Blob([preview.textContent], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.request_id || "model_request"}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  showToast("已生成下载文件");
}

document.querySelector("#projectSelect").addEventListener("change", (event) => {
  document.querySelector("#customProject").classList.toggle("hidden", Boolean(event.target.value));
  update();
});
document.querySelector("#customProject").addEventListener("input", update);
document.querySelector("#loadGcard").addEventListener("click", () => {
  applyState(defaults);
  showToast("已导入复借G卡模板");
});
document.querySelector("#copyMarkdown").addEventListener("click", copyMarkdown);
document.querySelector("#downloadMarkdown").addEventListener("click", downloadMarkdown);
document.querySelector("#resetForm").addEventListener("click", () => {
  localStorage.removeItem(STORAGE_KEY);
  applyState(defaults);
  showToast("已重置");
});
document.querySelector("#addExperiment").addEventListener("click", () => {
  experimentsEl.appendChild(experimentRow({ name: "", method: "xgboost", segment: "all" }));
  update();
});

form.addEventListener("input", update);
form.addEventListener("change", update);

document.querySelectorAll(".nav-list a").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-list a").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  });
});

const saved = localStorage.getItem(STORAGE_KEY);
applyState(saved ? JSON.parse(saved) : defaults);
