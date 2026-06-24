const form = document.querySelector("#requestForm");
const preview = document.querySelector("#markdownPreview");
const filenameHint = document.querySelector("#filenameHint");
const experimentsEl = document.querySelector("#experiments");
const toast = document.querySelector("#toast");
const markdownModal = document.querySelector("#markdownModal");
const currentStageTitleEl = document.querySelector("#currentStageTitle");
const currentStageHintEl = document.querySelector("#currentStageHint");
const sectionProgressEl = document.querySelector("#sectionProgress");
const summaryTextEl = document.querySelector("#summaryText");
const domainBadgeEl = document.querySelector("#domainBadge");
const profileBadgeEl = document.querySelector("#profileBadge");
const templateBadgeEl = document.querySelector("#templateBadge");
const stepCountBadgeEl = document.querySelector("#stepCountBadge");
const stepSummaryEl = document.querySelector("#stepSummary");
const nextActionsEl = document.querySelector("#nextActions");

const STORAGE_KEY = "risk_model_request_builder";
const LEGACY_STORAGE_KEY = "jingying_model_request_builder";
const CUSTOM_TEMPLATE_STORAGE_KEY = "risk_model_request_builder_custom_templates";
const DEFAULT_PROJECT_NAME = "new-model-project";

const FORM_STEPS = [
  {
    id: "basic",
    title: "基本信息",
    hint: "先确定标题、任务模式、业务域和建模目标。内部编号会自动生成，不需要手动维护。",
  },
  {
    id: "sample",
    title: "样本与切分",
    hint: "补齐样本位置、标签、时间字段和切分口径。主键和分区字段优先沿用 project.yml 数据契约。",
  },
  {
    id: "features",
    title: "特征筛选",
    hint: "选择具体筛选方法和阈值。可用性过滤会在执行阶段默认启用，不需要在需求里手动勾选。",
  },
  {
    id: "modeling",
    title: "建模实验",
    hint: "把 baseline、分客群、加权或不同 Y 的实验拆开写清，便于后续 run plan 追踪。",
  },
  {
    id: "evaluation",
    title: "评估报告",
    hint: "选择指标、报告章节和重点对比维度，避免评估口径在执行阶段临时漂移。",
  },
  {
    id: "notes",
    title: "补充说明",
    hint: "记录开放问题、禁止事项和特殊口径。这里的内容会作为 Agent 执行约束写入 Markdown。",
  },
  {
    id: "stageSteps",
    title: "执行总览",
    hint: "最后确认自动生成的 stage_steps。高级参数默认折叠，只有确实要覆盖 profile 默认值时才填写。",
  },
];

const STAGE_GROUPS = [
  { stage: "sample_check", key: "sample_check_steps" },
  { stage: "feature_metadata", key: "feature_metadata_steps" },
  { stage: "feature_prescreen", key: "feature_prescreen_steps" },
  { stage: "build_wide_sql", key: "build_wide_sql_steps" },
  { stage: "feature_refine", key: "feature_refine_steps" },
  { stage: "train_baseline", key: "train_baseline_steps" },
  { stage: "evaluate", key: "evaluate_steps" },
  { stage: "compare", key: "compare_steps" },
  { stage: "report", key: "report_steps" },
];

const STAGE_ALIASES = {
  d01_d02_screening: "feature_prescreen",
};

const STEP_ALIASES = {
  d01_d02_batch_screening: "feature_quality_prescreen",
};

const CHECKBOX_GROUPS = [
  "feature_steps",
  "metrics",
  "report_sections",
  ...STAGE_GROUPS.map((item) => item.key),
];

const FEATURE_METHOD_STEP_IDS = [
  "missing_rate_filter",
  "constant_value_filter",
  "iv_filter",
  "correlation_dedup",
  "random_noise_importance",
  "null_importance_filter",
  "baseline_importance_filter",
];

const BUSINESS_DOMAIN_LABELS = {
  acquisition: "获客",
  preloan: "贷前",
  inloan_risk: "贷中风险",
  inloan_operation: "贷中经营",
};

const TASK_MODE_LABELS = {
  full_modeling: "完整建模",
  feature_selection: "特征筛选",
  train_baseline: "训练基线",
  challenger_evaluation: "挑战者评估",
};

const PROFILE_LABELS = {
  acquisition: "获客通用",
  acquisition_quality: "获客质量",
  acquisition_conversion: "获客转化",
  preloan_credit_card: "贷前信用卡",
  credit_product_eval: "资信产品评估",
  inloan_behavior_card: "贷中行为卡",
  feature_gain_eval: "特征增益评估",
  inloan_operation: "贷中经营",
  fujie_gcard_main_lgbm: "复借 G 卡主模型",
};

const STAGE_LABELS = {
  sample_check: "样本检查",
  feature_metadata: "特征元数据",
  feature_prescreen: "特征初筛",
  build_wide_sql: "宽表 SQL",
  feature_refine: "特征精筛",
  train_baseline: "训练",
  evaluate: "评估",
  compare: "对比",
  report: "报告",
};

const STEP_LABELS = {
  field_contract: "字段契约",
  key_uniqueness: "主键去重",
  monthly_label_distribution: "月度标签分布",
  segment_distribution: "分客群分布",
  account_status_distribution: "账期分布",
  channel_distribution: "渠道统计",
  dual_target_split: "双 Y 标拆分",
  credit_product_coverage: "资信覆盖",
  feature_metadata_export: "元数据导出",
  feature_quality_prescreen: "特征质量初筛",
  wide_sql_generation: "宽表 SQL 生成",
  sql_review_gate: "SQL Review Gate",
  feature_availability_filter: "可用性过滤",
  missing_rate_filter: "缺失率过滤",
  constant_value_filter: "恒一值过滤",
  iv_filter: "IV 过滤",
  psi_filter: "PSI 过滤",
  correlation_dedup: "相关性去重",
  random_noise_importance: "随机噪声重要性",
  null_importance_filter: "空标签重要性",
  baseline_importance_filter: "基线模型重要性",
  gain_importance_filter: "Gain 重要性过滤",
  lightgbm_binary_training: "LightGBM 二分类训练",
  scale_pos_weight: "正负样本权重",
  hier_ranknet_training: "HierRankNet 训练",
  teacher_student_distillation: "蒸馏训练",
  auc_ks: "AUC / KS",
  decile_lift: "十分箱 Lift",
  monthly_stability: "月度稳定性",
  score_psi: "分数 PSI",
  segment_metrics: "分客群评估",
  intent_zc_cross_risk: "意愿 x 资质交叉风险",
  cross_gain_matrix: "交叉增益矩阵",
  roll_rate_analysis: "滚动率分析",
  channel_metrics: "分渠道评估",
  dual_model_synergy: "双模型协同",
  sub_funnel_metrics: "子漏斗评估",
  credit_product_standalone_eval: "资信产品单点评估",
  credit_product_fusion_eval: "资信产品融合评估",
  feature_gain_summary: "特征增益汇总",
  champion_challenger: "Champion / Challenger 对比",
  model_report: "模型报告",
  model_recovery_report: "模型回收报告",
  credit_product_report: "资信产品报告",
};

const BASE_PARAM_DEFAULTS = {
  sample_min_monthly_count: "",
  missing_rate_threshold: "0.9",
  constant_max_unique_values: "1",
  iv_min: "0.005",
  psi_max: "0.2",
  correlation_method: "spearman",
  correlation_max_abs: "0.8",
  gain_tail_fraction: "0.1",
  gain_max_auc_drop: "0.005",
  score_psi_warn: "",
  monthly_max_ks_std: "",
  sql_block_high_risk: true,
};

const EMPTY_STAGE_STEP_DEFAULTS = Object.fromEntries(STAGE_GROUPS.map(({ key }) => [key, []]));
const SYSTEM_DEFAULT_STAGE_STEPS = {
  feature_refine_steps: ["feature_availability_filter"],
};

const BUSINESS_DOMAIN_PROFILE_DEFAULTS = {
  acquisition: "acquisition",
  preloan: "preloan_credit_card",
  inloan_risk: "inloan_behavior_card",
  inloan_operation: "inloan_operation",
};

const PROFILE_BUSINESS_DOMAINS = {
  acquisition: "acquisition",
  acquisition_quality: "acquisition",
  acquisition_conversion: "acquisition",
  preloan_credit_card: "preloan",
  credit_product_eval: "preloan",
  inloan_behavior_card: "inloan_risk",
  feature_gain_eval: "inloan_operation",
  inloan_operation: "inloan_operation",
  fujie_gcard_main_lgbm: "inloan_operation",
};

const PROFILE_PARAM_DEFAULTS = {
  inloan_behavior_card: {
    psi_max: "0.25",
    monthly_max_ks_std: "0.03",
  },
  fujie_gcard_main_lgbm: {
    sql_block_high_risk: true,
  },
};

const PROFILE_FEATURE_ROUNDS = {
  acquisition: ["refine"],
  preloan_credit_card: ["refine"],
  inloan_behavior_card: ["refine"],
  inloan_operation: ["refine"],
  acquisition_quality: ["refine"],
  acquisition_conversion: ["refine"],
  feature_gain_eval: ["refine"],
  credit_product_eval: [],
  fujie_gcard_main_lgbm: ["metadata", "prescreen", "refine"],
};

const PROFILE_PRESETS = {
  acquisition: {
    sample_check_steps: [
      "field_contract",
      "key_uniqueness",
      "monthly_label_distribution",
      "channel_distribution",
      "dual_target_split",
    ],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training", "teacher_student_distillation", "hier_ranknet_training"],
    evaluate_steps: [
      "auc_ks",
      "decile_lift",
      "monthly_stability",
      "score_psi",
      "channel_metrics",
      "sub_funnel_metrics",
      "dual_model_synergy",
    ],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
  preloan_credit_card: {
    sample_check_steps: ["field_contract", "key_uniqueness", "monthly_label_distribution"],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training"],
    evaluate_steps: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "cross_gain_matrix"],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
  inloan_behavior_card: {
    sample_check_steps: [
      "field_contract",
      "key_uniqueness",
      "monthly_label_distribution",
      "account_status_distribution",
    ],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "psi_filter",
      "correlation_dedup",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training", "scale_pos_weight"],
    evaluate_steps: [
      "auc_ks",
      "decile_lift",
      "monthly_stability",
      "score_psi",
      "cross_gain_matrix",
      "roll_rate_analysis",
    ],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
  inloan_operation: {
    sample_check_steps: ["field_contract", "key_uniqueness", "monthly_label_distribution", "segment_distribution"],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "correlation_dedup",
      "random_noise_importance",
      "null_importance_filter",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training"],
    evaluate_steps: [
      "auc_ks",
      "decile_lift",
      "monthly_stability",
      "score_psi",
      "segment_metrics",
      "cross_gain_matrix",
      "feature_gain_summary",
    ],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
  acquisition_quality: {
    sample_check_steps: [
      "field_contract",
      "key_uniqueness",
      "monthly_label_distribution",
      "channel_distribution",
      "dual_target_split",
    ],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training", "teacher_student_distillation"],
    evaluate_steps: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "channel_metrics", "dual_model_synergy"],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
  acquisition_conversion: {
    sample_check_steps: [
      "field_contract",
      "key_uniqueness",
      "monthly_label_distribution",
      "channel_distribution",
      "dual_target_split",
    ],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training", "hier_ranknet_training"],
    evaluate_steps: [
      "auc_ks",
      "decile_lift",
      "monthly_stability",
      "score_psi",
      "channel_metrics",
      "sub_funnel_metrics",
      "dual_model_synergy",
    ],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
  feature_gain_eval: {
    sample_check_steps: ["field_contract", "key_uniqueness", "monthly_label_distribution"],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training"],
    evaluate_steps: ["auc_ks", "decile_lift", "monthly_stability", "cross_gain_matrix", "feature_gain_summary"],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
  credit_product_eval: {
    sample_check_steps: ["field_contract", "key_uniqueness", "monthly_label_distribution", "credit_product_coverage"],
    feature_metadata_steps: [],
    feature_prescreen_steps: [],
    build_wide_sql_steps: [],
    feature_refine_steps: [],
    train_baseline_steps: ["lightgbm_binary_training"],
    evaluate_steps: [
      "auc_ks",
      "decile_lift",
      "monthly_stability",
      "score_psi",
      "credit_product_standalone_eval",
      "credit_product_fusion_eval",
    ],
    compare_steps: ["champion_challenger"],
    report_steps: ["credit_product_report"],
  },
  fujie_gcard_main_lgbm: {
    sample_check_steps: ["field_contract", "key_uniqueness", "monthly_label_distribution", "segment_distribution"],
    feature_metadata_steps: ["feature_metadata_export"],
    feature_prescreen_steps: ["feature_quality_prescreen"],
    build_wide_sql_steps: ["wide_sql_generation", "sql_review_gate"],
    feature_refine_steps: [
      "missing_rate_filter",
      "constant_value_filter",
      "iv_filter",
      "correlation_dedup",
      "random_noise_importance",
      "null_importance_filter",
      "baseline_importance_filter",
    ],
    train_baseline_steps: ["lightgbm_binary_training"],
    evaluate_steps: [
      "auc_ks",
      "decile_lift",
      "monthly_stability",
      "score_psi",
      "segment_metrics",
      "intent_zc_cross_risk",
    ],
    compare_steps: ["champion_challenger"],
    report_steps: ["model_report"],
  },
};

const defaults = {
  request_id: "2026-06-fujie-gcard-baseline",
  title: "复借 G 卡 baseline 建模需求",
  project: "2026-05-fujie-gcard-v1",
  owner: "辜子骏",
  workflow: "full_modeling",
  business_domain: "inloan_operation",
  scenario_profile: "fujie_gcard_main_lgbm",
  objective: "训练候选复借意愿模型，并与历史 GCard 分数进行 champion/challenger 对比。",
  sample_location: "ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample",
  feature_location: "70张候选特征表；详见 configs/feature_tables.txt",
  target_column: "ftr_30d_ord_flag",
  id_columns: "uid, mdl_dte",
  time_column: "mdl_dte",
  period_column: "ds",
  split_column: "final_flag",
  dev_values: "DEV",
  oos_values: "DEV-OOS",
  oot_values: "OOT, OOT-OOS",
  sample_definition: "可经营、当前未逾期用户、重资产订单；标签为观察日30天内是否发起。",
  feature_sources: "已回溯的70张候选特征表，候选字段约15028个。",
  require_sql_approval: true,
  feature_notes: "真实拉数前必须 dry-run SQL 并获得明确批准；vendor/feature-select-v2/scripts/code/ 视为只读。",
  candidate_targets: "ftr_30d_ord_flag",
  sample_variants: "all, e2e3, b2",
  experiment_description: "先跑全客群 baseline，再补老户次新、流失户和分客群加权实验。",
  modeling_notes: "先跑全客群 baseline，再补老户次新、流失户和分客群加权实验。",
  champions: "gcard_v2, gcard_v4, gcard_v5, gcard_v6",
  comparison_dimensions: "split, month, segment, decile",
  risk_profile_dimensions: "blue_customer_flag, zc_level",
  report_outputs: "model_report.md, model_card.md, executive_summary.md",
  extra_notes: "缺失真实训练或评估结果时必须标记 scaffold，不得编造指标。",
  feature_steps: ["metadata", "prescreen", "refine"],
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
    { name: "baseline_all", method: "xgboost", segment: "all", description: "全客群 baseline。" },
    { name: "baseline_e2e3", method: "xgboost", segment: "e2e3", description: "老户次新客群 baseline。" },
    { name: "baseline_b2", method: "xgboost", segment: "b2", description: "流失户客群 baseline。" },
    { name: "weighted_by_segment_v1", method: "xgboost", segment: "all", description: "按客群权重调整的候选实验。" },
  ],
  ...BASE_PARAM_DEFAULTS,
  ...PROFILE_PRESETS.fujie_gcard_main_lgbm,
};

const newDocumentDefaults = {
  request_id: "",
  title: "",
  project: DEFAULT_PROJECT_NAME,
  owner: "",
  workflow: "full_modeling",
  business_domain: "preloan",
  scenario_profile: "preloan_credit_card",
  objective: "",
  sample_location: "",
  feature_location: "",
  target_column: "",
  id_columns: "",
  time_column: "",
  period_column: "",
  split_column: "",
  dev_values: "",
  oos_values: "",
  oot_values: "",
  sample_definition: "",
  feature_sources: "",
  require_sql_approval: true,
  feature_notes: "",
  candidate_targets: "",
  sample_variants: "",
  experiment_description: "",
  modeling_notes: "",
  champions: "",
  comparison_dimensions: "",
  risk_profile_dimensions: "",
  report_outputs: "model_report.md, model_card.md, executive_summary.md",
  extra_notes: "",
  feature_steps: PROFILE_FEATURE_ROUNDS.preloan_credit_card,
  metrics: ["auc", "ks", "decile_lift"],
  report_sections: [
    "sample_overview",
    "feature_screening",
    "modeling_plan",
    "model_performance",
    "next_action",
  ],
  experiments: [],
  ...BASE_PARAM_DEFAULTS,
  ...PROFILE_PRESETS.preloan_credit_card,
};

const blankTemplateDefaults = {
  ...newDocumentDefaults,
  feature_steps: [],
  metrics: [],
  report_sections: [],
  report_outputs: "",
  experiments: [],
  ...EMPTY_STAGE_STEP_DEFAULTS,
};

const BUILT_IN_TEMPLATES = {
  blank: {
    label: "空白模板",
    state: blankTemplateDefaults,
  },
  default: {
    label: "默认模板",
    state: newDocumentDefaults,
  },
  fujie_gcard: {
    label: "复借 G 卡主模型",
    state: defaults,
  },
};

let currentProject = DEFAULT_PROJECT_NAME;
let activeSectionId = FORM_STEPS[0].id;
let stageGroupSelections = { ...EMPTY_STAGE_STEP_DEFAULTS };

function pad2(value) {
  return String(value).padStart(2, "0");
}

function generateRequestId(date = new Date()) {
  const year = date.getFullYear();
  const month = pad2(date.getMonth() + 1);
  const day = pad2(date.getDate());
  const hour = pad2(date.getHours());
  const minute = pad2(date.getMinutes());
  return `${year}${month}${day}-${hour}${minute}-model-request`;
}

function ensureRequestId(value) {
  const existing = String(value || "").trim();
  if (existing) return existing;
  const generated = generateRequestId();
  setField("request_id", generated);
  return generated;
}

function taskModeLabel(workflow) {
  return TASK_MODE_LABELS[workflow] || workflow || TASK_MODE_LABELS.full_modeling;
}

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

function yamlMapOfLists(map, indent = 0) {
  const pad = " ".repeat(indent);
  const lines = [];
  Object.entries(map).forEach(([key, items]) => {
    if (!items.length) return;
    lines.push(`${pad}${key}:`);
    lines.push(yamlList(items, indent + 2));
  });
  return lines.length ? lines.join("\n") : `${pad}{}`;
}

function yamlMapOfMaps(map, indent = 0) {
  const pad = " ".repeat(indent);
  const lines = [];
  Object.entries(map).forEach(([stepId, params]) => {
    const entries = Object.entries(params).filter(([, value]) => value !== "" && value !== null && value !== undefined);
    if (!entries.length) return;
    lines.push(`${pad}${stepId}:`);
    entries.forEach(([key, value]) => {
      lines.push(`${pad}  ${key}: ${yamlScalar(value)}`);
    });
  });
  return lines.length ? lines.join("\n") : `${pad}{}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function profilePreset(profile) {
  return PROFILE_PRESETS[profile] || PROFILE_PRESETS.preloan_credit_card;
}

function paramDefaultsForProfile(profile) {
  return {
    ...BASE_PARAM_DEFAULTS,
    ...(PROFILE_PARAM_DEFAULTS[profile] || {}),
  };
}

function scenarioProfileForBusinessDomain(domain) {
  return BUSINESS_DOMAIN_PROFILE_DEFAULTS[domain] || BUSINESS_DOMAIN_PROFILE_DEFAULTS.preloan;
}

function businessDomainForProfile(profile) {
  return PROFILE_BUSINESS_DOMAINS[profile] || "preloan";
}

function featureRoundsForProfile(profile) {
  return PROFILE_FEATURE_ROUNDS[profile] || PROFILE_FEATURE_ROUNDS.preloan_credit_card;
}

function normalizeState(state = {}) {
  const source = state || {};
  const sourceWithSplitMigration = { ...source };
  if (!sourceWithSplitMigration.oos_values && String(sourceWithSplitMigration.oot_values || "").includes("DEV-OOS")) {
    const oldValues = list(sourceWithSplitMigration.oot_values);
    sourceWithSplitMigration.oos_values = oldValues.filter((item) => item.includes("OOS") && !item.startsWith("OOT")).join(", ");
    sourceWithSplitMigration.oot_values = oldValues.filter((item) => !item.includes("OOS") || item.startsWith("OOT")).join(", ");
  }
  const scenarioProfile = source.scenario_profile || (source.business_domain ? scenarioProfileForBusinessDomain(source.business_domain) : newDocumentDefaults.scenario_profile);
  const businessDomain = source.business_domain || businessDomainForProfile(scenarioProfile);
  const preset = profilePreset(scenarioProfile);
  const paramDefaults = paramDefaultsForProfile(scenarioProfile);
  const merged = {
    ...newDocumentDefaults,
    ...paramDefaults,
    ...preset,
    ...sourceWithSplitMigration,
    business_domain: businessDomain,
    scenario_profile: scenarioProfile,
  };

  STAGE_GROUPS.forEach(({ stage, key }) => {
    if (Object.prototype.hasOwnProperty.call(source, key)) return;
    if (source.stage_steps && Array.isArray(source.stage_steps[stage])) {
      merged[key] = source.stage_steps[stage].map((step) => STEP_ALIASES[step] || step);
      return;
    }
    const legacyStage = Object.entries(STAGE_ALIASES).find(([, target]) => target === stage)?.[0];
    if (legacyStage && source.stage_steps && Array.isArray(source.stage_steps[legacyStage])) {
      merged[key] = source.stage_steps[legacyStage].map((step) => STEP_ALIASES[step] || step);
      return;
    }
    merged[key] = preset[key] || [];
  });

  Object.entries(paramDefaults).forEach(([key, value]) => {
    if (!Object.prototype.hasOwnProperty.call(source, key)) {
      merged[key] = value;
    }
  });
  if (!Object.prototype.hasOwnProperty.call(source, "feature_steps")) {
    merged.feature_steps = featureRoundsForProfile(scenarioProfile);
  }
  merged.feature_steps = (merged.feature_steps || []).map((step) => (step === "d01_d02" ? "prescreen" : step));
  STAGE_GROUPS.forEach(({ key }) => {
    merged[key] = (merged[key] || []).map((step) => STEP_ALIASES[step] || step);
  });

  return merged;
}

function stageStepsForState(state) {
  const stageSteps = {};
  STAGE_GROUPS.forEach(({ stage, key }) => {
    const values = [...(SYSTEM_DEFAULT_STAGE_STEPS[key] || []), ...(state[key] || [])].filter(
      (value, index, array) => value && array.indexOf(value) === index,
    );
    if (values.length) stageSteps[stage] = values;
  });
  return stageSteps;
}

function featureRoundsForState(state) {
  const rounds = [];
  if ((state.feature_metadata_steps || []).length) rounds.push("metadata");
  if ((state.feature_prescreen_steps || []).length) rounds.push("prescreen");
  if ((stageStepsForState(state).feature_refine || []).length) rounds.push("refine");
  return rounds;
}

function selectedStepIds(stageSteps) {
  return new Set(Object.values(stageSteps).flat());
}

function addStepParam(params, stepId, key, value) {
  if (value === "" || value === null || value === undefined) return;
  params[stepId] = params[stepId] || {};
  params[stepId][key] = value;
}

function buildStepParams(state) {
  const stageSteps = stageStepsForState(state);
  const selectedSteps = selectedStepIds(stageSteps);
  const params = {};

  if (selectedSteps.has("monthly_label_distribution")) {
    addStepParam(params, "monthly_label_distribution", "min_samples_per_month", state.sample_min_monthly_count);
  }
  if (selectedSteps.has("feature_quality_prescreen")) {
    addStepParam(params, "feature_quality_prescreen", "require_sql_approval", Boolean(state.require_sql_approval));
  }
  if (selectedSteps.has("sql_review_gate")) {
    addStepParam(params, "sql_review_gate", "block_on_high_risk", Boolean(state.sql_block_high_risk));
  }
  if (selectedSteps.has("missing_rate_filter")) {
    addStepParam(params, "missing_rate_filter", "threshold", state.missing_rate_threshold);
  }
  if (selectedSteps.has("constant_value_filter")) {
    addStepParam(params, "constant_value_filter", "max_unique_values", state.constant_max_unique_values);
  }
  if (selectedSteps.has("iv_filter")) {
    addStepParam(params, "iv_filter", "min_iv", state.iv_min);
  }
  if (selectedSteps.has("psi_filter")) {
    addStepParam(params, "psi_filter", "max_psi", state.psi_max);
  }
  if (selectedSteps.has("correlation_dedup")) {
    addStepParam(params, "correlation_dedup", "method", state.correlation_method);
    addStepParam(params, "correlation_dedup", "max_abs_corr", state.correlation_max_abs);
  }
  if (selectedSteps.has("gain_importance_filter")) {
    addStepParam(params, "gain_importance_filter", "tail_fraction", state.gain_tail_fraction);
    addStepParam(params, "gain_importance_filter", "max_auc_drop", state.gain_max_auc_drop);
  }
  if (selectedSteps.has("monthly_stability")) {
    addStepParam(params, "monthly_stability", "max_ks_std", state.monthly_max_ks_std);
  }
  if (selectedSteps.has("score_psi")) {
    addStepParam(params, "score_psi", "warn_psi", state.score_psi_warn);
  }
  if (selectedSteps.has("scale_pos_weight")) {
    addStepParam(params, "scale_pos_weight", "mode", "negative_over_positive");
  }

  return params;
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

function currentTemplateLabel() {
  const select = document.querySelector("#templateSelect");
  return select?.selectedOptions?.[0]?.textContent || "空白模板";
}

function setList(el, items, emptyText) {
  if (!el) return;
  el.innerHTML = "";
  const values = items.length ? items : [emptyText];
  values.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    if (!items.length) li.className = "empty";
    el.appendChild(li);
  });
}

function formatStepLabel(stepId) {
  return STEP_LABELS[stepId] || stepId;
}

function updateExecutionSummary(state) {
  const stageSteps = stageStepsForState(state);
  STAGE_GROUPS.forEach(({ stage }) => {
    const listEl = document.querySelector(`[data-stage-summary="${stage}"]`);
    const labels = (stageSteps[stage] || []).map(formatStepLabel);
    setList(listEl, labels, "本场景暂不配置该环节。");
  });
}

function buildStepSummary(state) {
  const stageSteps = stageStepsForState(state);
  return STAGE_GROUPS.flatMap(({ stage }) => {
    const values = stageSteps[stage] || [];
    return values.length ? [`${STAGE_LABELS[stage] || stage}: ${values.length} 项`] : [];
  });
}

function buildNextActions(state) {
  const actions = [];
  const stageStepCount = Object.values(stageStepsForState(state)).flat().length;
  const selectedFeatureMethods = (state.feature_refine_steps || []).filter((stepId) => FEATURE_METHOD_STEP_IDS.includes(stepId));
  const experimentCount = effectiveExperiments(state).length;

  if (!state.title) {
    actions.push("补充需求标题。");
  }
  if (!state.objective) {
    actions.push("写清建模目标和预期决策动作。");
  }
  if (!state.target_column || !state.split_column) {
    actions.push("补齐标签字段和切分字段。");
  }
  if (!stageStepCount) {
    actions.push("应用模板或切换业务域，让系统生成执行步骤。");
  }
  if (!selectedFeatureMethods.length) {
    actions.push("至少选择一个特征筛选方法。");
  }
  if (!experimentCount) {
    actions.push("新增实验，或写一段实验一句话描述。");
  }
  if (!state.metrics.length || !state.report_sections.length) {
    actions.push("选择评估指标和报告章节。");
  }
  if (!actions.length) {
    actions.push("预览 Markdown 后复制或下载。");
  }

  return actions.slice(0, 5);
}

function updateHelperPanel(state) {
  const step = FORM_STEPS.find((item) => item.id === activeSectionId) || FORM_STEPS[0];
  const stageSteps = stageStepsForState(state);
  const stepCount = Object.values(stageSteps).flat().length;
  const experimentCount = effectiveExperiments(state).length;
  const domainLabel = BUSINESS_DOMAIN_LABELS[state.business_domain] || state.business_domain || "未选择";
  const profileLabel = PROFILE_LABELS[state.scenario_profile] || state.scenario_profile || "未选择";
  const title = state.title || state.request_id || "空白需求";

  if (currentStageTitleEl) currentStageTitleEl.textContent = step.title;
  if (currentStageHintEl) currentStageHintEl.textContent = step.hint;
  if (summaryTextEl) {
    summaryTextEl.textContent = `${title}：${domainLabel}场景，任务模式为${state.task_mode}，使用 ${profileLabel} profile，已选择 ${stepCount} 个执行步骤、${experimentCount} 个实验、${state.metrics.length} 个评估指标。`;
  }
  if (domainBadgeEl) domainBadgeEl.textContent = domainLabel;
  if (profileBadgeEl) profileBadgeEl.textContent = profileLabel;
  if (templateBadgeEl) templateBadgeEl.textContent = currentTemplateLabel();
  if (stepCountBadgeEl) stepCountBadgeEl.textContent = String(stepCount);
  setList(stepSummaryEl, buildStepSummary(state), "暂未选择执行步骤。");
  setList(nextActionsEl, buildNextActions(state), "暂无待处理事项。");
  updateExecutionSummary(state);
}

function updateMethodParamAvailability() {
  document.querySelectorAll(".method-card").forEach((card) => {
    const checkbox = card.querySelector('.method-check input[type="checkbox"]');
    if (!checkbox) return;
    const disabled = !checkbox.checked;
    card.querySelectorAll(".method-param input, .method-param select").forEach((input) => {
      input.disabled = disabled;
    });
    card.classList.toggle("is-disabled", disabled);
  });
}

function setActiveSection(sectionId) {
  const nextStep = FORM_STEPS.find((step) => step.id === sectionId) || FORM_STEPS[0];
  activeSectionId = nextStep.id;

  FORM_STEPS.forEach((step) => {
    const section = document.querySelector(`#${step.id}`);
    const link = document.querySelector(`.nav-list a[href="#${step.id}"]`);
    const isActive = step.id === activeSectionId;
    if (section) {
      section.classList.toggle("is-active", isActive);
      section.hidden = !isActive;
    }
    if (link) {
      link.classList.toggle("active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "step");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  });

  const index = FORM_STEPS.findIndex((step) => step.id === activeSectionId);
  const prevButton = document.querySelector("#prevSection");
  const nextButton = document.querySelector("#nextSection");
  if (sectionProgressEl) sectionProgressEl.textContent = `${index + 1} / ${FORM_STEPS.length} ${nextStep.title}`;
  if (prevButton) prevButton.disabled = index === 0;
  if (nextButton) nextButton.textContent = index === FORM_STEPS.length - 1 ? "回到开头" : "下一步";
  updateHelperPanel(collectState());
}

function openMarkdownPreview() {
  update({ persist: false });
  markdownModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  document.querySelector("#closeMarkdownPreview").focus();
}

function closeMarkdownPreview() {
  markdownModal.classList.add("hidden");
  document.body.classList.remove("modal-open");
}

function getProjectName() {
  return currentProject || DEFAULT_PROJECT_NAME;
}

function setCheckboxGroup(name, values) {
  const selected = new Set(values || []);
  document.querySelectorAll(`[data-group="${name}"] input[type="checkbox"]`).forEach((input) => {
    input.checked = selected.has(input.value);
  });
}

function checkboxGroupHasInputs(name) {
  return document.querySelectorAll(`[data-group="${name}"] input[type="checkbox"]`).length > 0;
}

function getCheckboxGroup(name) {
  return Array.from(new Set(Array.from(document.querySelectorAll(`[data-group="${name}"] input[type="checkbox"]:checked`)).map((input) => input.value)));
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

function experimentRow(exp = { name: "", method: "xgboost", segment: "all", description: "" }) {
  const row = document.createElement("div");
  row.className = "experiment-row";
  row.innerHTML = `
    <input data-exp="name" placeholder="experiment_id" value="${escapeHtml(exp.name || "")}" />
    <select data-exp="method">
      <option value="xgboost">xgboost</option>
      <option value="lightgbm">lightgbm</option>
      <option value="logistic_regression">logistic_regression</option>
      <option value="custom">custom</option>
    </select>
    <input data-exp="segment" placeholder="segment" value="${escapeHtml(exp.segment || "all")}" />
    <button type="button" class="remove-experiment" title="删除实验">×</button>
    <textarea data-exp="description" class="experiment-description" rows="2" placeholder="实验说明（可选）">${escapeHtml(exp.description || "")}</textarea>
  `;
  row.querySelector('[data-exp="method"]').value = exp.method || "xgboost";
  row.querySelector(".remove-experiment").addEventListener("click", () => {
    row.remove();
    update();
  });
  row.querySelectorAll("input, select, textarea").forEach((el) => el.addEventListener("input", update));
  return row;
}

function setExperiments(items) {
  experimentsEl.innerHTML = "";
  (Array.isArray(items) ? items : [{ name: "baseline_all", method: "xgboost", segment: "all" }]).forEach((item) => {
    experimentsEl.appendChild(experimentRow(item));
  });
}

function getExperiments() {
  return Array.from(experimentsEl.querySelectorAll(".experiment-row"))
    .map((row) => ({
      name: row.querySelector('[data-exp="name"]').value.trim(),
      method: row.querySelector('[data-exp="method"]').value,
      segment: row.querySelector('[data-exp="segment"]').value.trim() || "all",
      description: row.querySelector('[data-exp="description"]').value.trim(),
    }))
    .filter((item) => item.name);
}

function effectiveExperiments(state) {
  if (state.experiments.length) return state.experiments;
  const description = String(state.experiment_description || "").trim();
  if (!description) return [];
  return [
    {
      name: "baseline_from_description",
      method: "custom",
      segment: "all",
      description,
    },
  ];
}

function collectState() {
  const state = {
    request_id: ensureRequestId(field("request_id")),
    title: field("title"),
    owner: field("owner"),
    workflow: field("workflow"),
    task_mode: taskModeLabel(field("workflow")),
    business_domain: field("business_domain"),
    scenario_profile: field("scenario_profile") || scenarioProfileForBusinessDomain(field("business_domain")),
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
    oos_values: field("oos_values"),
    oot_values: field("oot_values"),
    sample_definition: field("sample_definition"),
    feature_steps: [],
    feature_sources: field("feature_sources"),
    require_sql_approval: true,
    feature_notes: field("feature_notes"),
    candidate_targets: field("candidate_targets"),
    sample_variants: field("sample_variants"),
    experiment_description: field("experiment_description"),
    experiments: getExperiments(),
    modeling_notes: field("modeling_notes"),
    metrics: getCheckboxGroup("metrics"),
    champions: field("champions"),
    comparison_dimensions: field("comparison_dimensions"),
    risk_profile_dimensions: field("risk_profile_dimensions"),
    report_sections: getCheckboxGroup("report_sections"),
    report_outputs: field("report_outputs"),
    extra_notes: field("extra_notes"),
    sample_min_monthly_count: field("sample_min_monthly_count"),
    missing_rate_threshold: field("missing_rate_threshold"),
    constant_max_unique_values: field("constant_max_unique_values"),
    iv_min: field("iv_min"),
    psi_max: field("psi_max"),
    correlation_method: field("correlation_method"),
    correlation_max_abs: field("correlation_max_abs"),
    gain_tail_fraction: field("gain_tail_fraction"),
    gain_max_auc_drop: field("gain_max_auc_drop"),
    score_psi_warn: field("score_psi_warn"),
    monthly_max_ks_std: field("monthly_max_ks_std"),
    sql_block_high_risk: field("sql_block_high_risk"),
  };

  const preset = profilePreset(state.scenario_profile);
  STAGE_GROUPS.forEach(({ key }) => {
    if (checkboxGroupHasInputs(key)) {
      state[key] = getCheckboxGroup(key);
    } else if (Array.isArray(stageGroupSelections[key])) {
      state[key] = [...stageGroupSelections[key]];
    } else {
      state[key] = [...(preset[key] || [])];
    }
    stageGroupSelections[key] = [...state[key]];
  });
  state.feature_steps = featureRoundsForState(state);

  return state;
}

function buildMarkdown(state) {
  const stageSteps = stageStepsForState(state);
  const stepParams = buildStepParams(state);
  const experiments = effectiveExperiments(state);
  const idColumnValues = list(state.id_columns);
  const optionalDataLines = [];
  if (state.feature_location) optionalDataLines.push(`feature_location: ${yamlScalar(state.feature_location)}`);
  if (idColumnValues.length) {
    optionalDataLines.push("id_columns:");
    optionalDataLines.push(yamlList(idColumnValues, 2));
  }
  if (state.period_column) optionalDataLines.push(`period_column: ${yamlScalar(state.period_column)}`);
  const experimentDescriptionLines = state.experiment_description
    ? [`experiment_description: ${yamlScalar(state.experiment_description)}`]
    : [];
  const experimentLines = experiments.length
    ? [
        "experiments:",
        ...experiments.flatMap((exp) => {
          const lines = [
            `  - name: ${yamlScalar(exp.name)}`,
            `    method: ${yamlScalar(exp.method)}`,
            `    segment: ${yamlScalar(exp.segment)}`,
          ];
          if (exp.description) {
            lines.push(`    description: ${yamlScalar(exp.description)}`);
          }
          return lines;
        }),
      ]
    : ["experiments: []"];
  const lines = [
    "---",
    `request_id: ${yamlScalar(state.request_id)}`,
    `title: ${yamlScalar(state.title)}`,
    `project: ${yamlScalar(state.project)}`,
    `workflow: ${yamlScalar(state.workflow)}`,
    `task_mode: ${yamlScalar(taskModeLabel(state.workflow))}`,
    `owner: ${yamlScalar(state.owner)}`,
    `business_domain: ${yamlScalar(state.business_domain)}`,
    `scenario_profile: ${yamlScalar(state.scenario_profile)}`,
    "",
    `sample_location: ${yamlScalar(state.sample_location)}`,
    `target_column: ${yamlScalar(state.target_column)}`,
    `time_column: ${yamlScalar(state.time_column)}`,
    ...optionalDataLines,
    `split_column: ${yamlScalar(state.split_column)}`,
    "splits:",
    "  dev:",
    "    values:",
    yamlList(list(state.dev_values), 6),
    "  oos:",
    "    values:",
    yamlList(list(state.oos_values), 6),
    "  oot:",
    "    values:",
    yamlList(list(state.oot_values), 6),
    "",
    "sample_checks:",
    yamlList(["sample_check_profile", "sample_check_stability"], 2),
    "",
    "stage_steps:",
    yamlMapOfLists(stageSteps, 2),
    "",
    "step_params:",
    yamlMapOfMaps(stepParams, 2),
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
    ...experimentDescriptionLines,
    ...experimentLines,
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
    multiline(state.feature_notes),
    "",
    "# 建模实验要求",
    "",
    state.experiment_description ? `实验描述：${state.experiment_description}` : "实验描述：待补充。",
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

function update({ persist = true } = {}) {
  const state = collectState();
  const markdown = buildMarkdown(state);
  preview.textContent = markdown;
  filenameHint.textContent = `${state.request_id || "model_request"}.md`;
  updateMethodParamAvailability();
  updateHelperPanel(state);
  if (persist) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }
  updateRestoreButton();
}

function savedStateExists() {
  return Boolean(localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_STORAGE_KEY));
}

function updateRestoreButton() {
  const restoreButton = document.querySelector("#restoreDraft");
  if (restoreButton) restoreButton.disabled = !savedStateExists();
}

function loadSavedState() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {
    try {
      return JSON.parse(saved);
    } catch (error) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
  }

  const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
  if (!legacy) return null;

  try {
    const parsed = JSON.parse(legacy);
    localStorage.setItem(STORAGE_KEY, legacy);
    return parsed;
  } catch (error) {
    localStorage.removeItem(LEGACY_STORAGE_KEY);
    return null;
  }
}

function loadCustomTemplates() {
  const saved = localStorage.getItem(CUSTOM_TEMPLATE_STORAGE_KEY);
  if (!saved) return [];

  try {
    const parsed = JSON.parse(saved);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => item && item.id && item.name && item.state);
  } catch (error) {
    localStorage.removeItem(CUSTOM_TEMPLATE_STORAGE_KEY);
    return [];
  }
}

function saveCustomTemplates(templates) {
  localStorage.setItem(CUSTOM_TEMPLATE_STORAGE_KEY, JSON.stringify(templates));
}

function optionExists(select, value) {
  return Array.from(select.options).some((option) => option.value === value);
}

function renderTemplateOptions(selectedValue = "blank") {
  const select = document.querySelector("#templateSelect");
  select.innerHTML = "";

  Object.entries(BUILT_IN_TEMPLATES).forEach(([id, template]) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = template.label;
    select.appendChild(option);
  });

  const customTemplates = loadCustomTemplates();
  if (customTemplates.length) {
    const group = document.createElement("optgroup");
    group.label = "自定义模板";
    customTemplates.forEach((template) => {
      const option = document.createElement("option");
      option.value = `custom:${template.id}`;
      option.textContent = template.name;
      group.appendChild(option);
    });
    select.appendChild(group);
  }

  select.value = optionExists(select, selectedValue) ? selectedValue : "blank";
  updateTemplateControls();
}

function selectedCustomTemplateId() {
  const value = document.querySelector("#templateSelect").value;
  return value.startsWith("custom:") ? value.slice("custom:".length) : "";
}

function getSelectedTemplate() {
  const value = document.querySelector("#templateSelect").value;
  if (value.startsWith("custom:")) {
    const id = value.slice("custom:".length);
    return loadCustomTemplates().find((template) => template.id === id) || null;
  }
  return BUILT_IN_TEMPLATES[value] || null;
}

function updateTemplateControls() {
  const customId = selectedCustomTemplateId();
  const deleteButton = document.querySelector("#deleteTemplate");
  deleteButton.disabled = !customId;

  if (customId) {
    const template = loadCustomTemplates().find((item) => item.id === customId);
    if (template) document.querySelector("#templateName").value = template.name;
  }
}

function applyState(state, { persist = true } = {}) {
  const normalized = normalizeState(state);
  normalized.request_id = ensureRequestId(normalized.request_id);
  currentProject = normalized.project || DEFAULT_PROJECT_NAME;
  stageGroupSelections = Object.fromEntries(
    STAGE_GROUPS.map(({ key }) => [key, Array.isArray(normalized[key]) ? [...normalized[key]] : []]),
  );
  Object.entries(normalized).forEach(([key, value]) => {
    if ([...CHECKBOX_GROUPS, "experiments", "stage_steps"].includes(key)) return;
    setField(key, value);
  });
  CHECKBOX_GROUPS.forEach((key) => {
    setCheckboxGroup(key, normalized[key] || []);
  });
  setExperiments(Array.isArray(normalized.experiments) ? normalized.experiments : []);
  update({ persist });
}

function applyBusinessDomainDefaults() {
  const businessDomain = field("business_domain") || newDocumentDefaults.business_domain;
  const profile = scenarioProfileForBusinessDomain(businessDomain);
  const preset = profilePreset(profile);
  const paramDefaults = paramDefaultsForProfile(profile);

  setField("scenario_profile", profile);
  STAGE_GROUPS.forEach(({ key }) => {
    const values = preset[key] || [];
    stageGroupSelections[key] = [...values];
    setCheckboxGroup(key, values);
  });
  setCheckboxGroup("feature_steps", featureRoundsForProfile(profile));
  Object.entries(paramDefaults).forEach(([key, value]) => {
    setField(key, value);
  });
  update();
}

async function copyMarkdown() {
  update({ persist: false });
  try {
    await navigator.clipboard.writeText(preview.textContent);
    showToast("Markdown 已复制");
  } catch (error) {
    const helper = document.createElement("textarea");
    helper.value = preview.textContent;
    helper.setAttribute("readonly", "");
    helper.style.position = "fixed";
    helper.style.opacity = "0";
    document.body.appendChild(helper);
    helper.select();
    document.execCommand("copy");
    helper.remove();
    showToast("Markdown 已复制");
  }
}

function downloadMarkdown() {
  update({ persist: false });
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

function applySelectedTemplate() {
  const template = getSelectedTemplate();
  if (!template) {
    showToast("未找到模板");
    return;
  }
  applyState(template.state);
  showToast("已应用模板");
}

function saveCurrentTemplate() {
  const nameInput = document.querySelector("#templateName");
  const name = nameInput.value.trim();
  if (!name) {
    showToast("请输入模板名称");
    nameInput.focus();
    return;
  }

  const selectedId = selectedCustomTemplateId();
  const templates = loadCustomTemplates();
  const existingByName = templates.find((template) => template.name === name);
  const id = selectedId || existingByName?.id || `custom-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const nextTemplate = {
    id,
    name,
    state: collectState(),
    updated_at: new Date().toISOString(),
  };
  const nextTemplates = templates.filter((template) => template.id !== id);
  nextTemplates.push(nextTemplate);
  saveCustomTemplates(nextTemplates);
  renderTemplateOptions(`custom:${id}`);
  showToast("已保存自定义模板");
}

function deleteSelectedTemplate() {
  const id = selectedCustomTemplateId();
  if (!id) return;

  const nextTemplates = loadCustomTemplates().filter((template) => template.id !== id);
  saveCustomTemplates(nextTemplates);
  document.querySelector("#templateName").value = "";
  renderTemplateOptions("blank");
  showToast("已删除自定义模板");
}

function restoreDraft() {
  const saved = loadSavedState();
  if (!saved) {
    updateRestoreButton();
    showToast("没有可恢复的草稿");
    return;
  }
  applyState(saved);
  showToast("已恢复草稿");
}

function newBlankDocument() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(LEGACY_STORAGE_KEY);
  document.querySelector("#templateSelect").value = "blank";
  document.querySelector("#templateName").value = "";
  updateTemplateControls();
  applyState(blankTemplateDefaults, { persist: false });
  showToast("已新建空白文档");
}

document.querySelector("#templateSelect").addEventListener("change", () => {
  updateTemplateControls();
  updateHelperPanel(collectState());
});
document.querySelector("#applyTemplate").addEventListener("click", applySelectedTemplate);
document.querySelector("#saveTemplate").addEventListener("click", saveCurrentTemplate);
document.querySelector("#deleteTemplate").addEventListener("click", deleteSelectedTemplate);
document.querySelector("#restoreDraft").addEventListener("click", restoreDraft);
document.querySelector("#newBlankRequest").addEventListener("click", newBlankDocument);
document.querySelector("#previewMarkdown").addEventListener("click", openMarkdownPreview);
document.querySelector("#copyMarkdown").addEventListener("click", copyMarkdown);
document.querySelector("#downloadMarkdown").addEventListener("click", downloadMarkdown);
document.querySelector("#closeMarkdownPreview").addEventListener("click", closeMarkdownPreview);
document.querySelectorAll(".info-tip").forEach((button) => {
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
});
markdownModal.addEventListener("click", (event) => {
  if (event.target === markdownModal) closeMarkdownPreview();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !markdownModal.classList.contains("hidden")) {
    closeMarkdownPreview();
  }
});
document.querySelector("#addExperiment").addEventListener("click", () => {
  experimentsEl.appendChild(experimentRow({ name: "", method: "xgboost", segment: "all" }));
  update();
});
document.querySelector("#prevSection").addEventListener("click", () => {
  const index = FORM_STEPS.findIndex((step) => step.id === activeSectionId);
  setActiveSection(FORM_STEPS[Math.max(0, index - 1)].id);
});
document.querySelector("#nextSection").addEventListener("click", () => {
  const index = FORM_STEPS.findIndex((step) => step.id === activeSectionId);
  const nextIndex = index === FORM_STEPS.length - 1 ? 0 : index + 1;
  setActiveSection(FORM_STEPS[nextIndex].id);
});

form.addEventListener("input", update);
form.addEventListener("change", (event) => {
  if (event.target.name === "business_domain") {
    applyBusinessDomainDefaults();
    return;
  }
  update();
});

document.querySelectorAll(".nav-list a").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    setActiveSection(link.getAttribute("href").slice(1));
  });
});

renderTemplateOptions("blank");
applyState(blankTemplateDefaults, { persist: false });
setActiveSection("basic");
