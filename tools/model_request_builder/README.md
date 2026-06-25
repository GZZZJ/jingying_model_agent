# 风险场景 AI 建模工作台需求生成器

这是一个纯静态页面，用于把用户选择和填空生成 `model_request.md`。
页面会同时保留历史字段（例如 `sample_checks`、`feature_selection.rounds`、
`workflow`、`experiments`、`evaluation`）并追加面向业务填写的
`task_mode`、`business_domain`、`scenario_profile`、`stage_steps` 和
`step_params`，用于驱动三域能力沉淀后的规划层。

页面会隐藏技术性的 `request_id` 并自动生成；`workflow` 在界面上显示为
中文“任务模式”，导出的 Markdown 仍保留底层 `workflow` 值以兼容 `rmw`。
建模场景 `scenario_profile` 在界面上显式选择，用于控制默认执行步骤、
参数和场景合同；复借 G 卡从 0 重跑应选择“复借 G 卡主模型”。
样本页要求用户确认样本位置、主键字段、标签字段、时间字段、分区字段、
切分字段和 DEV/OOS/OOT 取值，避免关键数据合同只藏在 project.yml 中。
样本页也会显式导出 `data_source_mode`：`remote_table` 表示 DP 表或
SQL 来源，`local_feather` 表示本地 `.feather` 文件。选择本地 feather
时，文件仅作为运行时输入，不能上传、复制进 Git 或注册为 tracked artifact。
真实拉数前 SQL 审批默认开启，不作为普通表单开关暴露。

页面默认打开为空白需求，不会自动应用 GCard 模板，也不会自动恢复本地草稿。
用户可以按需执行这些显式动作：

- `应用模板`：把下拉框选中的模板内容写入当前表单，例如空白模板、默认模板或复借 G 卡主模型。
- `保存为模板`：把当前表单配置保存为浏览器本地自定义模板，后续打开页面会自动读出。
- `恢复草稿`：读取浏览器本地保存的上一份编辑草稿。

页面上的业务域固定为 `获客 / 贷前 / 贷中风险 / 贷中经营` 四类；
切换业务域会映射到对应的通用场景；如果需要项目专用链路，应在“建模场景”
里显式选择，例如 `fujie_gcard_main_lgbm`。
如果只填写“实验一句话描述”而不添加结构化实验，页面会导出一个
`baseline_from_description` 兼容实验。
评估页的重点比较维度使用多选控件，中文选项会导出为稳定 token：
`split`、`month`、`segment`、`decile`。
页面本身不直接执行 project workspace；生成的 Markdown 会保留 `project`
front matter，实际校验和执行时仍以 `rmw --project <project>` 为准。

打开方式：

```bash
open tools/model_request_builder/index.html
```

生成的 Markdown 可以继续交给：

```bash
rmw request validate --project <project> --request <request.md>
rmw plan create --project <project> --request <request.md>
```

`jm` 是长期兼容别名，已有自动化可以继续使用；新文档和示例优先使用 `rmw`。
