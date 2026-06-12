# 风险场景 AI 建模工作台需求生成器

这是一个纯静态页面，用于把用户选择和填空生成 `model_request.md`。
页面会同时保留历史字段（例如 `sample_checks`、`feature_selection.rounds`、
`workflow`、`experiments`、`evaluation`）并追加面向业务填写的
`task_mode`、`business_domain`、`scenario_profile`、`stage_steps` 和
`step_params`，用于驱动三域能力沉淀后的规划层。

页面会隐藏技术性的 `request_id` 并自动生成；`workflow` 在界面上显示为
中文“任务模式”，导出的 Markdown 仍保留底层 `workflow` 值以兼容 `rmw`。
样本页只要求用户填写样本位置、标签字段、时间字段、切分字段和
DEV/OOS/OOT 取值；`id_columns` 可由 project.yml 的数据契约提供。
真实拉数前 SQL 审批默认开启，不作为普通表单开关暴露。

页面默认打开为空白需求，不会自动应用 GCard 模板，也不会自动恢复本地草稿。
用户可以按需执行这些显式动作：

- `应用模板`：把下拉框选中的模板内容写入当前表单，例如空白模板、默认模板或复借 G 卡主模型。
- `保存为模板`：把当前表单配置保存为浏览器本地自定义模板，后续打开页面会自动读出。
- `恢复草稿`：读取浏览器本地保存的上一份编辑草稿。

页面上的业务域固定为 `获客 / 贷前 / 贷中风险 / 贷中经营` 四类；
`scenario_profile` 会按业务域自动映射，模板只是初始化表单的一种来源。
如果只填写“实验一句话描述”而不添加结构化实验，页面会导出一个
`baseline_from_description` 兼容实验。
页面本身不直接和某个 project workspace 交互；生成的 Markdown 仍保留
`project` front matter 占位，实际校验和执行时以 `rmw --project <project>` 为准。

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
