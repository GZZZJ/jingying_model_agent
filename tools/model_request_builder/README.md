# 风险场景 AI 建模工作台需求生成器

这是一个纯静态页面，用于把用户选择和填空生成 `model_request.md`。

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
