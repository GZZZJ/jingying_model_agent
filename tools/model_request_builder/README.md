# 模型需求生成器

这是一个纯静态页面，用于把用户选择和填空生成 `model_request.md`。

打开方式：

```bash
open tools/model_request_builder/index.html
```

生成的 Markdown 可以继续交给：

```bash
jm request validate --project <project> --request <request.md>
jm plan create --project <project> --request <request.md>
```
