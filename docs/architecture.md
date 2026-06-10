# Architecture

`risk_model_workbench` is the 风险场景 AI 建模工作台, a local business modeling
workbench. Codex or Claude Code interprets user intent, while this repository
provides stable `rmw` commands, workflow definitions, run state, artifact
registration, and project templates.

The workbench is intentionally split into four layers:

1. User-facing request layer: a Markdown model request, usually generated from
   `tools/model_request_builder/index.html`.
2. Agent orchestration layer: Codex or Claude Code validates the request,
   creates an execution plan, initializes a run, and calls stable CLI commands.
3. Atomic capability layer: reusable modules under `src/risk_model_workbench/`
   implement sample checks, feature selection integration, training,
   evaluation, reporting, artifact registry, and run state.
4. Project workspace layer: every modeling attempt writes code snapshots,
   configs, intermediate outputs, final artifacts, and decisions under one
   `projects/<project>/runs/<run_id>/` directory.

Project-specific scripts are allowed during exploration, but once they become
repeatable they should be converted into CLI-backed modules. Legacy scripts
remain under `legacy_scripts/` as provenance, while the workbench API stays in
`src/risk_model_workbench/`.

The imported Fujie GCard `main_lgbm` run is the current real-project case for
this architecture. It proves that external project outputs can be normalized
into a standard run and then used to harden generic train, evaluate, and report
capabilities without making GCard the default workbench behavior.
