# Workbench Glossary

This glossary defines shared terms for `risk_model_workbench` harness work.

## Run

A concrete workflow execution workspace under `projects/<project>/runs/<run_id>/`.

## Stage

A named workflow step tracked in `run_state.yml`.

## Stage Contract

Workflow YAML requirements that define the artifact evidence needed to close a
stage. Contracts are validated by `rmw workflow validate` and enforced by
`rmw run audit`.

## Artifact Manifest

`audit/artifact_manifest.json`, the registered artifact inventory for a run.
Loose files are not closure evidence until registered.

## Imported Evidence

Artifacts copied from historical or external executions. They can be real
historical evidence, but they are not local reproduction evidence.

## Scaffold Evidence

Placeholder artifacts generated when local data, predictions, or execution
inputs are unavailable. Scaffold evidence is useful for continuity, but cannot
close a stage under strict audit.

## Guardrail

A rule enforced by CLI behavior, audit, tests, or repository policy.

## Proposed Rule

A lesson promoted into `docs/workbench_rules.yml` that still needs an
implementation, test, or explicit decision before it is `enforced`.
