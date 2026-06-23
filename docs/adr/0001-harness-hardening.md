# ADR 0001: Harness Hardening

- status: accepted
- date: 2026-06-18

## Context

The workbench relies on AI-assisted implementation and long-running modeling
runs. Soft instructions in AGENTS.md and skills help, but they are not enough to
prevent forgotten steps, weak evidence, or accidental closure of imported and
scaffold outputs.

## Decision

The reusable workbench will treat workflow contracts, run state, artifact
manifests, and rule registries as hard coordination surfaces.

- Workflow YAML may define `stage_contracts`.
- `rmw workflow validate` validates contract shape.
- `rmw run audit` validates stage closure against `run_state.yml`,
  `artifact_manifest.json`, and available workflow contracts.
- `rmw run audit --strict` returns non-zero unless the verdict is `complete`.
- `rmw run audit --json` provides a machine-readable feedback surface for
  agents and future dashboards.
- `rmw lesson promote` records reusable lessons in `docs/workbench_rules.yml`
  as proposed rules until they are implemented or explicitly accepted as
  enforced.

## Consequences

Imported and scaffold artifacts remain visible and useful for continuity, but
strict audit will not treat them as local reproduction evidence. Stage closure
now requires registered artifact evidence, so legacy commands that mark stages
done without registering artifacts will surface as incomplete until hardened.
