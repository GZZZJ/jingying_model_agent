"""Permission-scoped tool registry for rmw commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from risk_model_workbench.harness.actions import get_action_spec


@dataclass(frozen=True)
class ToolSpec:
    name: str
    action_id: str
    command: str
    permission: str
    description: str
    requires_approval: bool = False
    allowed_for_auditor: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="project_status",
        action_id="project_status",
        command="rmw project status --project <project>",
        permission="read_only",
        description="Read project continuity status.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="run_status",
        action_id="run_status",
        command="rmw run status --project <project> --run-id <run_id>",
        permission="read_only",
        description="Read run_state.yml.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="run_audit",
        action_id="run_audit",
        command="rmw run audit --project <project> --run-id <run_id>",
        permission="read_only",
        description="Audit run or stage evidence without mutation.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="workflow_validate",
        action_id="workflow_validate",
        command="rmw workflow validate --workflow <workflow>",
        permission="read_only",
        description="Validate workflow contracts.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="rules_list",
        action_id="rules_list",
        command="rmw rules list",
        permission="read_only",
        description="Read promoted workbench rules.",
        allowed_for_auditor=True,
    ),
    ToolSpec(
        name="sample_check",
        action_id="sample_check",
        command="rmw sample check --project <project> --run-id <run_id>",
        permission="writes_run",
        description="Write sample check artifacts and stage state.",
    ),
    ToolSpec(
        name="feature_metadata",
        action_id="feature_metadata",
        command="rmw feature metadata --project <project> --run-id <run_id>",
        permission="writes_run",
        description="Write feature metadata artifacts and stage state.",
    ),
    ToolSpec(
        name="feature_prescreen_dry_run",
        action_id="feature_prescreen",
        command="rmw feature prescreen --project <project> --run-id <run_id> --dry-run-sql",
        permission="writes_run",
        description="Generate feature prescreen SQL review artifacts without DP pull.",
    ),
    ToolSpec(
        name="feature_prescreen_pull",
        action_id="feature_prescreen",
        command="rmw feature prescreen --project <project> --run-id <run_id> --sql-approved",
        permission="dp_sql_pull",
        description="Run feature prescreening with approved DP/SQL access.",
        requires_approval=True,
    ),
    ToolSpec(
        name="build_wide_sql",
        action_id="build_wide_sql",
        command="rmw build-wide-sql --project <project> --run-id <run_id>",
        permission="writes_run",
        description="Generate wide-table SQL artifacts.",
    ),
    ToolSpec(
        name="feature_refine_dry_run",
        action_id="feature_refine",
        command="rmw feature refine --project <project> --run-id <run_id> --dry-run-sql",
        permission="writes_run",
        description="Generate feature refine SQL review artifacts without DP pull.",
    ),
    ToolSpec(
        name="feature_refine_pull",
        action_id="feature_refine",
        command="rmw feature refine --project <project> --run-id <run_id> --sql-approved",
        permission="dp_sql_pull",
        description="Run feature refinement with approved DP/SQL access.",
        requires_approval=True,
    ),
    ToolSpec(
        name="train_baseline",
        action_id="train_baseline",
        command="rmw train --project <project> --run-id <run_id> --experiment <name>",
        permission="writes_run",
        description="Train or scaffold baseline model artifacts.",
    ),
    ToolSpec(
        name="evaluate",
        action_id="evaluate",
        command="rmw evaluate --project <project> --run-id <run_id>",
        permission="writes_run",
        description="Write model evaluation artifacts.",
    ),
    ToolSpec(
        name="compare",
        action_id="compare",
        command="rmw compare --project <project> --run-id <run_id>",
        permission="writes_run",
        description="Write champion/challenger comparison artifacts.",
    ),
    ToolSpec(
        name="report",
        action_id="report",
        command="rmw report --project <project> --run-id <run_id>",
        permission="writes_run",
        description="Write report artifacts.",
    ),
)

TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}
TOOL_ALIASES = {
    "feature_d01_d02_dry_run": "feature_prescreen_dry_run",
    "feature_d01_d02_pull": "feature_prescreen_pull",
}


def list_tool_specs(*, permission: str | None = None) -> tuple[ToolSpec, ...]:
    specs = TOOL_SPECS
    if permission:
        specs = tuple(spec for spec in specs if spec.permission == permission)
    return tuple(sorted(specs, key=lambda spec: spec.name))


def get_tool_spec(name: str) -> ToolSpec:
    name = TOOL_ALIASES.get(name, name)
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown tool: {name}") from exc


def validate_tool_registry() -> list[str]:
    errors: list[str] = []
    for spec in TOOL_SPECS:
        try:
            action = get_action_spec(spec.action_id)
        except KeyError:
            errors.append(f"tool {spec.name} references unknown action: {spec.action_id}")
            continue
        if spec.requires_approval and not action.approval_required:
            errors.append(f"tool {spec.name} requires approval but action {spec.action_id} does not")
        if spec.allowed_for_auditor and spec.permission != "read_only":
            errors.append(f"tool {spec.name} is auditor-allowed but permission is {spec.permission}")
    return errors


def tool_to_dict(spec: ToolSpec) -> dict[str, object]:
    return spec.to_dict()


def format_tool_list(specs: tuple[ToolSpec, ...]) -> str:
    lines = ["Tool Name                  Permission    Approval  Auditor  Action ID             Command"]
    lines.append("-" * 110)
    for spec in specs:
        approval = "yes" if spec.requires_approval else "no"
        auditor = "yes" if spec.allowed_for_auditor else "no"
        lines.append(
            f"{spec.name:<26} {spec.permission:<13} {approval:<9} {auditor:<8} {spec.action_id:<21} {spec.command}"
        )
    return "\n".join(lines) + "\n"


def format_tool_detail(spec: ToolSpec) -> str:
    lines = [
        f"tool: {spec.name}",
        f"action_id: {spec.action_id}",
        f"permission: {spec.permission}",
        f"requires_approval: {spec.requires_approval}",
        f"allowed_for_auditor: {spec.allowed_for_auditor}",
        f"command: {spec.command}",
        f"description: {spec.description}",
    ]
    return "\n".join(lines) + "\n"
