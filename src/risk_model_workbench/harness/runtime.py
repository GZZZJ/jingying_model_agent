"""Runtime helpers for recording harnessed stage action execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, TypeVar

from risk_model_workbench.harness.actions import ActionSpec, get_action_spec
from risk_model_workbench.harness.errors import (
    DATA_MISSING,
    DEPENDENCY_MISSING,
    FAILURE_CODES,
    SCAFFOLD_ONLY,
    SQL_APPROVAL_REQUIRED,
    TRANSIENT_IO,
    UNKNOWN,
    get_failure_class,
)
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import (
    load_run_state,
    mark_stage_done,
    mark_stage_failed,
    mark_stage_started,
    register_artifact as state_register_artifact,
    save_run_state,
)


T = TypeVar("T")


@dataclass(frozen=True)
class ActionResult:
    status: str
    failure_code: str = ""
    message: str = ""
    retry_count: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def register_action_artifact(
    run_path: str | Path,
    action_id: str,
    artifact: str | Path,
    *,
    kind: str = "file",
    source: str = "generated",
    description: str = "",
) -> dict[str, Any]:
    """Register an artifact through the action harness."""
    spec = _require_stage_action(action_id)
    entry = state_register_artifact(
        run_path,
        str(spec.stage),
        artifact,
        kind=kind,
        source=source,
        description=description,
    )
    state = load_run_state(run_path)
    stage_state = state.setdefault("stages", {}).setdefault(str(spec.stage), {"status": "pending", "artifacts": []})
    stage_state["action"] = _action_metadata(spec)
    if isinstance(stage_state.get("last_result"), dict):
        stage_state["last_result"]["artifacts"] = _stage_artifact_results(run_path, state, str(spec.stage), spec)
    save_run_state(run_path, state)
    return _artifact_result(entry, spec)


def stage_action_started(run_path: str | Path, action_id: str) -> dict[str, Any]:
    spec = _require_stage_action(action_id)
    state = mark_stage_started(run_path, str(spec.stage))
    return _write_stage_action_metadata(run_path, state, spec, result=None)


def stage_action_done(
    run_path: str | Path,
    action_id: str,
    *,
    scaffold: bool = False,
    message: str = "",
    failure_code: str = "",
    retry_count: int = 0,
) -> dict[str, Any]:
    spec = _require_stage_action(action_id)
    state = mark_stage_done(run_path, str(spec.stage), scaffold=scaffold)
    result = ActionResult(
        status="scaffold" if scaffold else "done",
        failure_code=_normalize_failure_code(failure_code or (SCAFFOLD_ONLY if scaffold else "")),
        message=message,
        retry_count=retry_count,
        artifacts=_stage_artifact_results(run_path, state, str(spec.stage), spec),
        decision=_latest_stage_decision(state, str(spec.stage)),
    )
    state = _write_stage_action_metadata(run_path, state, spec, result=result)
    _emit_action_progress(run_path, spec, result)
    return state


def stage_action_failed(
    run_path: str | Path,
    action_id: str,
    reason: str,
    *,
    failure_code: str = "",
    retry_count: int = 0,
) -> dict[str, Any]:
    spec = _require_stage_action(action_id)
    normalized = _normalize_failure_code(failure_code or classify_exception_message(reason))
    state = mark_stage_failed(run_path, str(spec.stage), reason)
    result = ActionResult(
        status="failed",
        failure_code=normalized,
        message=reason,
        retry_count=retry_count,
        artifacts=_stage_artifact_results(run_path, state, str(spec.stage), spec),
        decision=_latest_stage_decision(state, str(spec.stage)),
    )
    state = _write_stage_action_metadata(run_path, state, spec, result=result)
    _emit_action_progress(run_path, spec, result)
    return state


def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, (FileNotFoundError, KeyError, ValueError)):
        return DATA_MISSING
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return DEPENDENCY_MISSING
    if isinstance(exc, (TimeoutError, OSError)):
        return TRANSIENT_IO
    return classify_exception_message(str(exc))


def classify_exception_message(message: str) -> str:
    lowered = message.lower()
    if "approval" in lowered or "approve" in lowered or "sql_review_required" in lowered:
        return SQL_APPROVAL_REQUIRED
    if "dependency" in lowered or "no module named" in lowered or "import" in lowered:
        return DEPENDENCY_MISSING
    if "not available" in lowered or "missing" in lowered or "not found" in lowered or "does not exist" in lowered:
        return DATA_MISSING
    if "timeout" in lowered or "temporar" in lowered or "transient" in lowered:
        return TRANSIENT_IO
    if "scaffold" in lowered:
        return SCAFFOLD_ONLY
    return UNKNOWN


def should_retry_failure(action_id: str, failure_code: str, *, attempt: int, max_attempts: int = 3) -> bool:
    spec = get_action_spec(action_id)
    code = _normalize_failure_code(failure_code)
    if attempt >= max_attempts:
        return False
    if spec.retry_policy == "never":
        return False
    return get_failure_class(code).retryable


def run_with_retry(action_id: str, operation: Callable[[], T], *, max_attempts: int = 3) -> tuple[T, int]:
    """Run a safe operation according to the action retry policy.

    The helper never retries unknown or non-retryable failures, and write-stage
    actions keep ``retry_policy=never`` unless their ActionSpec explicitly says
    otherwise.
    """
    attempt = 1
    while True:
        try:
            return operation(), attempt - 1
        except Exception as exc:
            failure_code = classify_exception(exc)
            if not should_retry_failure(action_id, failure_code, attempt=attempt, max_attempts=max_attempts):
                raise
            attempt += 1


def _write_stage_action_metadata(
    run_path: str | Path,
    state: dict[str, Any],
    spec: ActionSpec,
    *,
    result: ActionResult | None,
) -> dict[str, Any]:
    stage_state = state.setdefault("stages", {}).setdefault(str(spec.stage), {"status": "pending", "artifacts": []})
    stage_state["action"] = _action_metadata(spec)
    if result is not None:
        stage_state["last_result"] = result.to_dict()
        if result.failure_code:
            stage_state["failure_code"] = result.failure_code
        elif stage_state.get("failure_code"):
            stage_state.pop("failure_code", None)
    save_run_state(run_path, state)
    return state


def _require_stage_action(action_id: str) -> ActionSpec:
    spec = get_action_spec(action_id)
    if spec.kind != "stage" or not spec.stage:
        raise ValueError(f"action is not a stage action: {action_id}")
    return spec


def _action_metadata(spec: ActionSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "kind": spec.kind,
        "approval_required": spec.approval_required,
        "approval_type": spec.approval_type,
        "retry_policy": spec.retry_policy,
        "expected_inputs": list(spec.inputs),
        "expected_outputs": list(spec.outputs),
        "artifact_rules": list(spec.artifact_rules),
    }


def _stage_artifact_results(
    run_path: str | Path,
    state: dict[str, Any],
    stage: str,
    spec: ActionSpec,
) -> list[dict[str, Any]]:
    stage_state = (state.get("stages") or {}).get(stage) or {}
    paths = [str(path) for path in stage_state.get("artifacts", [])]
    if not paths:
        return []
    manifest = load_artifact_manifest(run_path)
    manifest_by_path = {
        (str(item.get("stage")), str(item.get("path"))): item
        for item in manifest.get("artifacts", [])
        if item.get("path")
    }
    results = []
    for artifact_path in paths:
        entry = manifest_by_path.get((stage, artifact_path), {"path": artifact_path, "stage": stage})
        results.append(_artifact_result(entry, spec))
    return results


def _artifact_result(entry: dict[str, Any], spec: ActionSpec) -> dict[str, Any]:
    path = str(entry.get("path") or "")
    rule = _matching_artifact_rule(path, spec.artifact_rules)
    return {
        "path": path,
        "kind": entry.get("kind", "file"),
        "source": entry.get("source", ""),
        "exists": bool(entry.get("exists", False)),
        "description": entry.get("description", ""),
        "artifact_rule": rule,
        "rule_matched": bool(rule),
    }


def _matching_artifact_rule(path: str, rules: tuple[str, ...]) -> str:
    for rule in rules:
        if path == rule or fnmatch(path, rule):
            return rule
    return ""


def _latest_stage_decision(state: dict[str, Any], stage: str) -> dict[str, Any] | None:
    for item in reversed(list(state.get("decisions") or [])):
        if item.get("stage") == stage:
            return dict(item)
    return None


def _emit_action_progress(run_path: str | Path, spec: ActionSpec, result: ActionResult) -> None:
    try:
        from risk_model_workbench.progress import emit_progress, stage_label

        metrics = {
            "action_id": spec.id,
            "failure_code": result.failure_code,
            "retry_count": result.retry_count,
            "artifact_count": len(result.artifacts),
        }
        if result.decision:
            metrics["decision"] = result.decision.get("decision", "")
        emit_progress(
            run_path,
            str(spec.stage),
            step="action_failed" if result.status == "failed" else "action_done",
            status=result.status,
            message=result.message or f"{stage_label(str(spec.stage))}{'失败' if result.status == 'failed' else '完成'}",
            percent=None if result.status == "failed" else 100,
            metrics=metrics,
            level="error" if result.status == "failed" else "info",
            emit_terminal=False,
        )
    except Exception:
        return


def _normalize_failure_code(code: str) -> str:
    if not code:
        return ""
    return code if code in FAILURE_CODES else UNKNOWN
