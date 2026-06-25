"""Static SQL review helpers for generated workbench SQL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_COMMENT_RE = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.S)
_DANGEROUS_RE = re.compile(
    r"\b(drop|delete|truncate|insert\s+overwrite|update|merge|alter|grant|revoke)\b",
    re.I,
)
_JOIN_RE = re.compile(r"\b(?:left|right|inner|full|cross)?\s+join\b", re.I)
_FUTURE_RE = re.compile(r"\b(future|after_label|post_label|label_future|next_|tomorrow)\w*\b", re.I)
_LEAKAGE_RE = re.compile(r"\b(label|target|bad|overdue|default|repay|risk)[A-Za-z0-9_]*(?:_score|_result|_flag|_amt)?\b", re.I)


@dataclass(frozen=True)
class SQLFinding:
    severity: str
    rule_id: str
    message: str
    evidence: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "rule_id": self.rule_id,
            "message": self.message,
            "evidence": self.evidence,
        }


def _strip_comments(sql: str) -> str:
    return _COMMENT_RE.sub(" ", sql)


def _compact(sql: str) -> str:
    return re.sub(r"\s+", " ", _strip_comments(sql)).strip()


def _join_clauses(compact_sql: str) -> list[str]:
    matches = list(_JOIN_RE.finditer(compact_sql))
    clauses: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(compact_sql)
        clauses.append(compact_sql[start:end])
    return clauses


def review_sql_text(
    sql: str,
    *,
    approved_for_execution: bool = False,
    target_columns: list[str] | None = None,
    time_columns: list[str] | None = None,
    allow_create_table_as: bool = True,
) -> dict[str, Any]:
    """Run conservative static checks over generated SQL.

    This gate is intentionally simple and dependency-free. It does not claim to
    prove SQL correctness; it blocks the obvious high-risk patterns that should
    never be bypassed by an approval flag.
    """
    compact_sql = _compact(sql)
    lowered = compact_sql.lower()
    findings: list[SQLFinding] = []

    if not compact_sql:
        findings.append(SQLFinding("high", "empty_sql", "SQL text is empty."))
    if ";" in compact_sql.rstrip(";"):
        findings.append(SQLFinding("high", "multi_statement", "Multiple SQL statements are not allowed."))

    dangerous = _DANGEROUS_RE.search(compact_sql)
    if dangerous:
        findings.append(
            SQLFinding(
                "high",
                "dangerous_statement",
                "SQL contains DDL/DML outside the allowed generated create-table flow.",
                dangerous.group(0),
            )
        )

    if not allow_create_table_as and re.search(r"\bcreate\s+table\b", lowered):
        findings.append(SQLFinding("high", "create_not_allowed", "CREATE TABLE is not allowed for this SQL gate."))
    if allow_create_table_as and not re.search(r"\bcreate\s+table\b.+\bas\s+select\b", lowered):
        findings.append(SQLFinding("medium", "non_ctas_sql", "Generated SQL is not a recognized create-table-as-select statement."))

    for clause in _join_clauses(compact_sql):
        clause_l = clause.lower()
        if re.search(r"\bcross\s+join\b", clause_l):
            findings.append(SQLFinding("high", "cross_join", "SQL contains CROSS JOIN.", clause[:180]))
        if " on " not in clause_l and " using " not in clause_l:
            findings.append(SQLFinding("high", "join_without_condition", "JOIN does not include ON or USING.", clause[:180]))
        if re.search(r"\bon\s+(?:1\s*=\s*1|true)\b", clause_l):
            findings.append(SQLFinding("high", "cartesian_join_condition", "JOIN condition looks like a Cartesian join.", clause[:180]))

    if _FUTURE_RE.search(compact_sql):
        findings.append(
            SQLFinding(
                "high",
                "future_time_reference",
                "SQL references fields that look like future or post-label data.",
                _FUTURE_RE.search(compact_sql).group(0),
            )
        )

    expected_targets = {str(item).lower() for item in (target_columns or []) if item}
    expected_times = {str(item).lower() for item in (time_columns or []) if item}
    for match in _LEAKAGE_RE.finditer(compact_sql):
        token = match.group(0)
        token_l = token.lower()
        if token_l in expected_targets:
            continue
        if token_l in expected_times:
            continue
        if token_l in {"target", "label", "final_flag"}:
            continue
        findings.append(SQLFinding("medium", "possible_target_leakage", "SQL references a column name that may leak target information.", token))
        break

    if not approved_for_execution:
        findings.append(SQLFinding("medium", "approval_required", "SQL execution requires explicit approval."))

    high_count = sum(1 for finding in findings if finding.severity == "high")
    medium_count = sum(1 for finding in findings if finding.severity == "medium")
    return {
        "status": "failed" if high_count else "passed",
        "risk_level": "high" if high_count else "medium" if medium_count else "low",
        "high_risk": bool(high_count),
        "approved_for_execution": bool(approved_for_execution),
        "finding_counts": {"high": high_count, "medium": medium_count},
        "findings": [finding.as_dict() for finding in findings],
    }
