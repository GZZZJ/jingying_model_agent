import json

from risk_model_workbench.data.table_profile import (
    build_profile_queries,
    build_static_table_profile,
    profile_remote_table,
    write_table_profile,
)


def test_profile_remote_table_uses_select_only_queries():
    calls = []

    class FakeClient:
        def query(self, sql):
            calls.append(sql)
            if "count(*) as row_count" in sql and "group by" not in sql:
                return [{"row_count": 1000}]
            if "group by final_flag" in sql:
                return [
                    {"split_value": "DEV", "row_count": 700},
                    {"split_value": "OOT", "row_count": 300},
                ]
            if "label_valid_count" in sql:
                return [{"label_valid_count": 950}]
            if "rand_flag0" in sql:
                return [{"column_name": "rand_flag0", "min_value": 0.0, "max_value": 0.999, "null_count": 0, "row_count": 1000}]
            raise AssertionError(sql)

    profile = profile_remote_table(
        "mart.sample",
        query_client=FakeClient(),
        split_column="final_flag",
        target_column="target",
        random_columns=["rand_flag0"],
    )

    assert all(sql.lstrip().lower().startswith("select") for sql in calls)
    assert profile["row_count"] == 1000
    assert profile["split_distribution"] == {"DEV": 700, "OOT": 300}
    assert profile["label_valid_count"] == 950
    assert profile["random_columns"][0]["column"] == "rand_flag0"


def test_build_profile_queries_never_emit_ctas_or_ddl():
    queries = build_profile_queries(
        "mart.sample",
        split_column="final_flag",
        target_column="target",
        random_columns=["rand_flag0"],
    )

    assert queries
    assert all(query.sql.lower().startswith("select") for query in queries)
    assert not any("create table" in query.sql.lower() for query in queries)


def test_write_static_table_profile(tmp_path):
    profile = build_static_table_profile(
        "mart.wide",
        row_count=None,
        column_count=103,
        feature_count=100,
        source="wide_sql_execution_metadata",
        status="metadata_only_after_ctas",
    )
    path = write_table_profile(profile, tmp_path / "feature_selection" / "profiles" / "wide_table_profile.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["table"] == "mart.wide"
    assert payload["candidate_feature_count"] == 100
    assert payload["status"] == "metadata_only_after_ctas"
