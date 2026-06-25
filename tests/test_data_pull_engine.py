import json

import pandas as pd
import pytest

from risk_model_workbench.dp_feather import load_or_fetch_dp_feather
from risk_model_workbench.data.pull_engine import (
    DataPullEngineUnavailable,
    SqlApprovalRequired,
    normalize_platform,
    pull_query_to_dataframe,
    select_data_pull_engine,
    write_execution_environment,
)


def test_normalize_platform_values():
    assert normalize_platform("Windows") == "windows"
    assert normalize_platform("Darwin") == "macos"
    assert normalize_platform("Linux") == "linux"
    assert normalize_platform("FreeBSD") == "other"


def test_select_default_engine_by_platform():
    windows = select_data_pull_engine(platform_system="Windows")
    macos = select_data_pull_engine(platform_system="Darwin")
    linux = select_data_pull_engine(platform_system="Linux")
    other = select_data_pull_engine(platform_system="FreeBSD")

    assert windows.engine is None
    assert macos.engine is None
    assert "not auto-selected" in windows.reason
    assert "not auto-selected" in macos.reason
    assert linux.engine == "tmlsqlclient"
    assert other.engine == "tmlsqlclient"


def test_local_feather_mode_bypasses_remote_pull_engine():
    selection = select_data_pull_engine(platform_system="Darwin", data_source_mode="local_feather")

    assert selection.engine is None
    assert selection.platform == "macos"
    assert "bypassed" in selection.reason


def test_override_and_availability_failure():
    def unavailable(_engine: str) -> bool:
        return False

    with pytest.raises(DataPullEngineUnavailable):
        select_data_pull_engine(
            platform_system="Linux",
            override="dp_cli",
            availability_checker=unavailable,
            require_available=True,
        )


def test_pull_query_requires_sql_approval_for_all_engines():
    class FakeClient:
        def sql(self, sql):
            raise AssertionError("client should not be called without approval")

    with pytest.raises(SqlApprovalRequired):
        pull_query_to_dataframe("select 1", engine="tmlsqlclient", client=FakeClient(), sql_approved=False)


def test_pull_query_to_dataframe_uses_fake_clients_after_approval():
    class FakeResult:
        def to_pandas(self):
            return {"rows": [1]}

    class FakeTmlClient:
        def __init__(self):
            self.queries = []

        def sql(self, sql):
            self.queries.append(sql)
            return FakeResult()

    class FakeDpClient:
        def __init__(self):
            self.queries = []

        def query_to_dataframe(self, sql):
            self.queries.append(sql)
            return {"rows": [2]}

    tml = FakeTmlClient()
    dp = FakeDpClient()

    assert pull_query_to_dataframe("select 1", engine="tmlsqlclient", client=tml, sql_approved=True) == {"rows": [1]}
    assert pull_query_to_dataframe("select 2", engine="dp_cli", client=dp, sql_approved=True) == {"rows": [2]}
    assert tml.queries == ["select 1"]
    assert dp.queries == ["select 2"]


def test_write_execution_environment_payload(tmp_path):
    run_dir = tmp_path / "run"
    selection = select_data_pull_engine(platform_system="Darwin", override="auto")

    path = write_execution_environment(run_dir, selection)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == run_dir / "feature_selection" / "execution_environment.json"
    assert payload["platform"] == "macos"
    assert payload["data_pull_engine"] is None
    assert payload["override"] == "auto"


def test_load_or_fetch_dp_feather_reads_approved_local_file_without_remote_pull(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    local_path = project_dir / "data" / "raw" / "sample.feather"
    local_path.parent.mkdir(parents=True)
    pd.DataFrame({"uid": [1, 2], "target": [0, 1]}).to_feather(local_path)

    df = load_or_fetch_dp_feather(
        project_dir=project_dir,
        sql="select * from mart.remote",
        dataset_id="sample",
        description="local sample",
        feather_path=project_dir / "data" / "local" / "unused.feather",
        metadata_path=project_dir / "data" / "profile" / "sample.json",
        approved_local_feather_path="data/raw/sample.feather",
    )

    assert df.shape == (2, 2)
    metadata = json.loads((project_dir / "data" / "profile" / "sample.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "approved_local_feather"
    assert metadata["storage"]["source_path"] == "data/raw/sample.feather"
