from pathlib import Path
import shutil

from openpyxl import load_workbook

from jingying_model_agent.cli import main
from jingying_model_agent.reporting.excel_report import REPORT_SHEETS, generate_excel_report


def test_report_scaffold():
    project = "projects/2026-05-fujie-gcard-v1"
    run_id = "pytest_report_scaffold"
    run_dir = Path(project) / "runs" / run_id
    try:
        main(["run", "init", "--project", project, "--workflow", "full_modeling", "--run-id", run_id, "--force"])
        assert main(["report", "--project", project, "--run-id", run_id]) == 0
        assert (run_dir / "reports" / "model_report.md").exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_imported_excel_report_layout(tmp_path):
    run_dir = Path("projects/2026-05-fujie-gcard-v1/runs/2026-06-imported-gcard-main-lgbm")
    output_path = tmp_path / "model_report.xlsx"

    generate_excel_report(
        eval_dir=run_dir / "evaluation",
        train_dir=run_dir / "modeling" / "main_lgbm",
        input_dir=run_dir / "modeling_input",
        feature_dir=run_dir / "feature_selection",
        output_path=output_path,
    )

    workbook = load_workbook(output_path)
    assert workbook.sheetnames == REPORT_SHEETS
    assert output_path.with_name("model_report_missing_results.md").exists()
    assert output_path.with_name("model_report.md").exists()
    assert output_path.with_name("model_report.html").exists()

    ws = workbook["模型效果-每月效果"]
    assert _find_cell(ws, "KS") is not None
    assert _find_cell(ws, "AUC") is not None
    assert _find_cell(ws, "本轮模型") is not None
    metric_cell = _cell_below_header(ws, "本轮模型")
    assert metric_cell is not None
    assert metric_cell.number_format == "0.000"
    assert not isinstance(metric_cell.value, str)

    screening = workbook["变量筛选过程和模型参数"]
    assert _find_cell(screening, "筛选方法") is not None
    assert _find_cell(screening, "分表基础预筛：缺失率 < 0.95，相关性 < 0.80，IV >= 0.005") is not None

    sloping = workbook["模型效果-模型sloping"]
    for header in ["分组", "占比", "累计发起率", "累计lift", "剩余发起率", "剩余lift"]:
        assert _find_cell(sloping, header) is not None

    assert len(workbook["模型效果-模型sloping"].conditional_formatting) == 0
    assert len(workbook["模型效果-意愿交叉风险（DEV-OOS）"].conditional_formatting) > 0
    assert len(workbook["模型稳定性"].conditional_formatting) > 0


def _cell_below_header(ws, header: str):
    for row in ws.iter_rows():
        for cell in row:
            if cell.value == header:
                return ws.cell(row=cell.row + 1, column=cell.column)
    return None


def _find_cell(ws, value: str):
    for row in ws.iter_rows():
        for cell in row:
            if cell.value == value:
                return cell
    return None
