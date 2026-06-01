"""
特征筛选 V2 主入口: 根据 config['steps'] 编排执行流程
"""
import gc
from procs.proc_01_prepare import Proc01Prepare
from procs.proc_02_select_d01 import Proc02SelectD01
from procs.proc_02_select_d01_merge import Proc02SelectD01Merge
from procs.proc_02_select_d02 import Proc02SelectD02PSI
from procs.proc_02_select_d03_d06 import Proc02SelectD03D06
from procs.proc_02_select_d07_d08 import Proc02SelectD07D08
from procs.proc_03_summary import Proc03Summary
from utils.log_config import init_logging, get_main_logger


def run_pipeline(config):
    """
    根据 config['steps'] 编排执行特征筛选流程
    steps 默认为 ['d01','d02','d03','d04','d05','d06','d07','d08']
    """
    # 初始化日志系统（必须在第一次使用 logger 之前）
    init_logging(config['project_path'])
    main_logger = get_main_logger()

    # 01 数据准备(始终执行)
    main_logger.info("=== 01 数据准备 ===")
    Proc01Prepare(config).run()
    gc.collect()

    steps = config.get('steps', ['d01', 'd02', 'd03', 'd04', 'd05', 'd06', 'd07', 'd08'])
    main_logger.info(f"执行步骤: {steps}")

    # d01: toad 筛选
    if 'd01' in steps:
        main_logger.info("=== d01 toad筛选 ===")
        Proc02SelectD01(config).run()
        gc.collect()
        # 多宽表时执行 merge
        if len(config['bigtable']) > 1:
            main_logger.info("=== d01 merge ===")
            Proc02SelectD01Merge(config).run()
            gc.collect()

    # d02: PSI 筛选
    if 'd02' in steps:
        main_logger.info("=== d02 PSI筛选 ===")
        Proc02SelectD02PSI(config).run()
        gc.collect()

    # d03-d06: 重要性相关筛选(任一在 steps 中就启动该 proc)
    if {'d03', 'd04', 'd05', 'd06'} & set(steps):
        main_logger.info("=== d03-d06 重要性筛选 ===")
        Proc02SelectD03D06(config).run()
        gc.collect()

    # d07-d08: WOE 筛选(任一在 steps 中就启动该 proc)
    if {'d07', 'd08'} & set(steps):
        main_logger.info("=== d07-d08 WOE筛选 ===")
        Proc02SelectD07D08(config).run()
        gc.collect()

    # 汇总报告(始终执行)
    main_logger.info("=== 03 汇总报告 ===")
    Proc03Summary(config).run()
    gc.collect()

    main_logger.info("=== Pipeline 完成 ===")
