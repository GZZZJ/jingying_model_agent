import logging
import logging.config
import sys
import os


# 全局标记：是否已初始化
_logging_initialized = False


def init_logging(project_path):
    """
    初始化日志系统，设置日志文件路径到项目的 logs 目录

    Args:
        project_path: 项目根目录路径
    """
    global _logging_initialized

    # 避免重复初始化
    if _logging_initialized:
        return

    # 创建 logs 目录
    logs_dir = os.path.join(project_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # 日志文件路径
    main_log_path = os.path.join(logs_dir, 'main.log')
    print_log_path = os.path.join(logs_dir, 'print.log')

    # 主日志配置
    main_log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'main_formatter': {
                'format': '[%(asctime)s] %(levelname)-7s %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'main_file': {
                'class': 'logging.FileHandler',
                'filename': main_log_path,
                'formatter': 'main_formatter',
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'main': {
                'handlers': ['main_file'],
                'level': 'INFO'
            }
        }
    }

    # Print日志配置
    print_log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'print_formatter': {
                'format': '[CAPTURED PRINT] %(asctime)s: %(message)s',
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {
            'print_file': {
                'class': 'logging.FileHandler',
                'filename': print_log_path,
                'formatter': 'print_formatter',
                'mode': 'a',
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'main.print_capture': {
                'handlers': ['print_file'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }

    # 应用配置
    logging.config.dictConfig(main_log_config)
    logging.config.dictConfig(print_log_config)

    _logging_initialized = True

    # 记录初始化信息
    logger = logging.getLogger('main')
    logger.info(f"日志系统已初始化，日志目录: {logs_dir}")


# 获取Logger的函数
def get_main_logger():
    return logging.getLogger('main')


def get_print_logger():
    return logging.getLogger('main.print_capture')
