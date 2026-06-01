from functools import wraps
from io import StringIO
import sys


class _TeeStream:
    """同时写入原始 stdout 和 StringIO 缓冲区"""
    def __init__(self, original, buffer):
        self.original = original
        self.buffer = buffer

    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original.flush()
        self.buffer.flush()


class _CaptureContext:
    """capture_print 的上下文管理器实现"""
    def __enter__(self):
        from log_config import get_print_logger  # 延迟导入，确保 init_logging 已调用
        self._print_logger = get_print_logger()
        self._old_stdout = sys.stdout
        self._buffer = StringIO()
        sys.stdout = _TeeStream(self._old_stdout, self._buffer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._old_stdout
        printed_content = self._buffer.getvalue().strip()
        if printed_content:
            for line in printed_content.split('\n'):
                if line.strip():
                    self._print_logger.info(line)
        return False


def capture_print(func_or_none=None):
    """
    将 print 输出同时保留到 print.log（控制台输出不受影响）

    支持两种用法:
    1. 作为装饰器: @capture_print
    2. 作为上下文管理器: with capture_print():
    """
    if func_or_none is None:
        # with capture_print(): 上下文管理器模式
        return _CaptureContext()

    # @capture_print 装饰器模式
    func = func_or_none

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _CaptureContext():
            return func(*args, **kwargs)

    return wrapper
