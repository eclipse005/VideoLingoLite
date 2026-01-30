import functools
import os
import time
from typing import Any, Optional
from rich import print as rprint
from rich.console import Console

console = Console()

# ------------------------------
# retry decorator
# ------------------------------

def except_handler(error_msg, retry=0, delay=1, default_return=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retry + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    rprint(f"[red]{error_msg}: {e}, retry: {i+1}/{retry}[/red]")
                    if i == retry:
                        if default_return is not None:
                            return default_return
                        raise last_exception
                    time.sleep(delay * (2**i))
        return wrapper
    return decorator


# ------------------------------
# timer decorator
# ------------------------------

def format_duration(seconds: float) -> str:
    """
    将秒数格式化为 时:分:秒 格式

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串，如 "1:23:45" 或 "23:45" 或 "45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    elif minutes > 0:
        return f"{minutes}:{secs:02d}"
    else:
        return f"{secs}s"


def timer(name: Optional[str] = None):
    """
    计时装饰器：自动统计并打印函数执行时间

    Args:
        name: 计时名称，如果为 None 则使用函数名

    Example:
        @timer("数据加载")
        def load_data():
            ...

        @timer()  # 使用函数名作为计时名称
        def process():
            ...

        # 多个装饰器组合
        @timer("翻译")
        def translate():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name if name is not None else func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                duration_str = format_duration(elapsed)
                console.print(f"[dim]⏱️ {timer_name}耗时: {duration_str}[/dim]")
        return wrapper
    return decorator


# ------------------------------
# check file exists decorator
# ------------------------------

def check_file_exists(file_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(file_path):
                rprint(f"[yellow]⚠️ File <{file_path}> already exists, skip <{func.__name__}> step.[/yellow]")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    @except_handler("function execution failed", retry=3, delay=1)
    def test_function():
        raise Exception("test exception")
    test_function()
