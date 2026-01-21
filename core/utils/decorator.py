import functools
import os
import pickle
from typing import Any
from rich import print as rprint

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


# ------------------------------
# cache objects decorator
# ------------------------------

def cache_objects(cache_file: str):
    """
    对象缓存装饰器：从 pickle 文件加载/保存 Python 对象

    如果缓存存在：直接加载并返回
    如果缓存不存在：执行函数，保存结果到 pickle

    Args:
        cache_file: 缓存文件路径（.pkl 文件）

    Example:
        @cache_objects("output/cache/sentences_nlp.pkl")
        def split_by_nlp(nlp):
            # ... 处理逻辑
            return sentences
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_file):
                rprint(f"[yellow]⏩ 从缓存加载: {cache_file}[/yellow]")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

            # 执行函数
            result = func(*args, **kwargs)

            # 确保输出目录存在
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            # 保存到缓存
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            rprint(f"[green]💾 已缓存: {cache_file}[/green]")

            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    @except_handler("function execution failed", retry=3, delay=1)
    def test_function():
        raise Exception("test exception")
    test_function()
