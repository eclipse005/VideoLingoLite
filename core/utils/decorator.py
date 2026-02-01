import functools
import os
import pickle
import time
from typing import Any, Optional
from rich import print as rprint
from rich.console import Console

console = Console()

# ------------------------------
# retry decorator
# ------------------------------

def except_handler(error_msg, retry=0, delay=1, default_return=None, verbose=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retry + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if verbose:
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
        @cache_objects(cache_file, text_file)
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


# ------------------------------
# cache objects decorator
# ------------------------------

def cache_objects(cache_file: str, text_file: str = None, text_attr: str = 'text'):
    """
    对象缓存装饰器：从 pickle 文件加载/保存 Python 对象

    如果缓存存在：直接加载并返回
    如果缓存不存在：执行函数，保存结果到 pickle

    Args:
        cache_file: 缓存文件路径（.pkl 文件）
        text_file: 可选，同时保存的文本文件路径
        text_attr: 从对象中提取文本的属性名（默认 'text'）

    Example:
        @cache_objects("output/cache/sentences_nlp.pkl")
        def split_by_nlp(nlp):
            # ... 处理逻辑
            return sentences

        @cache_objects("output/cache/sentences_split.pkl", "output/log/split_by_meaning.txt")
        def split_sentences_by_meaning(sentences):
            # ... 处理逻辑
            return sentences
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(cache_file):
                rprint(f"[yellow]⏩ 从缓存加载: {cache_file}[/yellow]")
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)

                    # 从缓存加载后，也要保存文本文件（如果指定且不存在）
                    if text_file and not os.path.exists(text_file):
                        _save_text_file(result, text_file, text_attr)

                    return result
                except Exception as e:
                    rprint(f"[red]❌ 加载缓存失败: {e}[/red]")
                    raise

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

            # 保存文本文件（如果指定）
            if text_file:
                _save_text_file(result, text_file, text_attr)

            return result
        return wrapper
    return decorator


def _save_text_file(obj, text_file: str, text_attr: str = 'text'):
    """从对象列表中提取文本并保存到文件"""
    # 确保目录存在
    text_dir = os.path.dirname(text_file)
    if text_dir:
        os.makedirs(text_dir, exist_ok=True)

    # 提取文本
    if isinstance(obj, list) and obj and hasattr(obj[0], text_attr):
        texts = [getattr(item, text_attr) for item in obj]
        content = '\n'.join(texts)
    else:
        content = str(obj)

    # 保存文件
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    @except_handler("function execution failed", retry=3, delay=1)
    def test_function():
        raise Exception("test exception")
    test_function()
