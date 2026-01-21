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
            rprint(f"[dim]🔍 函数 {func.__name__} 开始执行[/dim]")

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
            rprint(f"[dim]⚡ 执行函数 {func.__name__}...[/dim]")
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
    try:
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

        rprint(f"[green]💾 已保存文本文件: {text_file}[/green]")
    except Exception as e:
        rprint(f"[red]❌ 保存文本文件失败 {text_file}: {e}[/red]")
        raise

if __name__ == "__main__":
    @except_handler("function execution failed", retry=3, delay=1)
    def test_function():
        raise Exception("test exception")
    test_function()
