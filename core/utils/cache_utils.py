"""
缓存装饰器工具

提供基于参数序列化的缓存装饰器，用于缓存耗时的 LLM 调用
"""

import pickle
import hashlib
import os
import glob
from functools import wraps


def cache():
    """
    缓存装饰器：基于参数序列化自动计算 hash
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from core.utils import rprint
            from core.utils.config_utils import load_key

            cache_dir = "output/log"

            # 检查术语矫正开关（仅对 correct_terms_in_sentences 函数）
            if func.__name__ == "correct_terms_in_sentences":
                if not load_key("asr_term_correction.enabled"):
                    # 开关关闭，不缓存，直接执行函数
                    return func(*args, **kwargs)

            # 将参数转换为可序列化的格式
            serializable_args = _make_serializable(args)
            serializable_kwargs = _make_serializable(kwargs)

            # 计算缓存 key
            cache_key = _compute_hash(serializable_args, serializable_kwargs)
            cache_path = os.path.join(cache_dir, f"cache_{cache_key}.pkl")

            # 检查缓存
            if os.path.exists(cache_path):
                rprint(f"[dim][Cache] 加载缓存: {os.path.basename(cache_path)}[/dim]")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

            # 执行函数
            result = func(*args, **kwargs)

            # 保存缓存（并删除同函数的旧缓存）
            os.makedirs(cache_dir, exist_ok=True)

            # 删除同函数的所有旧缓存
            pattern = os.path.join(cache_dir, f"cache_*.pkl")
            for old_cache in glob.glob(pattern):
                if old_cache != cache_path:
                    os.remove(old_cache)

            # 保存新缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            rprint(f"[dim][Cache] 保存缓存: {os.path.basename(cache_path)}[/dim]")

            return result

        return wrapper
    return decorator


def _make_serializable(obj):
    """
    将对象转换为可序列化的格式

    对于 Sentence 对象，只提取关键属性（text）用于 hash 计算
    """
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Sentence':
        return {'text': obj.text, 'class': 'Sentence'}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def _compute_hash(args, kwargs):
    """计算参数组合的 hash 值"""
    # 序列化参数
    data = pickle.dumps({'args': args, 'kwargs': kwargs})

    # 计算 hash
    return hashlib.md5(data).hexdigest()[:16]
