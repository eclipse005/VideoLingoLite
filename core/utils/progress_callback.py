"""
进度回调管理器
用于在处理模块中报告细粒度进度，支持通过 WebSocket 推送到前端
"""

from typing import Optional, Callable
import threading
import asyncio

# 全局进度回调函数
_progress_callback: Optional[Callable[[int, int, str], None]] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_callback_lock = threading.Lock()


def set_progress_callback(
    callback: Optional[Callable[[int, int, str], None]],
    loop: Optional[asyncio.AbstractEventLoop] = None
) -> None:
    """
    设置进度回调函数

    Args:
        callback: 回调函数，接收 (current, total, message) 参数
                 - current: 当前进度
                 - total: 总数
                 - message: 进度描述信息
        loop: 异步事件循环（用于线程安全地提交异步任务）
    """
    global _progress_callback, _event_loop
    with _callback_lock:
        _progress_callback = callback
        _event_loop = loop


def report_progress(current: int, total: int, message: str = "") -> None:
    """
    报告进度（由各处理模块调用）

    Args:
        current: 当前进度
        total: 总数
        message: 进度描述信息
    """
    global _progress_callback, _event_loop
    with _callback_lock:
        if _progress_callback:
            try:
                _progress_callback(current, total, message)
            except Exception as e:
                # 静默失败，避免影响主流程
                print(f"进度回调失败: {e}")


def get_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """获取当前事件循环"""
    global _event_loop
    return _event_loop


def clear_progress_callback() -> None:
    """清除进度回调函数"""
    global _progress_callback, _event_loop
    with _callback_lock:
        _progress_callback = None
        _event_loop = None


class ProgressReporter:
    """
    上下文管理器，用于在特定作用域内设置进度回调

    Example:
        with ProgressReporter(callback_func, loop):
            # 在这个作用域内，所有 report_progress 调用都会触发 callback_func
            process_items()
        # 退出作用域后自动清除回调
    """

    def __init__(
        self,
        callback: Optional[Callable[[int, int, str], None]],
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        self.callback = callback
        self.loop = loop
        self.old_callback = None
        self.old_loop = None

    def __enter__(self):
        global _progress_callback, _event_loop
        with _callback_lock:
            self.old_callback = _progress_callback
            self.old_loop = _event_loop
            _progress_callback = self.callback
            _event_loop = self.loop
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _progress_callback, _event_loop
        with _callback_lock:
            _progress_callback = self.old_callback
            _event_loop = self.old_loop
        return False
