"""
状态持久化管理器
使用 JSON 文件保存文件和任务状态，支持重启后恢复
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
from threading import Lock

# 状态文件路径
STATE_FILE = "api/state.json"

# 线程锁（防止并发写入）
_lock = Lock()


def load_state() -> Dict[str, Any]:
    """
    从 JSON 文件加载状态

    Returns:
        Dict: {"files": {...}, "tasks": {...}, "metadata": {...}}
    """
    if not os.path.exists(STATE_FILE):
        return {
            "version": "1.0",
            "files": {},
            "tasks": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }

    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            return state
    except (json.JSONDecodeError, IOError) as e:
        print(f"加载状态文件失败: {e}", flush=True)
        return {
            "version": "1.0",
            "files": {},
            "tasks": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }


def save_state(state: Dict[str, Any]) -> bool:
    """
    保存状态到 JSON 文件

    Args:
        state: 完整的状态字典

    Returns:
        bool: 是否保存成功
    """
    try:
        with _lock:
            # 更新元数据
            state["metadata"] = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat()
            }

            # 确保目录存在
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

            # 写入文件（先写临时文件，再重命名，避免写入失败导致文件损坏）
            temp_file = STATE_FILE + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            # 重命名
            if os.path.exists(STATE_FILE):
                os.replace(temp_file, STATE_FILE)
            else:
                os.rename(temp_file, STATE_FILE)

            return True
    except Exception as e:
        print(f"保存状态文件失败: {e}", flush=True)
        return False


def update_files(files: Dict[str, Any]) -> bool:
    """
    更新文件记录

    Args:
        files: files_storage 字典

    Returns:
        bool: 是否保存成功
    """
    state = load_state()
    state["files"] = {fid: _serialize_file(f) for fid, f in files.items()}
    return save_state(state)


def update_tasks(tasks: Dict[str, Any]) -> bool:
    """
    更新任务记录

    Args:
        tasks: tasks_storage 字典

    Returns:
        bool: 是否保存成功
    """
    state = load_state()
    state["tasks"] = {tid: _serialize_task(t) for tid, t in tasks.items()}
    return save_state(state)


def add_file(file_id: str, file_info: Any) -> bool:
    """
    添加单个文件记录

    Args:
        file_id: 文件ID
        file_info: FileInfo 对象

    Returns:
        bool: 是否保存成功
    """
    state = load_state()
    state["files"][file_id] = _serialize_file(file_info)
    return save_state(state)


def remove_file(file_id: str, task_id: str = None) -> bool:
    """
    删除文件记录（及关联的任务记录）

    Args:
        file_id: 文件ID
        task_id: 可选，关联的任务ID

    Returns:
        bool: 是否保存成功
    """
    state = load_state()

    # 删除文件
    if file_id in state["files"]:
        del state["files"][file_id]

    # 删除所有关联该文件的任务
    tasks_to_delete = [
        tid for tid, tdata in state["tasks"].items()
        if tdata.get("file_id") == file_id
    ]
    for tid in tasks_to_delete:
        del state["tasks"][tid]

    return save_state(state)


def add_task(task_id: str, task_info: Any) -> bool:
    """
    添加单个任务记录

    Args:
        task_id: 任务ID
        task_info: TaskInfo 对象

    Returns:
        bool: 是否保存成功
    """
    state = load_state()
    state["tasks"][task_id] = _serialize_task(task_info)
    return save_state(state)


def update_task_status(task_id: str, **kwargs) -> bool:
    """
    更新任务状态

    Args:
        task_id: 任务ID
        **kwargs: 要更新的字段（status, progress, current_step, message, error 等）

    Returns:
        bool: 是否保存成功
    """
    state = load_state()

    if task_id not in state["tasks"]:
        return False

    # 更新字段
    for key, value in kwargs.items():
        if value is not None:
            state["tasks"][task_id][key] = _serialize_value(value)

    # 更新时间戳
    state["tasks"][task_id]["updated_at"] = datetime.now().isoformat()

    # 如果状态变为完成，添加完成时间
    if kwargs.get("status") == "completed":
        state["tasks"][task_id]["completed_at"] = datetime.now().isoformat()

    return save_state(state)


def update_file_active_task(file_id: str, task_id: str) -> bool:
    """
    更新文件的活跃任务ID

    Args:
        file_id: 文件ID
        task_id: 任务ID

    Returns:
        bool: 是否保存成功
    """
    state = load_state()

    if file_id not in state["files"]:
        return False

    state["files"][file_id]["active_task_id"] = task_id
    return save_state(state)


def clear_all_files() -> bool:
    """
    清空所有文件和任务记录

    Returns:
        bool: 是否保存成功
    """
    state = load_state()
    state["files"] = {}
    state["tasks"] = {}
    return save_state(state)


# ===== 辅助函数 =====

def _serialize_file(file_info: Any) -> Dict:
    """序列化 FileInfo 对象"""
    if hasattr(file_info, "dict"):
        data = file_info.dict()
        return {k: _serialize_value(v) for k, v in data.items()}
    elif hasattr(file_info, "__dict__"):
        return {k: _serialize_value(v) for k, v in file_info.__dict__.items() if not k.startswith("_")}
    else:
        return {"id": str(file_info)}


def _serialize_task(task_info: Any) -> Dict:
    """序列化 TaskInfo 对象"""
    if hasattr(task_info, "dict"):
        data = task_info.dict()
        return {k: _serialize_value(v) for k, v in data.items()}
    elif hasattr(task_info, "__dict__"):
        return {k: _serialize_value(v) for k, v in task_info.__dict__.items() if not k.startswith("_")}
    else:
        return {"id": str(task_info)}


def _serialize_value(value: Any) -> Any:
    """序列化值（处理 datetime、Enum 等）"""
    if isinstance(value, datetime):
        return value.isoformat()
    elif hasattr(value, "value"):  # 处理 Enum 类型
        return value.value
    elif hasattr(value, "dict"):
        data = value.dict()
        return {k: _serialize_value(v) for k, v in data.items()}
    elif hasattr(value, "__dict__"):
        return {k: _serialize_value(v) for k, v in value.__dict__.items() if not k.startswith("_")}
    else:
        return value
