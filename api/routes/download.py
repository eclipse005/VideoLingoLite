"""
下载接口
处理生成的字幕文件下载
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Literal
import os

router = APIRouter()


def sanitize_filename(filename: str) -> str:
    """清理文件名中的非法字符"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


@router.get("/download/{task_id}/{file_type}")
async def download_subtitle(task_id: str, file_type: Literal["src", "trans", "src_trans", "trans_src"]):
    """
    下载字幕文件

    支持下载不同格式的字幕文件：
    - src: 原文字幕 (src.srt)
    - trans: 翻译字幕 (trans.srt)
    - src_trans: 双语字幕-原文在前 (src_trans.srt)
    - trans_src: 双语字幕-译文在前 (trans_src.srt)
    """
    # 文件类型映射
    file_map = {
        "src": "src.srt",
        "trans": "trans.srt",
        "src_trans": "src_trans.srt",
        "trans_src": "trans_src.srt"
    }

    if file_type not in file_map:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_type}")

    # 导入任务存储（避免循环导入）
    from api.routes import tasks as tasks_module

    # 查找任务信息
    if task_id not in tasks_module.tasks_storage:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks_module.tasks_storage[task_id]

    # 检查任务是否完成
    if task.status.value != "completed":
        raise HTTPException(status_code=400, detail="任务未完成，无法下载字幕")

    # 获取文件名（去掉扩展名）
    file_name = os.path.splitext(task.file_name)[0]
    sanitized_name = sanitize_filename(file_name)

    # 构建归档目录路径: history/{文件名}_{task_id}
    history_dir = os.path.join("history", f"{sanitized_name}_{task_id}")
    file_path = os.path.join(history_dir, file_map[file_type])

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"字幕文件不存在，可能已被删除")

    # 返回文件
    filename = file_map[file_type]
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/plain"
    )
