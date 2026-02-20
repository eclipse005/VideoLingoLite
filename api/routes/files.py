"""
文件管理接口
处理文件上传、列表、删除等操作
"""

from fastapi import APIRouter, UploadFile, HTTPException, File, BackgroundTasks
from typing import List
import os
import shutil
import uuid
import asyncio
from datetime import datetime

from api.models.schemas import FileInfo, FileType, FileSource, UploadResponse, TaskInfo
from api import state_manager

router = APIRouter()

# 文件存储目录（保存带 file_id 前缀的文件，支持同名文件）
UPLOAD_DIR = "api/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 输出目录（核心处理模块从这里查找文件）
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 导出常量供其他模块使用
__all__ = ['router', 'files_storage', 'UPLOAD_DIR', 'OUTPUT_DIR']


def get_file_type(filename: str) -> FileType:
    """根据文件扩展名判断文件类型"""
    ext = filename.lower().split('.')[-1]
    audio_exts = {'mp3', 'wav', 'm4a', 'ogg', 'flac'}
    video_exts = {'mp4', 'webm', 'mkv', 'avi', 'mov'}

    if ext in audio_exts:
        return FileType.AUDIO
    elif ext in video_exts:
        return FileType.VIDEO
    else:
        return FileType.UNKNOWN


# 内存存储（生产环境应该使用数据库）
# 结构: {file_id: FileInfo}
files_storage: dict = {}


@router.post("/files/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    上传音视频文件

    支持批量上传，返回上传的文件信息列表
    """
    uploaded_files = []

    for file in files:
        # 生成唯一文件ID
        file_id = f"file_{uuid.uuid4().hex[:8]}"

        # 保存到 uploads（带 file_id 前缀，支持同名文件）
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 创建文件信息
            file_info = FileInfo(
                id=file_id,
                name=file.filename,
                size=file_size,
                type=get_file_type(file.filename)
            )

            # 存储到内存
            files_storage[file_id] = file_info
            uploaded_files.append(file_info)

            # 持久化到 JSON
            state_manager.add_file(file_id, file_info)

        except Exception as e:
            # 清理已上传的文件
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

    return UploadResponse(files=uploaded_files, count=len(uploaded_files))


@router.get("/files", response_model=List[FileInfo])
async def get_files():
    """
    获取已上传的文件列表

    返回所有已上传文件的信息，包含 active_task_id 指向当前活跃任务
    """
    return list(files_storage.values())


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    删除指定文件

    根据文件ID删除文件及存储的文件
    如果有关联的任务，也会一并删除（仅待处理和排队状态的任务）
    """
    if file_id not in files_storage:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_info = files_storage[file_id]

    # 检查并删除关联的任务
    from api.routes import tasks as tasks_module
    from api.models.schemas import TaskStatus

    if file_info.active_task_id:
        task_id = file_info.active_task_id
        if task_id in tasks_module.tasks_storage:
            task = tasks_module.tasks_storage[task_id]

            # 如果任务正在运行，不允许删除
            if task.status in [TaskStatus.ASR, TaskStatus.HOTWORD_CORRECTION,
                              TaskStatus.MEANING_SPLIT, TaskStatus.SUMMARIZING, TaskStatus.GENERATING]:
                raise HTTPException(status_code=400, detail="任务正在运行中，无法删除文件")

            # 删除任务（PENDING, QUEUED, COMPLETED, FAILED, CANCELLED 可以删除）
            del tasks_module.tasks_storage[task_id]

            # 如果是排队中的任务，从取消标志中清理
            if task_id in tasks_module.task_cancel_flags:
                del tasks_module.task_cancel_flags[task_id]

    # 删除 api/uploads 中的文件
    file_pattern = f"{file_id}_*"
    upload_files = os.listdir(UPLOAD_DIR)
    for filename in upload_files:
        if filename.startswith(file_id):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    # 删除 output 目录中的文件（如果已经复制过去处理）
    output_path = os.path.join(OUTPUT_DIR, file_info.name)
    if os.path.exists(output_path):
        os.remove(output_path)

    # 从存储中删除
    del files_storage[file_id]

    # 持久化到 JSON（删除文件和关联任务）
    state_manager.remove_file(file_id, file_info.active_task_id)

    return {"success": True, "message": "文件已删除"}


@router.delete("/files")
async def clear_all_files():
    """
    清空所有文件

    删除所有已上传的文件
    """
    # 删除 api/uploads 中的所有文件
    upload_files = os.listdir(UPLOAD_DIR)
    for filename in upload_files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    # 删除 output 目录中的所有媒体文件
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # 清空存储
    files_storage.clear()

    # 持久化到 JSON（清空所有记录）
    state_manager.clear_all_files()

    return {"success": True, "message": "所有文件已清空"}


# ============ YouTube 下载接口 ============

@router.post("/files/youtube/parse")
async def parse_youtube_video(request: dict):
    """
    解析 YouTube 视频，获取可用的格式和质量选项

    Request body:
    {
        "url": "https://www.youtube.com/watch?v=xxx"
    }

    Response:
    {
        "title": "视频标题",
        "thumbnail": "缩略图URL",
        "duration": 600,
        "uploader": "上传者",
        "formats": [
            {
                "format_id": "137+140",
                "quality": "2K",
                "ext": "mp4",
                "height": 1440,
                "width": 2560,
                "filesize": 450000000,
                "format_selector": "bestvideo[ext=mp4][height<=1440]+bestaudio/best[ext=mp4]",
                "label": "1440p 2K (MP4) - 429 MB",
                "is_recommended": false
            },
            ...
        ]
    }
    """
    url = request.get('url', '')

    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="无效的 URL")

    url = url.strip()

    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="URL 必须以 http:// 或 https:// 开头")

    # 验证是否是支持的域名
    supported_domains = ['youtube.com', 'youtu.be', 'm.youtube.com', 'www.youtube.com']
    if not any(domain in url for domain in supported_domains):
        raise HTTPException(status_code=400, detail="不支持的网站，仅支持 YouTube")

    try:
        import core._1_ytdlp as ytdlp_module
        result = ytdlp_module.parse_video_formats(url)
        return result
    except ImportError:
        raise HTTPException(status_code=500, detail="yt-dlp 未安装")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解析失败: {str(e)}")


@router.post("/files/youtube")
async def download_youtube_video(request: dict, background_tasks: BackgroundTasks):
    """
    下载 YouTube 视频

    Request body:
    {
        "url": "https://www.youtube.com/watch?v=xxx",
        "format_selector": "bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best"  // 可选
    }

    Response:
    {
        "file_id": "file_xxxxxx",
        "status": "downloading",
        "message": "开始下载..."
    }
    """
    url = request.get('url', '')
    format_selector = request.get('format_selector')

    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="无效的 URL")

    url = url.strip()

    if not url.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="URL 必须以 http:// 或 https:// 开头")

    # 验证是否是支持的域名
    supported_domains = ['youtube.com', 'youtu.be', 'm.youtube.com', 'www.youtube.com']
    if not any(domain in url for domain in supported_domains):
        raise HTTPException(status_code=400, detail="不支持的网站，仅支持 YouTube")

    # 生成唯一文件ID
    file_id = f"file_{uuid.uuid4().hex[:8]}"

    # 后台下载任务
    def download_task():
        try:
            import core._1_ytdlp as ytdlp_module

            result = ytdlp_module.download_video_ytdlp(
                url=url,
                output_dir=UPLOAD_DIR,
                file_id=file_id,
                format_selector=format_selector  # 传递格式选择器
            )

            # 获取文件信息
            filepath = result['filepath']
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"下载失败：文件不存在 {filepath}")

            filesize = os.path.getsize(filepath)
            filename = os.path.basename(filepath)

            # 去除 file_id_ 前缀，只显示视频标题（和本地上传保持一致）
            display_name = filename.replace(f'{file_id}_', '', 1) if filename.startswith(f'{file_id}_') else filename

            # 创建 FileInfo（和本地上传相同的结构）
            file_info = FileInfo(
                id=file_id,
                name=display_name,
                size=filesize,
                type=FileType.VIDEO,
                source=FileSource.YOUTUBE  # 标记来源
            )

            # 存储到内存
            files_storage[file_id] = file_info
            state_manager.add_file(file_id, file_info)

            # 更新进度为完成
            ytdlp_module._download_progress[file_id] = {
                'status': 'completed',
                'progress': 100,
                'message': '下载完成',
                'filepath': filepath
            }

        except Exception as e:
            import core._1_ytdlp as ytdlp_module
            ytdlp_module._download_progress[file_id] = {
                'status': 'error',
                'progress': 0,
                'message': f'下载失败: {str(e)}'
            }

    # 添加后台任务
    background_tasks.add_task(download_task)

    # 立即返回 file_id（前端开始轮询进度）
    return {
        "file_id": file_id,
        "status": "downloading",
        "message": "开始下载..."
    }


@router.get("/files/youtube/progress/{file_id}")
async def get_youtube_download_progress(file_id: str):
    """
    获取 YouTube 下载进度

    Response:
    {
        "status": "downloading" | "post-processing" | "completed" | "error",
        "progress": 45.5,
        "message": "下载中... 45.5%",
        "speed": 1048576,  # 字节/秒
        "eta": 120  # 剩余秒数
    }
    """
    import core._1_ytdlp as ytdlp_module
    progress = ytdlp_module.get_download_progress(file_id)

    # 如果下载完成且文件已添加到 storage，清理进度记录
    if progress.get('status') == 'completed' and file_id in files_storage:
        # 但保留一段时间以便前端获取最终状态
        pass

    return progress
