"""
任务管理接口
处理任务的创建、启动、取消、状态查询等操作
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import uuid
from datetime import datetime
import asyncio
import os
import concurrent.futures

from api.models.schemas import (
    TaskInfo,
    TaskStatus,
    CreateTaskRequest,
    CreateTaskResponse,
    TaskType,
    FileInfo
)
from core.utils.progress_callback import set_progress_callback, clear_progress_callback
from api import state_manager

router = APIRouter()

# 线程池（用于运行阻塞的后台任务）
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# 内存存储（生产环境应该使用数据库）
# 结构: {task_id: TaskInfo}
tasks_storage: dict = {}

# 任务取消标志（用于后台任务检查）
task_cancel_flags: dict = {}

# 当前运行的任务ID（用于确保串行执行）
current_running_task_id: Optional[str] = None

# 阶段进度范围定义（按时间权重分配）
STAGE_PROGRESS_RANGE = {
    TaskStatus.ASR: (0, 20),              # 0% → 20% (20%)
    TaskStatus.HOTWORD_CORRECTION: (20, 28),  # 20% → 28% (8%)
    TaskStatus.MEANING_SPLIT: (28, 36),    # 28% → 36% (8%)
    TaskStatus.SUMMARIZING: (36, 42),      # 36% → 42% (6%)
    TaskStatus.TRANSLATING: (42, 76),      # 42% → 76% (34%)
    TaskStatus.GENERATING: (76, 94),       # 76% → 94% (18%) - 字幕拆分对齐
    TaskStatus.COMPLETED: (100, 100),      # 100% - 任务完成
}

# 仅转录流程的进度范围
TRANSCRIBE_ONLY_RANGE = {
    TaskStatus.ASR: (0, 40),               # 0% → 40% (40%)
    TaskStatus.HOTWORD_CORRECTION: (40, 50),  # 40% → 50% (10%)
    TaskStatus.MEANING_SPLIT: (50, 60),    # 50% → 60% (10%)
    TaskStatus.GENERATING: (60, 97),       # 60% → 97% (37%)
    TaskStatus.COMPLETED: (100, 100),      # 100%
}


def update_task_status(task_id: str, status: TaskStatus = None, progress: int = None,
                       current_step: str = None, message: str = None):
    """
    更新任务状态到内存（方便前端轮询获取）

    注意：只有关键状态变更（status）才会持久化到 JSON
    进度更新（progress, current_step, message）仅在内存中
    """
    if task_id in tasks_storage:
        should_save = False  # 是否需要持久化到 JSON

        if status is not None:
            tasks_storage[task_id].status = status
            should_save = True  # 状态变更需要持久化
        if progress is not None:
            tasks_storage[task_id].progress = progress
        if current_step is not None:
            tasks_storage[task_id].current_step = current_step
        if message is not None:
            tasks_storage[task_id].message = message
        tasks_storage[task_id].updated_at = datetime.now()

        # 关键状态变更时持久化到 JSON
        if should_save:
            state_manager.update_task_status(
                task_id,
                status=status.value if status else None,
                progress=progress,
                current_step=current_step,
                message=message,
                updated_at=datetime.now().isoformat()
            )


def start_next_queued_task():
    """查找并启动下一个排队的任务（同步函数）"""
    global current_running_task_id

    # 如果已有任务在运行，不做处理
    if current_running_task_id:
        return

    # 查找最早创建的排队任务
    queued_tasks = [
        (task_id, task)
        for task_id, task in tasks_storage.items()
        if task.status == TaskStatus.QUEUED
    ]

    if not queued_tasks:
        return

    # 按创建时间排序，取最早的
    queued_tasks.sort(key=lambda x: x[1].created_at)
    next_task_id, next_task = queued_tasks[0]

    print(f"\n>>> 自动启动排队任务: {next_task_id} ({next_task.file_name}) <<<", flush=True)

    # 获取文件信息
    from api.routes.files import files_storage, UPLOAD_DIR, OUTPUT_DIR
    import shutil
    import glob

    file_info = files_storage[next_task.file_id]
    upload_file_path = os.path.join(UPLOAD_DIR, f"{next_task.file_id}_{file_info.name}")
    output_file_path = os.path.join(OUTPUT_DIR, file_info.name)

    # 清理 output 目录
    for ext in ['*.mp4', '*.webm', '*.mkv', '*.avi', '*.mov', '*.mp3', '*.wav', '*.m4a', '*.ogg', '*.flac']:
        for old_file in glob.glob(os.path.join(OUTPUT_DIR, ext)):
            if os.path.exists(old_file):
                os.remove(old_file)

    # 确保 output 目录存在（归档后可能被删除）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 复制文件
    shutil.copy2(upload_file_path, output_file_path)

    # 标记为运行中
    current_running_task_id = next_task_id
    update_task_status(next_task_id, TaskStatus.ASR, 0, "准备中", "任务已启动")

    # 在线程池中运行（尝试获取当前事件循环，如果失败则创建新的）
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_in_executor(executor, process_task_task_sync, next_task_id, next_task.file_id, next_task.file_name, next_task.task_type)


def process_task_task_sync(task_id: str, file_id: str, file_name: str, task_type: TaskType):
    """
    同步包装函数，用于在线程池中运行
    """
    return asyncio.run(process_task_task(task_id, file_id, file_name, task_type))


async def process_task_task(task_id: str, file_id: str, file_name: str, task_type: TaskType):
    """
    后台任务处理函数
    处理音视频转写翻译的完整流程
    """
    print(f"\n>>> 任务开始执行: task_id={task_id}, file_id={file_id}, task_type={task_type} <<<", flush=True)

    # 定义细粒度进度回调函数（使用闭包捕获 task_id 和 task_type）
    def on_detailed_progress(current: int, total: int, message: str):
        """处理细粒度进度更新，计算渐进式进度百分比"""
        try:
            # 获取当前任务状态
            current_status = tasks_storage.get(task_id).status if task_id in tasks_storage else None
            if not current_status:
                return

            # 根据任务类型选择进度范围
            progress_range = STAGE_PROGRESS_RANGE.get(current_status) if task_type == TaskType.TRANSCRIBE_AND_TRANSLATE else TRANSCRIBE_ONLY_RANGE.get(current_status)

            if progress_range:
                start, end = progress_range
                # 计算渐进式进度：start + (current/total) * (end - start)
                calculated_progress = int(start + (current / total) * (end - start))
                update_task_status(task_id, progress=calculated_progress, message=message)
            else:
                # 没有定义范围的阶段，只更新 message
                update_task_status(task_id, message=message)
        except Exception as e:
            print(f"更新细粒度进度失败: {e}", flush=True)

    # 设置进度回调
    set_progress_callback(on_detailed_progress)

    try:
        # 导入VideoLingoLite核心模块
        from core import _2_asr, _3_1_split_nlp, _3_2_hotword, _3_3_split_meaning
        from core import _4_1_summarize, _4_2_translate, _5_split_sub, _6_gen_sub
        from core.utils.onekeycleanup import cleanup

        # 语音识别转录
        spinner_text = "正在进行人声分离与语音转录..." if os.getenv("vocal_separation.enabled", "").lower() == "true" else "正在进行语音转录..."
        update_task_status(task_id, TaskStatus.ASR, 0, "ASR 转录中", spinner_text)

        if task_cancel_flags.get(task_id):
            update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
            return

        _2_asr.transcribe()

        # NLP 分句（无细粒度进度，直接设置为 ASR 完成后的进度）
        update_task_status(task_id, TaskStatus.NLP_SPLIT, 20, "NLP 分句中", "正在进行NLP分句...")

        if task_cancel_flags.get(task_id):
            update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
            return

        sentences = _3_1_split_nlp.split_by_spacy()

        # 根据任务类型决定后续流程
        if task_type == TaskType.TRANSCRIBE_ONLY:
            # 仅转录流程（包含热词矫正和长句切分）
            update_task_status(task_id, TaskStatus.HOTWORD_CORRECTION, 20, "热词矫正中", "正在进行热词矫正...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            sentences = _3_2_hotword.correct_terms_in_sentences(sentences)

            update_task_status(task_id, TaskStatus.MEANING_SPLIT, 28, "语义分句中", "正在使用LLM切分长句...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            sentences = _3_3_split_meaning.split_sentences_by_meaning(sentences)

            update_task_status(task_id, TaskStatus.GENERATING, 60, "生成字幕中", "正在生成原文字幕...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            _6_gen_sub.align_timestamp_main(sentences, transcript_only=True)

            # 字幕生成完成，归档前
            update_task_status(task_id, TaskStatus.GENERATING, 97, "准备归档", "正在准备归档...")
        else:  # TRANSCRIBE_AND_TRANSLATE
            # 完整流程（转录+翻译）
            update_task_status(task_id, TaskStatus.HOTWORD_CORRECTION, 20, "热词矫正中", "正在进行热词矫正...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            sentences = _3_2_hotword.correct_terms_in_sentences(sentences)

            update_task_status(task_id, TaskStatus.MEANING_SPLIT, 28, "语义分句中", "正在使用LLM切分长句...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            sentences = _3_3_split_meaning.split_sentences_by_meaning(sentences)

            update_task_status(task_id, TaskStatus.SUMMARIZING, 36, "摘要中", "正在进行总结...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            _4_1_summarize.get_summary(sentences)

            # 翻译阶段
            update_task_status(task_id, TaskStatus.TRANSLATING, 42, "翻译中", "正在进行翻译...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            sentences = _4_2_translate.translate_all(sentences)

            update_task_status(task_id, TaskStatus.GENERATING, 76, "处理对齐中", "正在处理和对齐字幕...")

            if task_cancel_flags.get(task_id):
                update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
                return

            sentences = _5_split_sub.split_for_sub_main(sentences)
            _6_gen_sub.align_timestamp_main(sentences, transcript_only=False)

            # 字幕生成完成，归档前
            update_task_status(task_id, TaskStatus.GENERATING, 94, "准备归档", "正在准备归档...")

        # 归档
        update_task_status(task_id, TaskStatus.GENERATING, 97, "归档中", "正在归档处理结果...")

        if task_cancel_flags.get(task_id):
            update_task_status(task_id, TaskStatus.CANCELLED, message="任务已取消")
            return

        cleanup(task_id=task_id)

        # 任务完成
        update_task_status(task_id, TaskStatus.COMPLETED, 100, "已完成", "任务处理完成")

    except Exception as e:
        # 任务失败
        import traceback
        print(f"\n>>> 任务执行失败: task_id={task_id} <<<", flush=True)
        print(f"错误类型: {type(e).__name__}", flush=True)
        print(f"错误信息: {str(e)}", flush=True)
        print(f"堆栈跟踪:\n{traceback.format_exc()}", flush=True)

        if task_id in tasks_storage:
            tasks_storage[task_id].status = TaskStatus.FAILED
            tasks_storage[task_id].error = str(e)
            tasks_storage[task_id].updated_at = datetime.now()
    finally:
        # 清理取消标志和进度回调
        if task_id in task_cancel_flags:
            del task_cancel_flags[task_id]
        clear_progress_callback()

        # 清理运行中任务标记
        global current_running_task_id
        if current_running_task_id == task_id:
            current_running_task_id = None

        # 自动启动下一个排队的任务（直接调用同步函数）
        start_next_queued_task()


@router.post("/tasks", response_model=CreateTaskResponse)
async def create_tasks(request: CreateTaskRequest):
    """
    创建处理任务

    为指定的文件创建处理任务，并更新文件的 active_task_id
    """
    # 导入必须在这里，确保获取最新的 files_storage
    import api.routes.files as files_module
    from api.routes.files import FileInfo

    files_storage = files_module.files_storage

    created_tasks = []

    for file_id in request.file_ids:
        # 检查文件是否存在
        if file_id not in files_storage:
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_id}")

        file_info = files_storage[file_id]
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # 创建任务
        task = TaskInfo(
            id=task_id,
            file_id=file_id,
            file_name=file_info.name,
            task_type=request.task_type,
            status=TaskStatus.PENDING,
            progress=0,
            current_step="等待开始",
            message="任务已创建，等待启动"
        )

        tasks_storage[task_id] = task
        created_tasks.append(task)

        # 更新文件的活跃任务ID - 直接修改属性
        file_info.active_task_id = task_id

        # 持久化到 JSON（添加任务，更新文件的 active_task_id）
        state_manager.add_task(task_id, task)
        state_manager.update_file_active_task(file_id, task_id)

    return CreateTaskResponse(tasks=created_tasks, count=len(created_tasks))


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: str):
    """
    开始执行任务

    启动后台任务处理流程
    如果已有任务正在运行，新任务将排队等待
    """
    global current_running_task_id

    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks_storage[task_id]

    if task.status not in [TaskStatus.PENDING, TaskStatus.QUEUED]:
        raise HTTPException(status_code=400, detail=f"任务状态不允许启动: {task.status}")

    # 检查是否有任务正在运行
    if current_running_task_id and current_running_task_id != task_id:
        # 有任务正在运行，设置为排队状态
        update_task_status(task_id, TaskStatus.QUEUED, message="排队中，等待当前任务完成")
        return {
            "success": True,
            "message": "任务已加入队列",
            "task_id": task_id,
            "queued": True
        }

    # 从 api/uploads 复制文件到 output 目录
    from api.routes.files import files_storage, UPLOAD_DIR, OUTPUT_DIR
    import shutil

    file_info = files_storage[task.file_id]
    upload_file_path = os.path.join(UPLOAD_DIR, f"{task.file_id}_{file_info.name}")
    output_file_path = os.path.join(OUTPUT_DIR, file_info.name)

    # 清理 output 目录中可能存在的旧文件
    import glob
    for ext in ['*.mp4', '*.webm', '*.mkv', '*.avi', '*.mov', '*.mp3', '*.wav', '*.m4a', '*.ogg', '*.flac']:
        for old_file in glob.glob(os.path.join(OUTPUT_DIR, ext)):
            if os.path.exists(old_file):
                os.remove(old_file)

    # 确保 output 目录存在（归档后可能被删除）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 复制文件到 output 目录
    shutil.copy2(upload_file_path, output_file_path)
    print(f"文件已复制: {upload_file_path} -> {output_file_path}", flush=True)

    # 更新任务状态
    update_task_status(task_id, TaskStatus.ASR, 0, "准备中", "任务已启动")

    # 标记任务为运行中
    current_running_task_id = task_id

    # 在线程池中运行阻塞的任务处理函数（避免阻塞事件循环）
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, process_task_task_sync, task.id, task.file_id, task.file_name, task.task_type)

    return {"success": True, "message": "任务已启动", "task_id": task_id}


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """
    取消正在执行的任务

    设置取消标志，后台任务会检查并退出
    """
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks_storage[task_id]

    # 只能取消正在运行的任务
    if task.status in [TaskStatus.PENDING, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail=f"任务状态不允许取消: {task.status}")

    # 设置取消标志
    task_cancel_flags[task_id] = True

    return {"success": True, "message": "任务取消请求已发送"}


@router.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """
    获取任务状态

    返回指定任务的详细状态信息
    """
    if task_id not in tasks_storage:
        raise HTTPException(status_code=404, detail="任务不存在")

    return tasks_storage[task_id]


@router.get("/tasks", response_model=List[TaskInfo])
async def get_tasks(file_id: Optional[str] = Query(None, description="按文件ID筛选任务")):
    """
    获取任务列表

    返回所有任务的列表，或按 file_id 筛选特定文件的任务
    """
    tasks = list(tasks_storage.values())
    if file_id:
        tasks = [t for t in tasks if t.file_id == file_id]
    return tasks
