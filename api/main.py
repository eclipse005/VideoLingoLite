"""
VideoLingoLite API Server
FastAPI backend for video/audio transcription and translation
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import sys
from contextlib import asynccontextmanager
import uvicorn
import os
import logging

# 配置日志，仅输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 屏蔽 httpx 的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)

from api.routes import files, tasks, config, terms, download, test
import json


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("🚀 VideoLingoLite API Server starting...")

    # 从 state.json 恢复状态
    from api import state_manager
    from api.routes import files, tasks as tasks_module
    from api.models.schemas import FileInfo, TaskInfo, TaskStatus, TaskType, FileType

    state = state_manager.load_state()
    files_count = len(state.get("files", {}))
    tasks_count = len(state.get("tasks", {}))

    if files_count > 0 or tasks_count > 0:
        print(f"📂 恢复状态: {files_count} 个文件, {tasks_count} 个任务")

        # 恢复文件记录
        for file_id, file_data in state.get("files", {}).items():
            try:
                # 动态生成 upload_path
                import os
                file_name = file_data.get("name", "")
                upload_path = os.path.join("api/uploads", f"{file_id}_{file_name}")

                # 检查 uploads 文件是否仍然存在
                if os.path.exists(upload_path):
                    # 处理 Enum 类型
                    if "type" in file_data and isinstance(file_data["type"], str):
                        file_data["type"] = FileType(file_data["type"])
                    # 重建 FileInfo 对象
                    file_info = FileInfo(**file_data)
                    files.files_storage[file_id] = file_info
                else:
                    print(f"⚠️  文件不存在，跳过恢复: {file_data.get('name', file_id)}")
            except Exception as e:
                print(f"❌ 恢复文件失败 {file_id}: {e}")

        # 恢复任务记录
        for task_id, task_data in state.get("tasks", {}).items():
            try:
                # 重建 TaskInfo 对象
                task_data["task_type"] = TaskType(task_data.get("task_type", "transcribe_and_translate"))

                # 处理中状态的任务改为 failed（程序重启后实际已停止）
                original_status = task_data.get("status", "pending")
                processing_statuses = ["asr", "hotword_correction", "meaning_split", "summarizing", "translating", "generating"]
                if original_status in processing_statuses:
                    task_data["status"] = "failed"
                    task_data["error"] = "程序重启，任务已停止"
                    task_data["message"] = "任务已停止，请重新点击按钮继续处理"
                    print(f"⚠️  任务 {task_id} 原状态为 {original_status}，已标记为失败")
                else:
                    task_data["status"] = TaskStatus(original_status)

                task_info = TaskInfo(**task_data)
                tasks_module.tasks_storage[task_id] = task_info
            except Exception as e:
                print(f"❌ 恢复任务失败 {task_id}: {e}")

        print(f"✅ 状态恢复完成: {len(files.files_storage)} 个文件, {len(tasks_module.tasks_storage)} 个任务")
    else:
        print("📂 无历史状态，从头开始")

    yield
    # 关闭时执行
    print("👋 VideoLingoLite API Server shutting down...")


# 创建 FastAPI 应用
app = FastAPI(
    title="VideoLingoLite API",
    description="Audio/video transcription and translation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理 Pydantic 验证错误"""
    logger.error(f"!!! Pydantic Validation Error !!!")
    logger.error(f"Request: {request.method} {request.url.path}")
    logger.error(f"Errors: {exc.errors()}")
    logger.error(f"Body: {exc.body}")
    sys.stdout.flush()
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理所有未捕获的异常"""
    logger.error(f"!!! Unhandled Exception !!!")
    logger.error(f"Request: {request.method} {request.url.path}")
    logger.error(f"Error: {type(exc).__name__}: {exc}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求和响应（排除轮询接口以减少日志噪音）"""
    # 排除轮询接口
    excluded_paths = ["/api/files", "/api/tasks", "/api/config", "/api/terms"]
    if request.url.path.startswith("/api") and request.url.path not in excluded_paths:
        logger.info(f">>> {request.method} {request.url.path}")
        sys.stdout.flush()

    # 让请求通过
    response = await call_next(request)

    if request.url.path.startswith("/api") and request.url.path not in excluded_paths:
        logger.info(f"<<< {request.method} {request.url.path} - Status: {response.status_code}")
        sys.stdout.flush()

    return response

# 注册路由
app.include_router(files.router, prefix="/api", tags=["files"])
app.include_router(tasks.router, prefix="/api", tags=["tasks"])
app.include_router(config.router, prefix="/api", tags=["config"])
app.include_router(terms.router, prefix="/api", tags=["terms"])
app.include_router(download.router, prefix="/api", tags=["download"])
app.include_router(test.router, prefix="/api", tags=["test"])


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "service": "VideoLingoLite API",
        "version": "1.0.0"
    }


# 静态文件服务（前端 Web UI）
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "webui")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="webui")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式自动重载
        log_level="info"
    )
