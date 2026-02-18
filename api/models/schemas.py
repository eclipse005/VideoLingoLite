"""
API 数据模型定义
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
from enum import Enum
from datetime import datetime


# ============ 任务类型枚举 ============
class TaskType(str, Enum):
    """任务类型"""
    TRANSCRIBE_AND_TRANSLATE = "transcribe_and_translate"  # 转录和翻译（完整流程）
    TRANSCRIBE_ONLY = "transcribe_only"  # 仅转录


# ============ 任务状态枚举 ============
class TaskStatus(str, Enum):
    """任务处理状态"""
    PENDING = "pending"
    QUEUED = "queued"
    ASR = "asr"
    HOTWORD_CORRECTION = "hotword_correction"
    MEANING_SPLIT = "meaning_split"
    SUMMARIZING = "summarizing"
    TRANSLATING = "translating"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============ 文件类型 ============
class FileType(str, Enum):
    """文件类型"""
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


# ============ 文件信息 ============
class FileInfo(BaseModel):
    """文件信息"""
    id: str = Field(..., description="文件唯一ID")
    name: str = Field(..., description="文件名")
    size: int = Field(..., description="文件大小（字节）")
    type: FileType = Field(..., description="文件类型")
    uploaded_at: datetime = Field(default_factory=datetime.now, description="上传时间")
    active_task_id: Optional[str] = Field(None, description="当前活跃任务的ID")


# ============ 任务信息 ============
class TaskInfo(BaseModel):
    """任务信息"""
    id: str = Field(..., description="任务唯一ID")
    file_id: str = Field(..., description="关联的文件ID")
    file_name: str = Field(..., description="文件名")
    task_type: TaskType = Field(default=TaskType.TRANSCRIBE_AND_TRANSLATE, description="任务类型")
    status: TaskStatus = Field(..., description="任务状态")
    progress: int = Field(default=0, ge=0, le=100, description="进度百分比")
    current_step: str = Field(default="", description="当前步骤描述")
    message: str = Field(default="", description="状态消息")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    error: Optional[str] = Field(None, description="错误信息（如果失败）")


# ============ 上传响应 ============
class UploadResponse(BaseModel):
    """文件上传响应"""
    files: List[FileInfo]
    count: int = Field(..., description="文件数量")


# ============ 任务创建请求 ============
class CreateTaskRequest(BaseModel):
    """创建任务请求"""
    file_ids: List[str] = Field(..., description="要处理的文件ID列表")
    task_type: TaskType = Field(default=TaskType.TRANSCRIBE_AND_TRANSLATE, description="任务类型")


# ============ 任务创建响应 ============
class CreateTaskResponse(BaseModel):
    """创建任务响应"""
    tasks: List[TaskInfo]
    count: int


# ============ API 渠道配置 ============
class ApiChannelConfig(BaseModel):
    """API 渠道配置"""
    key: str = Field(..., description="API Key 或环境变量名")
    base_url: str = Field(..., description="API Base URL")
    model: str = Field(..., description="模型名称")


# ============ ASR 配置 ============
class AsrConfig(BaseModel):
    """ASR 配置"""
    language: str = Field(default="en", description="源语言 ISO 639-1 代码")
    runtime: Literal["custom", "parakeet"] = Field(default="custom", description="ASR 引擎")


# ============ 热词分组 ============
class HotwordGroup(BaseModel):
    """热词分组"""
    id: str = Field(..., description="分组唯一标识")
    name: str = Field(..., description="分组显示名称")
    keyterms: List[str] = Field(default_factory=list, description="热词列表")


# ============ 热词矫正配置 ============
class HotwordCorrectionConfig(BaseModel):
    """热词矫正配置"""
    enabled: bool = Field(default=False, description="是否启用")
    active_group_id: str = Field(default="group-0", description="激活分组 ID")
    groups: List[HotwordGroup] = Field(default_factory=list, description="分组列表")


# ============ 人声分离配置 ============
class VocalSeparationConfig(BaseModel):
    """人声分离配置"""
    enabled: bool = Field(default=False, description="是否启用")


# ============ 高级配置 ============
class AdvancedConfig(BaseModel):
    """高级配置"""
    max_workers: int = Field(default=8, ge=1, le=32, description="LLM 并发数")
    summary_length: int = Field(default=8000, ge=1000, le=32000, description="摘要长度限制")
    pause_split_threshold: Optional[float] = Field(None, ge=0, description="停顿切分阈值（秒）")


# ============ 完整配置 ============
class AppConfig(BaseModel):
    """应用配置"""
    api_channels: Dict[str, ApiChannelConfig] = Field(
        default_factory=lambda: {
            "api": ApiChannelConfig(key="", base_url="", model=""),
            "api_split": ApiChannelConfig(key="", base_url="", model=""),
            "api_summary": ApiChannelConfig(key="", base_url="", model=""),
            "api_translate": ApiChannelConfig(key="", base_url="", model=""),
            "api_reflection": ApiChannelConfig(key="", base_url="", model=""),
            "api_hotword": ApiChannelConfig(key="", base_url="", model=""),
        }
    )
    asr: AsrConfig = Field(default_factory=AsrConfig)
    target_language: str = Field(default="简体中文", description="目标语言")
    transcript_only: bool = Field(default=False, description="仅转录模式")
    hotword_correction: HotwordCorrectionConfig = Field(default_factory=HotwordCorrectionConfig)
    vocal_separation: VocalSeparationConfig = Field(default_factory=VocalSeparationConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)


# ============ 术语 ============
class Term(BaseModel):
    """术语"""
    original: str = Field(..., description="原文")
    translation: str = Field(..., description="译文")
    notes: Optional[str] = Field("", description="说明")


# ============ 术语列表 ============
class TermsList(BaseModel):
    """术语列表"""
    terms: List[Term]
    count: int = None
