"""
配置管理接口
处理应用配置的读取、保存、重置等操作
操作 config.yaml 文件
"""

from fastapi import APIRouter, HTTPException, Request
import os
import sys
import logging

logger = logging.getLogger(__name__)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from api.models.schemas import AppConfig

router = APIRouter()

# 启动时验证模块加载
logger.info("✅ api/routes/config.py loaded - DEBUG MODE")




def convert_yaml_to_app_config() -> AppConfig:
    """
    从 config.yaml 转换为 AppConfig 格式
    使用现有的 load_key 函数读取配置
    """
    try:
        from core.utils.config_utils import load_key

        # 读取 API 渠道配置
        api_channels = {
            "api": {
                "key": load_key("api.key") or "",
                "base_url": load_key("api.base_url") or "",
                "model": load_key("api.model") or ""
            },
            "api_split": {
                "key": load_key("api_split.key") or "",
                "base_url": load_key("api_split.base_url") or "",
                "model": load_key("api_split.model") or ""
            },
            "api_summary": {
                "key": load_key("api_summary.key") or "",
                "base_url": load_key("api_summary.base_url") or "",
                "model": load_key("api_summary.model") or ""
            },
            "api_translate": {
                "key": load_key("api_translate.key") or "",
                "base_url": load_key("api_translate.base_url") or "",
                "model": load_key("api_translate.model") or ""
            },
            "api_reflection": {
                "key": load_key("api_reflection.key") or "",
                "base_url": load_key("api_reflection.base_url") or "",
                "model": load_key("api_reflection.model") or ""
            },
            "api_hotword": {
                "key": load_key("api_hotword.key") or "",
                "base_url": load_key("api_hotword.base_url") or "",
                "model": load_key("api_hotword.model") or ""
            }
        }

        # 读取 ASR 配置
        asr_config = {
            "language": load_key("asr.language"),
            "runtime": load_key("asr.runtime"),
            "model": load_key("asr.model", default="Qwen3-ASR-0.6B")
        }

        # 读取其他配置
        target_language = load_key("target_language")
        transcript_only = load_key("transcript_only")

        # 读取热词矫正配置（分组结构）
        hotword_groups = load_key("asr_term_correction.groups") or []
        hotword_enabled = load_key("asr_term_correction.enabled") or False
        hotword_active_group_id = load_key("asr_term_correction.active_group_id") or "group-0"

        if not hotword_groups:
            hotword_groups = [{"id": "group-0", "name": "默认分组", "keyterms": []}]

        hotword_correction = {
            "enabled": hotword_enabled,
            "active_group_id": hotword_active_group_id,
            "groups": hotword_groups
        }

        # 读取人声分离配置
        vocal_separation = {
            "enabled": load_key("vocal_separation.enabled")
        }

        # 读取高级配置
        advanced = {
            "max_workers": load_key("max_workers"),
            "summary_length": load_key("summary_length"),
            "pause_split_threshold": load_key("pause_split_threshold")
        }

        return AppConfig(
            api_channels=api_channels,
            asr=asr_config,
            target_language=target_language,
            transcript_only=transcript_only,
            hotword_correction=hotword_correction,
            vocal_separation=vocal_separation,
            advanced=advanced
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"配置读取失败: {str(e)}")


def convert_app_config_to_yaml(config: AppConfig):
    """
    将 AppConfig 格式转换回 config.yaml
    使用批量更新方法提高效率
    """
    try:
        from core.utils.config_manager import update_keys

        # 构建所有需要更新的键值对
        updates = {}

        # API 渠道配置
        for channel, values in config.api_channels.items():
            updates[f"{channel}.key"] = values.key
            updates[f"{channel}.base_url"] = values.base_url
            updates[f"{channel}.model"] = values.model

        # ASR 配置
        updates["asr.language"] = config.asr.language
        updates["asr.runtime"] = config.asr.runtime
        updates["asr.model"] = config.asr.model

        # 其他配置
        updates["target_language"] = config.target_language
        updates["transcript_only"] = config.transcript_only

        # 热词矫正配置（分组结构）
        updates["asr_term_correction.enabled"] = config.hotword_correction.enabled
        updates["asr_term_correction.active_group_id"] = config.hotword_correction.active_group_id
        updates["asr_term_correction.groups"] = [g.model_dump() for g in config.hotword_correction.groups]

        # 人声分离配置
        updates["vocal_separation.enabled"] = config.vocal_separation.enabled

        # 高级配置
        updates["max_workers"] = config.advanced.max_workers
        updates["summary_length"] = config.advanced.summary_length
        updates["pause_split_threshold"] = config.advanced.pause_split_threshold

        # 批量更新所有配置
        update_keys(updates)

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"配置保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"配置保存失败: {str(e)}")


@router.get("/config", response_model=AppConfig)
async def get_config():
    """
    获取应用配置

    从 config.yaml 读取当前配置（基础 + 高级）
    """
    return convert_yaml_to_app_config()


@router.put("/config")
async def save_config(request: Request):
    """
    保存配置

    将配置更新保存到 config.yaml
    """
    try:
        import json
        body = await request.body()
        body_dict = json.loads(body)
        config = AppConfig(**body_dict)
        convert_app_config_to_yaml(config)
        return {"success": True, "message": "配置已保存"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"配置保存失败: {str(e)}")


@router.post("/config/reset")
async def reset_config():
    """
    重置配置为默认值
    """
    try:
        from core.utils.config_utils import update_key
        import shutil

        # 备份当前配置
        if os.path.exists("config.yaml"):
            backup_file = f"config.yaml.backup.{int(__import__('time').time())}"
            shutil.copy("config.yaml", backup_file)
            logger.info(f"已备份当前配置到: {backup_file}")

        # 重置为默认值
        defaults = {
            # API 渠道 - 重置为空
            "api.key": "",
            "api.base_url": "",
            "api.model": "",
            "api_split.key": "",
            "api_split.base_url": "",
            "api_split.model": "",
            "api_summary.key": "",
            "api_summary.base_url": "",
            "api_summary.model": "",
            "api_translate.key": "",
            "api_translate.base_url": "",
            "api_translate.model": "",
            "api_reflection.key": "",
            "api_reflection.base_url": "",
            "api_reflection.model": "",
            "api_hotword.key": "",
            "api_hotword.base_url": "",
            "api_hotword.model": "",

            # ASR 配置
            "asr.language": "en",
            "asr.runtime": "qwen",
            "asr.model": "Qwen3-ASR-0.6B",

            # 其他配置
            "target_language": "简体中文",
            "transcript_only": False,
            "asr_term_correction.enabled": False,
            "asr_term_correction.active_group_id": "group-0",
            "asr_term_correction.groups": [{"id": "group-0", "name": "默认分组", "keyterms": []}],

            # 人声分离
            "vocal_separation.enabled": False,

            # 高级配置
            "max_workers": 8,
            "summary_length": 8000,
            "pause_split_threshold": None,
        }

        for key, value in defaults.items():
            if value is None:
                # 对于 None 值，跳过（相当于删除该配置）
                continue
            update_key(key, value)

        logger.info("配置已重置为默认值")
        return {
            "success": True,
            "message": "配置已重置为默认值",
            "backup": backup_file if os.path.exists("config.yaml") else None
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"重置配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"重置配置失败: {str(e)}")
