"""
API 测试连接路由
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
from openai import OpenAI

router = APIRouter()


class TestConnectionRequest(BaseModel):
    """测试连接请求"""
    channel: str  # API 渠道名称
    key: str  # API 密钥
    base_url: str  # API 基础 URL
    model: str  # 模型名称


class TestConnectionResponse(BaseModel):
    """测试连接响应"""
    success: bool
    message: str
    details: Optional[dict] = None


@router.post("/test-connection", response_model=TestConnectionResponse)
async def test_api_connection(request: TestConnectionRequest):
    """
    测试 API 连接

    发送一个简单的测试请求到指定的 API 端点，验证配置是否正确。
    """
    try:
        # 验证必要参数
        if not request.key:
            return TestConnectionResponse(
                success=False,
                message="API 密钥不能为空"
            )

        if not request.base_url:
            return TestConnectionResponse(
                success=False,
                message="API 基础 URL 不能为空"
            )

        if not request.model:
            return TestConnectionResponse(
                success=False,
                message="模型名称不能为空"
            )

        # 创建 OpenAI 客户端
        client = OpenAI(
            api_key=request.key,
            base_url=request.base_url,
            timeout=10.0  # 10 秒超时
        )

        # 发送测试请求 - 使用最简单的聊天完成请求
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=request.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=10,
                timeout=10.0
            )

            # 测试成功
            usage = response.usage if hasattr(response, 'usage') else None
            return TestConnectionResponse(
                success=True,
                message=f"连接成功！模型: {request.model}",
                details={
                    "channel": request.channel,
                    "model": request.model,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                }
            )

        except Exception as api_error:
            error_message = str(api_error)

            # 解析常见错误
            if "401" in error_message or "authentication" in error_message.lower():
                return TestConnectionResponse(
                    success=False,
                    message="API 密钥无效或未授权"
                )
            elif "404" in error_message:
                return TestConnectionResponse(
                    success=False,
                    message=f"模型 '{request.model}' 不存在"
                )
            elif "timeout" in error_message.lower() or "timed out" in error_message.lower():
                return TestConnectionResponse(
                    success=False,
                    message="连接超时，请检查网络或 API 地址"
                )
            elif "connection" in error_message.lower():
                return TestConnectionResponse(
                    success=False,
                    message="无法连接到服务器，请检查 URL 是否正确"
                )
            else:
                return TestConnectionResponse(
                    success=False,
                    message=f"API 调用失败: {error_message}"
                )

    except Exception as e:
        return TestConnectionResponse(
            success=False,
            message=f"测试失败: {str(e)}"
        )
