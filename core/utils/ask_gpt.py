import os
import json
import re
import base64
from threading import Lock
from typing import List, Dict, Any, Optional
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'

# Global variables for token usage tracking
TOTAL_TOKENS = {
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0
}

# Mapping of log_title to specific API configurations
LOG_TITLE_TO_API = {
    # Splitting
    'llm_sentence_segmentation': 'api_split',
    'split_single_sentence': 'api_split',

    # Summary
    'summary': 'api_summary',

    # Translation
    'translate_faithfulness': 'api_translate',

    # Reflection
    'translate_expressiveness': 'api_reflection',

    # Hotword correction
    'hotword_correction': 'api_hotword',
}

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append({"model": model, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False

def _get_api_config_for_log_title(log_title):
    """根据 log_title 获取对应的API配置前缀"""
    # 检查精确匹配
    if log_title in LOG_TITLE_TO_API:
        return LOG_TITLE_TO_API[log_title]

    # 检查是否是翻译相关的（如 translate_faithfulness, translate_expressiveness）
    if log_title.startswith('translate_'):
        if 'faith' in log_title:
            return 'api_translate'
        elif 'express' in log_title:
            return 'api_reflection'
        else:
            return 'api_translate'  # 默认翻译

    # 如果没有找到匹配项，返回默认API配置
    return 'api'

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=5, verbose=False)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    # 根据 log_title 确定使用的API配置
    api_config_prefix = _get_api_config_for_log_title(log_title)

    # 检查API密钥是否设置
    api_key = load_key(f"{api_config_prefix}.key")
    if not api_key:
        raise ValueError(f"API key for {api_config_prefix} is not set")

    # check cache
    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        # rprint("use cache response")  # 缓存日志会干扰进度条显示
        # Return cached result without incrementing token usage
        return cached

    model = load_key(f"{api_config_prefix}.model")
    base_url = load_key(f"{api_config_prefix}.base_url")

    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [{"role": "user", "content": prompt}]

    params = dict(
        model=model,
        messages=messages,
        timeout=300
    )
    resp_raw = client.chat.completions.create(**params)

    # process and return full result
    resp_content = resp_raw.choices[0].message.content

    # Extract and accumulate token usage
    if hasattr(resp_raw, 'usage') and resp_raw.usage is not None:
        with LOCK:
            TOTAL_TOKENS['prompt_tokens'] += resp_raw.usage.prompt_tokens
            TOTAL_TOKENS['completion_tokens'] += resp_raw.usage.completion_tokens
            TOTAL_TOKENS['total_tokens'] += resp_raw.usage.total_tokens
    if resp_type == "json":
        resp = json_repair.loads(resp_content)
    else:
        resp = resp_content

    # check if the response format is valid
    message = None
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")
        message = valid_resp['message']

    # Save with message (could be None)
    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title, message=message)
    return resp


# ------------
# ask gpt with tools (function calling)
# ------------

def ask_gpt_with_tools(
    system_prompt: str,
    prompt: str,
    tools: List[dict],
    tool_executor: object,
    max_rounds: int = 10,
    log_title: str = "default"
) -> Any:
    """
    支持 Function Calling 的 LLM 调用

    Args:
        system_prompt: 系统提示词
        prompt: 用户任务描述
        tools: Function Calling 工具定义列表
        tool_executor: 工具执行器实例（需要有 execute_tool 方法）
        max_rounds: 最大对话轮次
        log_title: 日志标题（用于缓存和 API 配置选择）

    Returns:
        工具执行器的最终结果
    """
    # 根据 log_title 确定使用的 API 配置
    api_config_prefix = _get_api_config_for_log_title(log_title)

    # 检查 API 密钥
    api_key = load_key(f"{api_config_prefix}.key")
    if not api_key:
        raise ValueError(f"API key for {api_config_prefix} is not set")

    model = load_key(f"{api_config_prefix}.model")
    base_url = load_key(f"{api_config_prefix}.base_url")

    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 构建初始消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # 多轮对话循环
    for round_num in range(max_rounds):
        # 调用 LLM
        params = dict(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            timeout=300
        )

        resp_raw = client.chat.completions.create(**params)
        response_message = resp_raw.choices[0].message

        # 累加 Token 统计
        if hasattr(resp_raw, 'usage') and resp_raw.usage is not None:
            with LOCK:
                TOTAL_TOKENS['prompt_tokens'] += resp_raw.usage.prompt_tokens
                TOTAL_TOKENS['completion_tokens'] += resp_raw.usage.completion_tokens
                TOTAL_TOKENS['total_tokens'] += resp_raw.usage.total_tokens

        # 添加助手响应到历史（转换为字典格式）
        msg_dict = {
            "role": "assistant",
            "content": response_message.content if hasattr(response_message, 'content') else None
        }
        if response_message.tool_calls:
            msg_dict["tool_calls"] = response_message.tool_calls
        messages.append(msg_dict)

        # 检查是否有工具调用
        tool_calls = response_message.tool_calls
        if tool_calls:
            # 执行每个工具调用
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                # arguments 可能是字符串或字典
                if isinstance(tool_call.function.arguments, str):
                    func_args = json.loads(tool_call.function.arguments)
                else:
                    func_args = tool_call.function.arguments

                # 调用执行器的方法
                try:
                    result = getattr(tool_executor, func_name)(**func_args)
                except Exception as e:
                    import traceback
                    result = f"错误：{str(e)}\n{traceback.format_exc()}"

                # 添加工具结果到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

                # 检查是否完成
                if func_name == "finish":
                    return json.loads(result) if isinstance(result, str) else result
        else:
            # LLM 返回文本内容，没有工具调用
            content = msg_dict.get("content") or ""
            rprint(f"[yellow][{log_title}] LLM 返回了文本内容而非调用工具: {content[:200]}[/yellow]")
            break

    # 达到最大轮次或 LLM 停止调用工具
    rprint(f"[yellow][{log_title}] LLM 未正常完成（达到最大轮次或停止调用工具）[/yellow]")
    return None



if __name__ == '__main__':
    from rich import print as rprint

    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
