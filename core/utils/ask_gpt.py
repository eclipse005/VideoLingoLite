import os
import json
import re
from threading import Lock
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
    # 分割功能
    'llm_sentence_segmentation': 'api_split',
    'split_single_sentence': 'api_split',

    # 总结功能
    'summary': 'api_summary',

    # 翻译功能
    'translate_faithfulness': 'api_translate',

    # 反思功能
    'translate_expressiveness': 'api_reflection',
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

@except_handler("GPT request failed", retry=5)
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
        rprint("use cache response")
        # Return cached result without incrementing token usage
        return cached

    model = load_key(f"{api_config_prefix}.model")
    base_url = load_key(f"{api_config_prefix}.base_url")
    llm_support_json = load_key(f"{api_config_prefix}.llm_support_json")

    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    client = OpenAI(api_key=api_key, base_url=base_url)
    response_format = {"type": "json_object"} if resp_type == "json" and llm_support_json else None

    messages = [{"role": "user", "content": prompt}]

    params = dict(
        model=model,
        messages=messages,
        response_format=response_format,
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
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'])
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp


def get_token_usage():
    """
    Get the accumulated token usage statistics
    Returns a dictionary with prompt_tokens, completion_tokens, and total_tokens
    """
    with LOCK:
        return TOTAL_TOKENS.copy()


def reset_token_usage():
    """
    Reset the accumulated token usage statistics to zero
    """
    with LOCK:
        TOTAL_TOKENS['prompt_tokens'] = 0
        TOTAL_TOKENS['completion_tokens'] = 0
        TOTAL_TOKENS['total_tokens'] = 0


if __name__ == '__main__':
    from rich import print as rprint

    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
