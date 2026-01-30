import streamlit as st
from core.utils import *

def config_input(label, key, help=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help)
    if val != load_key(key):
        update_key(key, val)
    return val

@st.fragment
def llm_config_section():
    with st.expander("LLM配置", expanded=True):
        # API功能选择
        api_channel = st.selectbox(
            "步骤",
            options=["Other", "Split", "Summary", "Translate", "Reflection"],
            index=0,
            format_func=lambda x: {
                "Other": "其他",
                "Split": "分割",
                "Summary": "总结",
                "Translate": "直译",
                "Reflection": "反思"
            }[x]
        )

        # 根据选择的功能确定API配置前缀
        api_prefix_map = {
            "Other": "api",
            "Split": "api_split",
            "Summary": "api_summary",
            "Translate": "api_translate",
            "Reflection": "api_reflection"
        }
        current_api_prefix = api_prefix_map[api_channel]

        config_input("API密钥", f"{current_api_prefix}.key")
        config_input("BASE_URL", f"{current_api_prefix}.base_url", help="OpenAI格式,将自动添加/v1/chat/completions")

        c1, c2 = st.columns([4, 1])
        with c1:
            config_input("模型", f"{current_api_prefix}.model", help="点击检查API有效性"+ " 👉")
        with c2:
            st.write("") # Add a spacer
            if st.button("📡", key=f"api_{current_api_prefix}"):
                st.toast("API密钥有效" if check_api(current_api_prefix) else "API密钥无效",
                        icon="✅" if check_api(current_api_prefix) else "❌")

@st.fragment
def subtitle_settings_section():
    with st.expander("字幕设置", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇯🇵 日本語": "ja",
                "🇰🇷 한국어": "ko",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it"
            }
            lang = st.selectbox(
                "识别语言",
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("asr.language"))
            )
            if langs[lang] != load_key("asr.language"):
                update_key("asr.language", langs[lang])

        runtime_options = ["qwen"]
        runtime = st.selectbox("语音识别引擎", options=runtime_options, index=0, help="Qwen3-ASR 本地语音识别（52种语言+22中文方言）")

        if runtime != load_key("asr.runtime"):
            update_key("asr.runtime", runtime)

        # Qwen3-ASR model selection
        if runtime == "qwen":
            model_options = ["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"]
            current_model = load_key("asr.model", default="Qwen3-ASR-1.7B")
            model = st.selectbox(
                "Qwen3-ASR 模型",
                options=model_options,
                index=model_options.index(current_model) if current_model in model_options else 0,
                help="1.7B: 更高准确率 | 0.6B: 更快速度"
            )
            if model != current_model:
                update_key("asr.model", model)

        # 人声分离和只转录开关（同一排）
        col1, col2 = st.columns(2)
        with col1:
            vocal_sep_enabled = st.toggle("人声分离", value=load_key("vocal_separation.enabled"),
                                         help="嘈杂环境下启用，有助于提升转录准确率")
            if vocal_sep_enabled != load_key("vocal_separation.enabled"):
                update_key("vocal_separation.enabled", vocal_sep_enabled)

        with col2:
            transcript_only_enabled = st.toggle("只转录", value=load_key("transcript_only"),
                                                help="跳过翻译流程，仅生成原文字幕")
            if transcript_only_enabled != load_key("transcript_only"):
                update_key("transcript_only", transcript_only_enabled)

        # 热词功能（开关在上方，输入框在下方）
        hotword_enabled = st.toggle("热词", value=load_key("asr.hotword_enabled"),
                                   help="启用热词提升专业术语识别准确率")
        if hotword_enabled != load_key("asr.hotword_enabled"):
            update_key("asr.hotword_enabled", hotword_enabled)

        hotword = st.text_input("热词内容", value=load_key("asr.hotword", default=""),
                               help="输入专业术语，用空格分隔，如：15 minute 交易 停滞",
                               disabled=not load_key("asr.hotword_enabled"))
        if hotword != load_key("asr.hotword", default=""):
            update_key("asr.hotword", hotword)

        with c2:
            target_language = st.text_input("目标语言", value=load_key("target_language"), help="用自然语言输入任何语言,只要LLM能理解即可")
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)

def page_setting():

    # with st.expander(t("Youtube Settings"), expanded=True):
    #     config_input(t("Cookies Path"), "youtube.cookies_path")

    llm_config_section()
    subtitle_settings_section()

def check_api(api_prefix="api"):
    try:
        # 根据API前缀获取相应的key，用于测试
        api_key = load_key(f"{api_prefix}.key")
        if not api_key or api_key == 'YOUR_API_KEY_HERE':
            return False
        resp = ask_gpt("This is a test, response 'message':'success' in json format.",
                      resp_type="json", log_title='None')
        return resp.get('message') == 'success'
    except Exception:
        return False

if __name__ == "__main__":
    check_api()
