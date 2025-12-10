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
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja"
            }
            lang = st.selectbox(
                "识别语言",
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("asr.language"))
            )
            if langs[lang] != load_key("asr.language"):
                update_key("asr.language", langs[lang])

        runtime = st.selectbox("语音识别引擎", options=["gemini", "whisperX"], index=["gemini", "whisperX"].index(load_key("asr.runtime")), help="选择ASR服务进行转录")
        if runtime != load_key("asr.runtime"):
            update_key("asr.runtime", runtime)

        if runtime == "whisperX":
            use_hotwords = st.toggle(
                "启用热词", 
                value=load_key("asr.use_hotwords"), 
                help="提供逗号分隔的单词列表以提高识别准确性。"
            )
            if use_hotwords != load_key("asr.use_hotwords"):
                update_key("asr.use_hotwords", use_hotwords)

            if use_hotwords:
                hotwords = st.text_input(
                    "热词", 
                    value=load_key("asr.hotwords")
                )
                if hotwords != load_key("asr.hotwords"):
                    update_key("asr.hotwords", hotwords)

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
