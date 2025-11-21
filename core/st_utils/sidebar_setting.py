import streamlit as st
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *

def config_input(label, key, help=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help)
    if val != load_key(key):
        update_key(key, val)
    return val

@st.fragment
def llm_config_section():
    with st.expander(t("LLM Configuration"), expanded=True):
        # API功能选择
        api_channel = st.selectbox(
            t("Process"),
            options=["Other", "Split", "Summary", "Translate", "Reflection"],
            index=0,
            format_func=lambda x: {
                "Other": t("Other"),
                "Split": t("Split"),
                "Summary": t("Summary"),
                "Translate": t("Translate"),
                "Reflection": t("Reflection")
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

        config_input(t("API_KEY"), f"{current_api_prefix}.key")
        config_input(t("BASE_URL"), f"{current_api_prefix}.base_url", help=t("Openai format, will add /v1/chat/completions automatically"))

        c1, c2 = st.columns([4, 1])
        with c1:
            config_input(t("MODEL"), f"{current_api_prefix}.model", help=t("click to check API validity")+ " 👉")
        with c2:
            if st.button("📡", key=f"api_{current_api_prefix}"):
                st.toast(t("API Key is valid") if check_api(current_api_prefix) else t("API Key is invalid"),
                        icon="✅" if check_api(current_api_prefix) else "❌")

@st.fragment
def subtitle_settings_section():
    with st.expander(t("Subtitles Settings"), expanded=True):
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
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("asr.language"))
            )
            if langs[lang] != load_key("asr.language"):
                update_key("asr.language", langs[lang])

        runtime = st.selectbox(t("ASR Runtime"), options=["gemini"], index=["gemini"].index(load_key("asr.runtime")), help=t("Select ASR service for transcription"))
        if runtime != load_key("asr.runtime"):
            update_key("asr.runtime", runtime)

        with c2:
            target_language = st.text_input(t("Target Lang"), value=load_key("target_language"), help=t("Input any language in natural language, as long as llm can understand"))
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)

        burn_subtitles = st.toggle(t("Burn-in Subtitles"), value=load_key("burn_subtitles"), help=t("Whether to burn subtitles into the video, will increase processing time"))
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)

def page_setting():

    display_language = st.selectbox("Display Language 🌐",
                                  options=list(DISPLAY_LANGUAGES.keys()),
                                  index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")))
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

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
