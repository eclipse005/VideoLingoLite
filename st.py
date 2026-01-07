# Windows compatibility fix for NeMo toolkit (SIGKILL not available on Windows)
import sys
import signal
if sys.platform == 'win32' and not hasattr(signal, 'SIGKILL'):
    signal.SIGKILL = signal.SIGTERM

import streamlit as st
import os
from core.st_utils.imports_and_utils import download_subtitle_zip_button, give_star_button, button_style, page_setting
from core.st_utils.upload_media_section import upload_media_section
from core import *
from core.utils.ask_gpt import get_token_usage

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PATH'] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/src.srt"

@st.fragment
def text_processing_section():
    st.header("b. 翻译并生成字幕")
    with st.container(border=True):
        st.markdown(f"""
        <p style='font-size: 20px;'>
        {"此阶段包含以下步骤:"}
        <p style='font-size: 20px;'>
            1. {"语音识别转录"}<br>
            2. {"使用LLM进行句子分段"}<br>
            3. {"切分长句"}<br>
            4. {"摘要和多步翻译"}<br>
            5. {"切割和对齐长字幕"}<br>
            6. {"生成时间轴和字幕"}<br>
        """, unsafe_allow_html=True)

        if not os.path.exists(SUB_VIDEO):
            if st.button("开始处理字幕", key="text_processing_button"):
                process_text()
                st.rerun()
        else:
            # Subtitle merging functionality has been removed
            download_subtitle_zip_button(text="下载所有Srt文件")

            if st.button("归档到'history'", key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True

def process_text():
    with st.spinner("正在使用语音识别进行转录..."):
        _2_asr.transcribe()
    with st.spinner("正在使用LLM进行句子分段..."):
        _3_llm_sentence_split.llm_sentence_split()
    with st.spinner("正在切分长句..."):
        _3_2_split_meaning.split_sentences_by_meaning()
    with st.spinner("正在总结和翻译..."):
        _4_1_summarize.get_summary()
        if load_key("pause_before_translate"):
            input("⚠️ 翻译前暂停。请前往`output/log/terminology.json`编辑术语。然后按回车键继续...")
        _4_2_translate.translate_all()
    with st.spinner("正在处理和对齐字幕..."):
        _5_split_sub.split_for_sub_main()
        _6_gen_sub.align_timestamp_main()

    st.success("字幕处理完成! 🎉")
    st.balloons()

    # Print token usage statistics to console
    token_usage = get_token_usage()
    print(f"\n--- GPT Token Usage Statistics ---")
    print(f"Total Prompt Tokens: {token_usage['prompt_tokens']}")
    print(f"Total Completion Tokens: {token_usage['completion_tokens']}")
    print(f"Total Tokens: {token_usage['total_tokens']}")
    print(f"--- End of Token Usage ---\n")

def main():
    # logo_col, _ = st.columns([1,1])
    # with logo_col:
    #     st.image("docs/logo.png", use_column_width=True)
    st.markdown(button_style, unsafe_allow_html=True)
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)
    upload_media_section()
    text_processing_section()

if __name__ == "__main__":
    main()
