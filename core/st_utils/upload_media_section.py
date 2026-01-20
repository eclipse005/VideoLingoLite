import os
import re
import shutil
import subprocess
from time import sleep

import streamlit as st
from core._1_ytdlp import find_video_files
from core.utils import *


OUTPUT_DIR = "output"

def upload_media_section():
    st.header("a. 上传视频")
    with st.container(border=True):
        try:
            media_file = find_video_files()
            # 根据文件扩展名决定显示方式
            ext = os.path.splitext(media_file)[1][1:].lower()
            if ext in load_key("allowed_audio_formats"):
                st.audio(media_file)
            else:
                st.video(media_file)
            if st.button("删除并重新选择", key="delete_video_button"):
                os.remove(media_file)
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                sleep(1)
                # 删除后重新运行以显示上传选项
                st.rerun()
            return True
        except:
            # 只显示文件上传功能，移除YouTube下载功能
            uploaded_file = st.file_uploader(
                "上传视频文件",
                type=load_key("allowed_video_formats") + load_key("allowed_audio_formats")
            )

            if uploaded_file:
                if os.path.exists(OUTPUT_DIR):
                    shutil.rmtree(OUTPUT_DIR)
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                raw_name = uploaded_file.name.replace(' ', '_')
                name, ext = os.path.splitext(raw_name)
                clean_name = re.sub(r'[^\w\-_\.]', '', name) + ext.lower()

                with open(os.path.join(OUTPUT_DIR, clean_name), "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 音频文件直接处理，不再转换成MP4
                # 后续的 convert_video_to_audio() 函数可以处理音频文件
                # 上传完成后重新运行以显示文件
                st.rerun()
            else:
                return False
