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

def convert_audio_to_video(audio_file: str) -> str:
    """
    ⚠️ 此函数已不再使用
    现在音频文件直接处理，无需转换为MP4
    保留此函数以备将来需要生成带黑屏的视频
    """
    output_video = os.path.join(OUTPUT_DIR, 'black_screen.mp4')
    if not os.path.exists(output_video):
        print(f"🎵➡️🎬 Converting audio to video with FFmpeg ......")
        ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=640x360', '-i', audio_file, '-shortest', '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p', output_video]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"🎵➡️🎬 Converted <{audio_file}> to <{output_video}> with FFmpeg\n")
        # delete audio file
        os.remove(audio_file)
    return output_video

# 保持向后兼容的别名
def download_video_section():
    return upload_media_section()
