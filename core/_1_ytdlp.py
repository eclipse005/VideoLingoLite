import os,sys
import glob
from core.utils import *

def find_video_files(save_path='output'):
    """
    查找视频或音频媒体文件
    支持所有在config.yaml中定义的视频和音频格式
    """
    # 支持视频和音频文件
    all_formats = load_key("allowed_video_formats") + load_key("allowed_audio_formats")
    media_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in all_formats]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        media_files = [file.replace("\\", "/") for file in media_files]
    # Filter out the "output/output" directory (if exists), but keep "output/output.wav"
    media_files = [file for file in media_files if file != "output/output"]
    if len(media_files) != 1:
        raise ValueError(f"Number of media files found {len(media_files)} is not unique. Please check.")
    return media_files[0]

if __name__ == '__main__':
    print(f"🎥 Found video file: {find_video_files()}")
