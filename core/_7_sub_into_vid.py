import os, subprocess, time
from core._1_ytdlp import find_video_files
import cv2
import numpy as np
import platform
from core.utils import *

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Arial'

# Linux need to install google noto fonts: apt-get install fonts-noto
if platform.system() == 'Linux':
    FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
# Mac OS has different font names
elif platform.system() == 'Darwin':
    FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/output_sub.mp4"
SRC_SRT = f"{OUTPUT_DIR}/src.srt"
TRANS_SRT = f"{OUTPUT_DIR}/trans.srt"
    
def get_best_encoder():
    """
    自动检测可用的最佳硬件编码器
    优先级: Intel (qsv) > NVIDIA (nvenc) > Mac (videotoolbox) > CPU (libx264)
    """
    try:
        # 获取 ffmpeg 支持的所有编码器列表
        result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True).stdout

        # 检测编码器是否实际可用（测试编码器初始化）
        def is_encoder_available(encoder_name):
            try:
                test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=1',
                           '-c:v', encoder_name, '-t', '1', '-f', 'null', 'NUL']
                # 只保留错误输出，隐藏标准输出
                test_result = subprocess.run(test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                return test_result.returncode == 0
            except Exception:
                return False

        # 优先检测Intel QSV，因为它在你的系统上应该可用
        if 'h264_qsv' in result and is_encoder_available('h264_qsv'):
            return 'h264_qsv'         # Intel 核显 (支持 Ultra 系列)

        # 然后检测NVIDIA，但要验证实际可用性
        elif 'h264_nvenc' in result and is_encoder_available('h264_nvenc'):
            return 'h264_nvenc'       # NVIDIA 显卡

        elif 'h264_videotoolbox' in result:
            return 'h264_videotoolbox' # Mac 系统
    except Exception as e:
        print(f"Encoder detection failed: {e}")
        pass

    return 'libx264' # 默认回落到 CPU

def merge_subtitles_to_video():
    video_file = find_video_files()
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    # Check resolution
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    if not os.path.exists(SRC_SRT) or not os.path.exists(TRANS_SRT):
        rprint("Subtitle files not found in the 'output' directory.")
        exit(1)

    video = cv2.VideoCapture(video_file)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    # 构建 FFmpeg 基础命令
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_file,
        '-vf', (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"subtitles={SRC_SRT}:force_style='FontSize={SRC_FONT_SIZE},FontName={FONT_NAME}," 
            f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={SRC_OUTLINE_WIDTH},"
            f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1',"
            f"subtitles={TRANS_SRT}:force_style='FontSize={TRANS_FONT_SIZE},FontName={TRANS_FONT_NAME},"
            f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        ).encode('utf-8'),
    ]

    # 智能选择编码器
    ffmpeg_gpu = load_key("ffmpeg_gpu")
    encoder = 'libx264' # 默认 CPU
    
    if ffmpeg_gpu:
        detected_encoder = get_best_encoder()
        if detected_encoder != 'libx264':
            encoder = detected_encoder
            rprint(f"[bold green]🚀 Will use GPU acceleration: {encoder}[/bold green]")
        else:
            rprint("[bold yellow]⚠️ GPU requested but no hardware encoder found. Falling back to CPU.[/bold yellow]")

    ffmpeg_cmd.extend(['-c:v', encoder])

    # 根据不同的编码器应用优化参数
    if encoder == 'h264_nvenc':
        ffmpeg_cmd.extend(['-preset', 'p4', '-cq', '23'])
    elif encoder == 'h264_qsv':
        # Intel QSV 参数：global_quality 类似 crf，look_ahead 提升画质
        ffmpeg_cmd.extend(['-global_quality', '23', '-look_ahead', '1'])
    elif encoder == 'h264_videotoolbox':
        ffmpeg_cmd.extend(['-q:v', '50'])
    else:
        # CPU libx264 参数
        ffmpeg_cmd.extend(['-preset', 'medium', '-crf', '23'])

    ffmpeg_cmd.extend(['-y', OUTPUT_VIDEO])

    rprint("🎬 Start merging subtitles to video...")
    start_time = time.time()

    # 打印完整命令以便调试
    # print(" ".join([str(x) for x in ffmpeg_cmd]))

    # 执行ffmpeg命令，隐藏详细输出，只在错误时显示
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        stderr, _ = process.communicate()  # 等待进程完成并获取错误输出
        if process.returncode == 0:
            rprint(f"\n✅ Done! Time taken: {time.time() - start_time:.2f} seconds")
        else:
            rprint(f"\n❌ FFmpeg execution error\nError output: {stderr.decode('utf-8', errors='ignore')}")
    except Exception as e:
        rprint(f"\n❌ Error occurred: {e}")
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    merge_subtitles_to_video()