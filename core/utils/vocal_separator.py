"""
音频人声分离模块 - 使用 audio-separator 库
用于嘈杂环境下提升 ASR 转录准确率
"""
import os
from audio_separator.separator import Separator
from core.utils.models import _RAW_AUDIO_FILE, _VOCAL_AUDIO_FILE
from core.utils.config_utils import load_key
from rich.console import Console

console = Console()

# 模型配置
VOCAL_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
MODEL_DIR = load_key("model_dir")


def separate_vocals(input_file=None, output_file=None, model_name=None):
    """
    分离人声：从原始音频中提取人声轨道

    Args:
        input_file: 输入音频文件路径（默认使用 config.yaml 中的 RAW_AUDIO_FILE）
        output_file: 输出人声文件路径（默认使用 output/audio/vocals.wav）
        model_name: 分离模型名称（默认 UVR-MDX-NET-Inst_HQ_3.onnx）

    Returns:
        bool: 成功返回 True，失败返回 False
    """
    # 1. 参数默认值
    if input_file is None:
        input_file = _RAW_AUDIO_FILE
    if output_file is None:
        output_file = _VOCAL_AUDIO_FILE
    if model_name is None:
        model_name = VOCAL_MODEL

    # 2. 检查文件状态
    if not os.path.exists(input_file):
        console.print(f"[red]输入文件不存在: {input_file}[/red]")
        return False

    if os.path.exists(output_file):
        console.print(f"[yellow]人声文件已存在，跳过分离: {output_file}[/yellow]")
        return True

    # 3. 准备路径
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    console.print(f"[cyan]开始分离人声 (模型: {model_name})...[/cyan]")

    try:
        # 初始化 Separator
        separator = Separator(
            model_file_dir=MODEL_DIR,
            output_dir=output_dir,
            output_format="wav",
            # 只输出人声轨道，减少IO和计算时间
            output_single_stem="Vocals",
            log_level=40 
        )

        separator.load_model(model_filename=model_name)

        # 执行分离
        output_files = separator.separate(input_file)

        console.print(f"[dim]原始分离文件生成: {output_files}[/dim]")

        # 4. 查找并重命名文件
        generated_file = None

        if output_files and isinstance(output_files, list):
            for f in output_files:
                # 查找包含 Vocals 的文件
                if "Vocals" in f or "vocals" in f or "Vocal" in f:
                    generated_file = os.path.join(output_dir, f)
                    break

            # 如果没找到带 Vocals 的，但列表里只有一个文件
            if not generated_file and len(output_files) == 1:
                generated_file = os.path.join(output_dir, output_files[0])

        # 5. 执行重命名操作
        if generated_file and os.path.exists(generated_file):
            if os.path.exists(output_file):
                os.remove(output_file)

            os.rename(generated_file, output_file)
            console.print(f"[green]人声已保存到: {output_file}[/green]")
            return True
        else:
            console.print(f"[red]未能找到生成的人声文件: {output_files}[/red]")
            return False

    except Exception as e:
        console.print(f"[red]音频人声分离失败: {e}[/red]")
        return False


if __name__ == "__main__":
    separate_vocals()
