import os
import warnings
import time
import torch
import librosa
from rich import print as rprint
from core.utils import *
import nemo.collections.asr as nemo_asr

warnings.filterwarnings("ignore")

# 1. 路径配置
MODEL_DIR = load_key("model_dir") 
ABS_MODEL_DIR = os.path.abspath(MODEL_DIR)

def _load_or_download_model():
    """纯本地加载 Parakeet 模型"""
    # 明确指向你手动存放的文件
    LOCAL_MODEL_PATH = os.path.join(ABS_MODEL_DIR, "parakeet-tdt-0.6b-v3.nemo")

    rprint(f"[cyan]📥 Loading Parakeet model from project cache...[/cyan]")

    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            # 纯本地加载，不涉及任何联网校验
            model = nemo_asr.models.ASRModel.restore_from(restore_path=LOCAL_MODEL_PATH)
            rprint(f"[green]✅ Model loaded successfully.[/green]")
            return model
        except Exception as e:
            rprint(f"[red]❌ Failed to load local model: {e}[/red]")
            raise RuntimeError(f"Local model file at {LOCAL_MODEL_PATH} is corrupted.")
    else:
        rprint(f"[red]❌ Model file not found at: {LOCAL_MODEL_PATH}[/red]")
        raise FileNotFoundError(f"Please ensure 'parakeet-tdt-0.6b-v3.nemo' is in {ABS_MODEL_DIR}")

def _convert_parakeet_to_standard_asr_format(output, start_offset=0):
    """将 Parakeet 输出转换为标准 ASR 格式"""
    if not output or len(output) == 0:
        return {'segments': [], 'language': 'unknown'}

    result = output[0]
    segment_timestamps = result.timestamp.get('segment', [])
    word_timestamps = result.timestamp.get('word', [])

    segments = []
    word_idx = 0
    for seg_stamp in segment_timestamps:
        seg_start = round(seg_stamp['start'] + start_offset, 2)
        seg_end = round(seg_stamp['end'] + start_offset, 2)
        seg_text = seg_stamp['segment']

        words = []
        while word_idx < len(word_timestamps):
            word = word_timestamps[word_idx]
            word_start = round(word['start'] + start_offset, 2)
            word_end = round(word['end'] + start_offset, 2)
            if word_start > seg_end:
                break
            words.append({'start': word_start, 'end': word_end, 'word': word['word']})
            word_idx += 1
        segments.append({'start': seg_start, 'end': seg_end, 'text': seg_text, 'words': words})

    return {'segments': segments, 'language': 'en'}

@except_handler("Parakeet processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    """主转录函数"""
    if not torch.cuda.is_available():
        raise RuntimeError("Parakeet requires NVIDIA GPU.")

    # 1. 加载本地模型
    model = _load_or_download_model()
    audio_length = end - start

    # 2. 长音频优化（超过 2 分钟使用局部注意力）
    if audio_length > 120:
        rprint(f"[yellow]⚠️ Long audio (>2min), enabling local attention...[/yellow]")
        model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])

    # 3. 加载音频片段
    rprint(f"[cyan]🎵 Loading audio segment...[/cyan]")
    audio, sr = librosa.load(raw_audio_file, sr=16000, offset=start, duration=audio_length, mono=True)

    # 4. 执行转录
    import tempfile
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        sf.write(tmp_path, audio, sr)

    try:
        rprint("[bold green]Transcribing...[/bold green]")
        output = model.transcribe([tmp_path], timestamps=True)
        return _convert_parakeet_to_standard_asr_format(output, start_offset=start)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        del model
        torch.cuda.empty_cache()