import os
import sys
import logging
import warnings
import io

# ------------
# Suppress all warnings (MUST be before any library imports)
# ------------

# 1. Suppress Python warnings
warnings.filterwarnings("ignore")

# 2. Set root logging level to ERROR before any imports
logging.basicConfig(level=logging.ERROR)

# 3. Suppress specific NeMo/PyTorch loggers
for logger_name in [
    'nemo',
    'nemo.utils',
    'nemo.collections.asr',
    'megatron',
    'torch.distributed.elastic',
    'lhotse',
    'one_logger',
    'nv_one_logger',
    'nv_one_logger.api.config',
    'nv_one_logger.exporter.export_config_manager',
    'nv_one_logger.training_telemetry.api.training_telemetry_provider',
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

# 4. Disable NeMo telemetry and other features via environment variables
os.environ.setdefault('NEMO_LOGGING_LEVEL', 'ERROR')
os.environ.setdefault('NEMO_DISABLE_TENSORBOARD', '1')

# 5. Redirect stderr during imports to suppress NeMo warnings
_original_stderr = sys.stderr

class _SuppressStdErr:
    """Redirect stderr to suppress NeMo warnings during import"""
    def write(self, text):
        # Suppress NeMo warnings, PyTorch distributed warnings, OneLogger messages
        if any(x in text for x in ['[NeMo W', 'W0121', 'OneLogger', 'redirects.py']):
            return
        # Write to original stderr if not suppressed
        _original_stderr.write(text)

    def flush(self):
        _original_stderr.flush()

    def close(self):
        """No-op close method to prevent errors on exit"""
        pass

sys.stderr = _SuppressStdErr()

# Now safe to import other libraries (warnings will be suppressed)
import torch
import librosa
import gc
from rich import print as rprint
from core.utils import *
import nemo.collections.asr as nemo_asr

# Restore stderr after imports
sys.stderr = _original_stderr

# Ensure NeMo logging is silenced
from nemo.utils import logging as nemo_logging
nemo_logging.setLevel(logging.ERROR)

# 1. 路径配置
MODEL_DIR = load_key("model_dir")
ABS_MODEL_DIR = os.path.abspath(MODEL_DIR)

# 全局模型缓存
_model_instance = None
_model_device = None

def _load_or_download_model():
    """纯本地加载 Parakeet 模型（带缓存）"""
    global _model_instance, _model_device

    # 如果模型已加载，直接返回
    if _model_instance is not None:
        return _model_instance, _model_device

    # 明确指向你手动存放的文件
    LOCAL_MODEL_PATH = os.path.join(ABS_MODEL_DIR, "parakeet-tdt-0.6b-v2.nemo")

    if not os.path.exists(LOCAL_MODEL_PATH):
        rprint(f"[red]❌ Model file not found at: {LOCAL_MODEL_PATH}[/red]")
        raise FileNotFoundError(f"Please ensure 'parakeet-tdt-0.6b-v2.nemo' is in {ABS_MODEL_DIR}")

    rprint(f"[cyan]📥 Loading Parakeet model from project cache...[/cyan]")

    try:
        # 纯本地加载，不涉及任何联网校验
        model = nemo_asr.models.ASRModel.restore_from(restore_path=LOCAL_MODEL_PATH)
        rprint(f"[green]✅ Model loaded successfully.[/green]")
        _model_instance = model
        return model, None  # 返回模型和设备（设备尚未设置）
    except Exception as e:
        rprint(f"[red]❌ Failed to load local model: {e}[/red]")
        raise RuntimeError(f"Local model file at {LOCAL_MODEL_PATH} is corrupted.")

def _setup_model_device(model):
    """设置模型设备和配置"""
    # 自动判断设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        rprint(f"[bold green]🚀 GPU Detected: {device_name} | Mode: CUDA Acceleration[/bold green]")
    else:
        device = torch.device("cpu")
        rprint(f"[bold yellow]⚠️ GPU Not Found. Switching to CPU Mode.[/bold yellow]")
        rprint(f"[dim]Running on CPU may take longer. Performance depends on your CPU power.[/dim]")

    model = model.to(device)
    model.eval()

    from omegaconf import open_dict
    if hasattr(model, 'change_decoding_strategy'):
        cfg = model.cfg.decoding
        with open_dict(cfg):
            cfg.cuda_graphs = False
            if "strategy" in cfg:
                cfg.strategy = "greedy"
        model.change_decoding_strategy(cfg)

    return device

def release_model():
    """释放模型和GPU资源"""
    global _model_instance, _model_device

    if _model_instance is not None:
        del _model_instance
        _model_instance = None
        _model_device = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rprint("[dim]♻️ Model resources released.[/dim]")

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
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end, model=None):
    """
    Parakeet 转录主函数 (使用已加载的模型)

    Args:
        raw_audio_file: 原始音频文件
        vocal_audio_file: 人声分离后的音频文件
        start: 开始时间
        end: 结束时间
        model: 已加载的模型实例（可选）
    """

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ------------------------------------

    # 1. 加载或使用传入的模型
    if model is None:
        model, _ = _load_or_download_model()
        device = _setup_model_device(model)
        should_release = True
    else:
        # 使用传入的模型（已配置好）
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        should_release = False

    audio_length = end - start

    # 2. 长音频优化
    if audio_length > 120:
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
        # 抑制 NeMo/Lhotse 的运行时日志
        for logger_name in ['lhotse', 'nemo', 'root']:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        rprint(f"[bold green]Transcribing on {str(device).upper()}...[/bold green]")
        with torch.no_grad():
            output = model.transcribe([tmp_path], timestamps=True)
        return _convert_parakeet_to_standard_asr_format(output, start_offset=start)

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # 只在函数内部加载模型时才释放
        if should_release:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rprint("[dim]♻️ Resources released.[/dim]")
        # ---------------------------