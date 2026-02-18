"""Qwen 转录模块"""
import os
import torch
import logging
import io
import numpy as np
from typing import List, Optional, Callable, Tuple
from pydub import AudioSegment

# 抑制 transformers 的冗余日志
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None

logger = logging.getLogger(__name__)

# 模型路径（写死）
ASR_MODEL_PATH = "models/Qwen3-ASR-0.6B"


class QwenTranscriber:
    """Qwen 音频转录器"""

    def __init__(self):
        """
        初始化转录器
        """
        if Qwen3ASRModel is None:
            raise ImportError(
                "请先安装 qwen_asr: pip install qwen-asr"
            )

        # 检测设备
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info(f"正在加载 Qwen 模型: {ASR_MODEL_PATH}")
        logger.info(f"使用设备: {device}")
        self.model = Qwen3ASRModel.from_pretrained(
            ASR_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=4096,
        )
        logger.info("Qwen 模型加载完成")

    def unload_model(self):
        """卸载模型，释放内存"""
        if hasattr(self, 'model') and self.model is not None:
            logger.info("正在卸载 Qwen 转录模型...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen 转录模型已卸载")

    def _convert_bytes_to_wav_array(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        将音频 bytes 转换为 (wav_array, sample_rate) 格式

        Args:
            audio_bytes: 音频数据 (MP3/WAV 格式 bytes)

        Returns:
            (numpy.ndarray, int): (音频数组, 采样率)
        """
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # 转换为 16kHz 单声道
        wav = audio.set_frame_rate(16000).set_channels(1)
        wav_array = np.array(wav.get_array_of_samples(), dtype=np.float32) / 32768
        return wav_array, 16000

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        progress_callback: Callable[[int, int], None] = None,
        segment_idx: int = 0,
        total_segments: int = 1,
    ) -> str:
        """
        转录单个音频分段（内存版本，直接接受 bytes）

        Args:
            audio_bytes: 音频数据 (MP3/WAV 格式 bytes)
            progress_callback: 进度回调函数 (current, total)
            segment_idx: 当前分段索引
            total_segments: 总分段数

        Returns:
            转录文本（如果为空返回空字符串）
        """
        # 转换为 (wav_array, sr) 格式
        wav_array, sr = self._convert_bytes_to_wav_array(audio_bytes)

        # 调用 Qwen 模型进行转录
        results = self.model.transcribe(
            audio=(wav_array, sr),
            language=None,  # 自动检测语言
            return_time_stamps=False,
        )

        if results and len(results) > 0:
            text = results[0].text
        else:
            text = ""

        # 更新进度
        if progress_callback:
            progress_callback(segment_idx + 1, total_segments)

        return text

    def transcribe_bytes_list(
        self,
        audio_bytes_list: List[bytes],
        progress_callback: Callable[[int, int], None] = None,
    ) -> List[str]:
        """
        转录所有音频分段（内存版本，接受 bytes 列表）

        Args:
            audio_bytes_list: 音频分段 bytes 列表 (MP3/WAV 格式)
            progress_callback: 进度回调函数

        Returns:
            转录文本列表（每个元素对应一个分段的转录文本）
        """
        results = []

        for idx, audio_bytes in enumerate(audio_bytes_list):
            text = self.transcribe_bytes(
                audio_bytes,
                progress_callback,
                idx,
                len(audio_bytes_list),
            )
            results.append(text)

        return results
