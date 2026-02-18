"""强制对齐模块 - Qwen3ForcedAligner"""
import os
import logging
from typing import List, Any, Callable, Optional, Tuple
import torch
import numpy as np
import io
from pydub import AudioSegment
from qwen_asr import Qwen3ForcedAligner

# 抑制 transformers 的冗余日志
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logger = logging.getLogger(__name__)

# 模型路径（写死）
ALIGNER_MODEL_PATH = "models/Qwen3-ForcedAligner-0.6B"


class Aligner:
    """强制对齐器"""

    def __init__(
        self,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            device: 设备（None 表示自动检测）
            dtype: 数据类型
        """
        # 自动检测设备
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_path = ALIGNER_MODEL_PATH
        self.device = device
        self.dtype = dtype
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            logger.info(f"正在加载强制对齐模型: {self.model_path}")
            logger.info(f"使用设备: {self.device}")
            self.model = Qwen3ForcedAligner.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                device_map=self.device,
            )
            logger.info("强制对齐模型加载成功")
        except Exception as e:
            logger.error(f"强制对齐模型加载失败: {e}")
            raise

    def unload_model(self):
        """卸载模型，释放内存"""
        if self.model is not None:
            logger.info("正在卸载强制对齐模型...")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("强制对齐模型已卸载")

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

    def align_bytes(
        self,
        audio_bytes: bytes,
        text: str,
        language: str,
        segment_idx: int = 0,
        total_segments: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """
        对齐单个音频分段（内存版本，接受 bytes）

        Args:
            audio_bytes: 音频数据 (MP3/WAV 格式 bytes)
            text: 对应的文本
            language: 语言代码（英文全称，如 "Chinese"）
            segment_idx: 当前分段索引
            total_segments: 总分段数
            progress_callback: 进度回调函数

        Returns:
            对齐结果列表，每个元素包含 text, start_time, end_time
        """
        # 转换为 (wav_array, sr) 格式
        wav_array, sr = self._convert_bytes_to_wav_array(audio_bytes)

        # 清理文本：移除空行，合并成单行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = " ".join(lines)

        results = self.model.align(
            audio=(wav_array, sr),
            text=clean_text,
            language=language,
        )

        if results and len(results) > 0:
            align_result = results[0]
            if progress_callback:
                progress_callback(segment_idx + 1, total_segments)
            logger.info(f"分段 {segment_idx + 1}/{total_segments} 对齐成功 ({len(align_result)} 个词)")
            return align_result
        else:
            logger.warning(f"分段 {segment_idx + 1} 对齐返回空结果")
            return []

    def align_bytes_list(
        self,
        audio_bytes_list: List[bytes],
        transcript_segments: List[str],
        language: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[Any]]:
        """
        对齐所有音频分段（内存版本，接受 bytes 列表）

        Args:
            audio_bytes_list: 音频分段 bytes 列表 (MP3/WAV 格式)
            transcript_segments: 转录文本列表（与音频分段一一对应）
            language: 语言代码（英文全称）
            progress_callback: 进度回调函数

        Returns:
            对齐结果列表（每个元素是一个分段的对齐结果）
        """
        if len(audio_bytes_list) != len(transcript_segments):
            raise ValueError(f"音频分段数 ({len(audio_bytes_list)}) 与文本分段数 ({len(transcript_segments)}) 不匹配")

        results = []

        for idx, (audio_bytes, text) in enumerate(zip(audio_bytes_list, transcript_segments)):
            align_result = self.align_bytes(
                audio_bytes,
                text,
                language,
                idx,
                len(audio_bytes_list),
                progress_callback,
            )
            results.append(align_result)

        return results
