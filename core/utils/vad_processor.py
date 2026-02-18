"""
VAD (语音活动检测) 处理器 - 使用 Ten VAD
"""

import numpy as np
from ten_vad import TenVad
import librosa

# 通用 VAD 参数
DEFAULT_THRESHOLD = 0.4       # 语音概率阈值
DEFAULT_MIN_SPEECH_MS = 150   # 最小语音时长（毫秒）
DEFAULT_MIN_SILENCE_MS = 200  # 静音多久结束语音段（毫秒）
DEFAULT_MERGE_GAP_MS = 150    # 合并间隔小于此值的相邻片段（毫秒）


def get_speech_segments(
    wav_file,
    start_sec=None,
    end_sec=None,
    threshold=DEFAULT_THRESHOLD,
    min_silence_ms=DEFAULT_MIN_SILENCE_MS,
    min_speech_ms=DEFAULT_MIN_SPEECH_MS,
    merge_gap_ms=DEFAULT_MERGE_GAP_MS,
    max_segment_duration=None,
):
    """
    利用 Ten VAD 检测语音片段

    参数：
      - wav_file: 音频文件路径
      - start_sec: 检测起始时间（秒），可选
      - end_sec: 检测结束时间（秒），可选
      - threshold: 语音概率阈值（0-1），默认0.4
      - min_silence_ms: 静音多久结束语音段（毫秒），默认200
      - min_speech_ms: 语音段最短时长（毫秒），默认150
      - merge_gap_ms: 合并间隔小于此值的相邻片段（毫秒），默认150
      - max_segment_duration: 合并后的最大时长（秒），超过此值不再合并

    返回一个列表，每项为 (start, end) 单位秒。
    """
    # 读取音频（librosa 自动处理采样率、单声道、浮点格式）
    data, sr = librosa.load(wav_file, sr=16000, mono=True)

    # 转换为 int16（Ten VAD 要求）
    data = (data * 32767).astype(np.int16)

    # 时间窗口裁剪
    if start_sec is not None or end_sec is not None:
        total_len = len(data)
        s = int((start_sec or 0) * sr)
        e = int((end_sec * sr) if end_sec is not None else total_len)
        if s >= e:
            s = max(0, e - sr)
        data = data[s:e]
        offset = start_sec or 0
    else:
        offset = 0

    # VAD 参数
    hop_size = 256
    hop_dur = hop_size / sr
    vad = TenVad(hop_size=hop_size, threshold=threshold)

    # 状态机
    in_speech = False
    speech_start = 0
    silence_dur = 0.0
    segments = []

    # 逐帧处理
    for start in range(0, len(data) - hop_size + 1, hop_size):
        frame = data[start : start + hop_size]
        _, flag = vad.process(frame)
        is_voice = flag == 1

        if is_voice:
            if not in_speech:
                speech_start = start
                in_speech = True
            silence_dur = 0.0
        else:
            if in_speech:
                silence_dur += hop_dur
                if silence_dur * 1000 >= min_silence_ms:
                    end = start
                    dur_ms = (end - speech_start) / sr * 1000
                    if dur_ms >= min_speech_ms:
                        segments.append((speech_start / sr, end / sr))
                    in_speech = False
                    silence_dur = 0.0

    # 处理最后一段
    if in_speech:
        end = len(data)
        dur_ms = (end - speech_start) / sr * 1000
        if dur_ms >= min_speech_ms:
            segments.append((speech_start / sr, end / sr))

    # 合并接近的片段
    merged = []
    for start_s, end_s in segments:
        if not merged or start_s - merged[-1][1] > merge_gap_ms / 1000:
            merged.append([start_s, end_s])
        else:
            # 检查合并后是否会超过 max_segment_duration
            if max_segment_duration is None or end_s - merged[-1][0] <= max_segment_duration:
                merged[-1][1] = end_s
            else:
                # 超过限制，不合并，开始新片段
                merged.append([start_s, end_s])

    # 加上时间偏移并转为元组
    result = [(s + offset, e + offset) for s, e in merged]

    return result
