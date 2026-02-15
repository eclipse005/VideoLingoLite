import os
import json
import requests
import time
import unicodedata
from typing import Optional, List, Dict
import io
import librosa
import soundfile as sf
from rich import print as rprint

OUTPUT_LOG_DIR = "output/log"

def is_sentence_terminator(char: str) -> bool:
    """
    判断字符是否为句子结束符号（使用 Unicode 类别）。

    涵盖多语言：
    - 中文/日文：。！？
    - 英文：.!?
    - 其他语言的句子结束符号
    """
    if not char:
        return False

    # 常见句子结束符号（作为补充）
    terminators = {'.', '!', '?', '。', '！', '？', '‼', '⁇', '⁈', '⁉'}
    if char in terminators:
        return True

    # 使用 Unicode 类别判断
    # Po: Other punctuation（包含大多数标点符号）
    category = unicodedata.category(char)
    if category == 'Po':
        # 进一步过滤，只保留句子结束类的标点
        # 排除逗号、顿号、引号等非句子结束符号
        non_terminators = {',', '，', '、', ';', '；', ':', '：', '"', "'", '「', '」', '『', '』', '（', '）', '(', ')', '[', ']', '{', '}', '・', '·', '•'}
        if char not in non_terminators:
            return True

    return False


def group_words_into_sentences(word_response: List[Dict]) -> List[List[Dict]]:
    """
    根据句子结束符号将 words 分组为句子。

    Args:
        word_response: [{'word': str, 'start': float, 'end': float}, ...]

    Returns:
        List of sentence groups, each group is a list of words
    """
    if not word_response:
        return []

    sentences = []
    current_sentence = []

    for word_info in word_response:
        word = word_info['word']
        current_sentence.append(word_info)

        # 检查该词是否包含句子结束符号
        if word:
            # 检查词的最后一个字符（或任意字符）是否为句子结束符
            for char in word:
                if is_sentence_terminator(char):
                    sentences.append(current_sentence)
                    current_sentence = []
                    break

    # 处理剩余的词
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def convert_custom_response(word_response: list, start_time_offset: Optional[float] = None) -> dict:
    """
    Convert custom ASR API response to standard format.

    Args:
        word_response: List of word-level timestamps from custom ASR API
        start_time_offset: Optional time offset for chunked audio

    Returns:
        Dictionary with segments containing words with timestamps
    """
    # 根据标点符号将 words 分组为句子
    sentence_groups = group_words_into_sentences(word_response)

    segments = []

    for sentence_words in sentence_groups:
        if not sentence_words:
            continue

        # 应用时间偏移
        words_with_offset = []
        for word in sentence_words:
            start = word['start'] + start_time_offset if start_time_offset else word['start']
            end = word['end'] + start_time_offset if start_time_offset else word['end']
            words_with_offset.append({
                'word': word['word'],
                'start': start,
                'end': end
            })

        # 构建 segment（带 text, start, end，与 Parakeet 格式一致）
        seg_start = words_with_offset[0]['start']
        seg_end = words_with_offset[-1]['end']
        seg_text = ''.join([w['word'] for w in words_with_offset])

        segments.append({
            'start': seg_start,
            'end': seg_end,
            'text': seg_text,
            'words': words_with_offset
        })

    return {
        "segments": segments
    }

def transcribe_audio_custom(raw_audio_path: str, audio_path: str, start: float = None, end: float = None):
    """
    Transcribe audio using custom ASR API.

    This function sends audio to a user-defined custom ASR API endpoint.
    The API should accept:
        - audio: WAV file (16kHz, 16-bit PCM)
        - start: Start time in seconds (optional)
        - end: End time in seconds (optional)
        - filename: Original audio filename

    And return a JSON array of word-level timestamps:
        [{"word": "text", "start": 0.0, "end": 0.5}, ...]

    Default endpoint: http://localhost:5000/transcribe
    Configure by modifying the URL in this file.

    Args:
        raw_audio_path: Path to raw audio file (unused, for compatibility)
        audio_path: Path to input audio file
        start: Start time in seconds (optional)
        end: End time in seconds (optional)

    Returns:
        Dictionary with transcription results

    Raises:
        Exception: If transcription API returns non-200 status
    """
    start_time = time.time()
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    if not os.path.exists(audio_path):
        return None
    LOG_FILE = f"{OUTPUT_LOG_DIR}/custom_{start}_{end}.json" if start and end else f"{OUTPUT_LOG_DIR}/custom.json"
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Always use librosa for audio slicing and write to buffer
    y, sr = librosa.load(audio_path, sr=16000)
    audio_duration = len(y) / sr
    if start is None or end is None:
        start = 0
        end = audio_duration
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_slice = y[start_sample:end_sample]
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, y_slice, sr, format='WAV', subtype='PCM_16')
    audio_buffer.seek(0)
    headers = {
        "accept": "application/json"
    }
    files = {"audio": (os.path.basename(audio_path), audio_buffer.getvalue(), "audio/wav"),
             "start": (None, str(start) if start is not None else ""),
             "end": (None, str(end) if end is not None else ""),
             "filename": (None, os.path.basename(audio_path))}
    # TODO: Make endpoint URL configurable via config.yaml
    response = requests.post("http://localhost:5000/transcribe", headers=headers, files=files, proxies={'http': None, 'https': None})
    if response.status_code != 200:
        raise Exception(f"Transcription failed with status {response.status_code}: {response.text}")
    word_response = response.json()
    result = convert_custom_response(word_response, start)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    elapsed_time = time.time() - start_time
    rprint(f"[green]✓ Transcription completed in {elapsed_time:.2f} seconds[/green]")
    return result

if __name__ == "__main__":
    print(transcribe_audio_custom("output/audio/raw.wav"))
