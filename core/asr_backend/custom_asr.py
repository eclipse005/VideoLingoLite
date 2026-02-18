import os
import json
import requests
import time
from typing import Optional, List, Dict
import io
import librosa
import soundfile as sf
from rich import print as rprint

OUTPUT_LOG_DIR = "output/log"


def convert_custom_response(word_response: list, start_time_offset: Optional[float] = None) -> dict:
    """
    Convert custom ASR API response to standard format.

    Args:
        word_response: List of word-level timestamps from custom ASR API
        start_time_offset: Optional time offset for chunked audio

    Returns:
        Dictionary with single segment containing all words with timestamps
    """
    if not word_response:
        return {"segments": []}

    # 应用时间偏移
    words_with_offset = []
    for word in word_response:
        start = word['start'] + start_time_offset if start_time_offset else word['start']
        end = word['end'] + start_time_offset if start_time_offset else word['end']
        words_with_offset.append({
            'word': word['word'],
            'start': start,
            'end': end
        })

    # 构建单个 segment（与 qwen3_asr.py 格式一致）
    # 句子分割由 _3_1_split_by_punctuation.py 处理
    seg_start = words_with_offset[0]['start']
    seg_end = words_with_offset[-1]['end']
    seg_text = ''.join([w['word'] for w in words_with_offset])

    return {
        "segments": [{
            'start': seg_start,
            'end': seg_end,
            'text': seg_text,
            'words': words_with_offset
        }]
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
