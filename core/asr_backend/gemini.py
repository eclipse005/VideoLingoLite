import os
import json
import requests
import time
from typing import Optional
import io
import librosa
import soundfile as sf
from rich import print as rprint

OUTPUT_LOG_DIR = "output/log"

def convert_gemini_response(word_response: list, start_time_offset: Optional[float] = None) -> dict:
    segments = []
    current_segment = None
    
    for word in word_response:
        # Apply time offset if provided
        start = word['start'] + start_time_offset if start_time_offset else word['start']
        end = word['end'] + start_time_offset if start_time_offset else word['end']
        # Create or extend segment
        if not current_segment:
            current_segment = {
                'words': []
            }            
        current_segment['words'].append({
            'word': word['word'],
            'start': start,
            'end': end
        })
    if current_segment:
        segments.append(current_segment)
    return {
        "segments": segments
    }

def transcribe_audio_gemini(raw_audio_path: str, audio_path: str, start: float = None, end: float = None):
    start_time = time.time()
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    if not os.path.exists(audio_path):
        return None
    LOG_FILE = f"{OUTPUT_LOG_DIR}/gemini_{start}_{end}.json" if start and end else f"{OUTPUT_LOG_DIR}/gemini.json"
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # 始终用 librosa 切片并写入 buffer
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
             "filename": (None, os.path.basename(raw_audio_path))}
    response = requests.post("http://localhost:5000/transcribe", headers=headers, files=files, proxies={'http': None, 'https': None})
    if response.status_code != 200:
        raise Exception(f"Transcription failed with status {response.status_code}: {response.text}")
    word_response = response.json()
    result = convert_gemini_response(word_response, start)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    elapsed_time = time.time() - start_time
    rprint(f"[green]✓ Transcription completed in {elapsed_time:.2f} seconds[/green]")
    return result

if __name__ == "__main__":
    print(transcribe_audio_302("output/audio/raw.mp3"))
