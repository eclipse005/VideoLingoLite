from core.utils import *
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results
from core._1_ytdlp import find_video_files
from core.utils.models import *
import json
import os
import pandas as pd
from typing import List

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Optional vocal separation for noisy environments
    if load_key("vocal_separation.enabled"):
        from core.utils.vocal_separator import separate_vocals
        rprint("[cyan]🎤 Separating vocals from audio...[/cyan]")
        if separate_vocals():
            vocal_audio = _VOCAL_AUDIO_FILE
            rprint("[green]✅ Vocal separation completed, using vocals for transcription[/green]")
        else:
            rprint("[yellow]⚠️ Vocal separation failed, falling back to original audio[/yellow]")
            vocal_audio = _RAW_AUDIO_FILE
    else:
        vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(vocal_audio)

    # 4. Select ASR backend based on config
    asr_runtime = load_key("asr.runtime")
    if asr_runtime == "gemini":
        from core.asr_backend.gemini import transcribe_audio_gemini as ts
        rprint("[cyan]🎤 Transcribing audio with Gemini API...[/cyan]")
    elif asr_runtime == "qwen":
        from core.asr_backend.qwen3_asr import transcribe_audio as ts
        model = load_key("asr.model", default="Qwen3-ASR-1.7B")
        rprint(f"[cyan]🎤 Transcribing audio with {model}...[/cyan]")
    elif asr_runtime == "parakeet":
        from core.asr_backend.parakeet_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with NVIDIA Parakeet...[/cyan]")
    else:
        raise ValueError(f"Unsupported ASR runtime: {asr_runtime}")

    # 5. Transcribe audio by clips
    all_results = []
    for start, end in segments:
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        all_results.append(result)

    # 6. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])

    # 7. Save ASR result to JSON
    asr_json_path = "output/log/asr.json"
    os.makedirs(os.path.dirname(asr_json_path), exist_ok=True)
    with open(asr_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_result, f, indent=2, ensure_ascii=False)
    rprint(f"[green]💾 ASR result saved to: {asr_json_path}[/green]")

    # 8. Process df (always generate cleaned_chunks.csv for word-level data)
    df = process_transcription(combined_result)
    save_results(df)


def load_chunks() -> List[Chunk]:
    """
    从 cleaned_chunks.csv 加载 Chunk 对象列表

    Returns:
        List[Chunk]: 词/字级别的 Chunk 对象列表
    """
    df = safe_read_csv(_2_CLEANED_CHUNKS)
    chunks = []

    for row in df.itertuples(index=True):
        speaker_id = row.speaker_id if pd.notna(row.speaker_id) and row.speaker_id else None
        chunk = Chunk(
            text=row.text.strip('"'),
            start=float(row.start),
            end=float(row.end),
            speaker_id=speaker_id,
            index=row.Index
        )
        chunks.append(chunk)

    rprint(f"[green]✅ Loaded {len(chunks)} chunks from {_2_CLEANED_CHUNKS}[/green]")
    return chunks

if __name__ == "__main__":
    transcribe()