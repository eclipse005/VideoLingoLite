from core.utils import *
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, split_audio_by_vad, save_results
from core._1_ytdlp import find_video_files
from core.utils.models import *
import json
import os
import pandas as pd
from typing import List

@timer("ASR 转录")
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

    # 3. Extract audio segments (VAD or FFMPEG silence detection)
    if load_key("vad.enabled", default=False):
        segments = split_audio_by_vad(vocal_audio)
    else:
        segments = split_audio(vocal_audio)

    # 4. Select ASR backend (only qwen in this branch)
    from core.asr_backend.qwen3_asr import transcribe_batch
    model = load_key("asr.model", default="Qwen3-ASR-0.6B")
    rprint(f"[cyan]🎤 Transcribing audio with {model}...[/cyan]")

    # 5. Batch transcribe (load model once, process all segments)
    rprint(f"[cyan]📋 Processing {len(segments)} audio segments...[/cyan]")
    all_results = transcribe_batch(vocal_audio, segments)
    rprint(f"[green]✅ Received {len(all_results)} transcription results[/green]")

    # 6. Combine results (keep multiple segments) - use aligned version for processing
    combined_result = {'segments': []}
    combined_raw_result = {'segments': []}  # Raw version for asr.json
    segment_count = 0
    for aligned, raw in all_results:
        for segment in aligned.get('segments', []):
            combined_result['segments'].append(segment)
            segment_count += 1
        for segment in raw.get('segments', []):
            combined_raw_result['segments'].append(segment)
    rprint(f"[cyan]📊 Total segments in combined_result: {segment_count}[/cyan]")
    combined_result['language'] = all_results[0][0].get('language', 'unknown') if all_results else 'unknown'
    combined_raw_result['language'] = all_results[0][1].get('language', 'unknown') if all_results else 'unknown'

    # 7. Save ASR result to JSON (use raw version for debugging)
    asr_json_path = "output/log/asr.json"
    os.makedirs(os.path.dirname(asr_json_path), exist_ok=True)
    with open(asr_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_raw_result, f, indent=2, ensure_ascii=False)
    rprint(f"[green]💾 ASR result saved to: {asr_json_path}[/green]")

    # 8. Process df (use aligned version for processing)
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