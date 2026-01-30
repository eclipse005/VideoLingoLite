import os, subprocess
import pandas as pd
from typing import Dict, List, Tuple
from pydub import AudioSegment
from core.utils import *
from core.utils.models import *
from pydub.silence import detect_silence
from pydub.utils import mediainfo
from rich import print as rprint

def convert_video_to_audio(video_file: str):
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    if not os.path.exists(_RAW_AUDIO_FILE):
        rprint(f"[blue]🎬➡️🎵 Converting to high quality audio with FFmpeg ......[/blue]")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn',
            '-c:a', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-metadata', 'encoding=UTF-8', _RAW_AUDIO_FILE
        ], check=True, stderr=subprocess.PIPE)
        rprint(f"[green]🎬➡️🎵 Converted <{video_file}> to <{_RAW_AUDIO_FILE}> with FFmpeg\n[/green]")

def split_audio(audio_file: str, target_len: float = 2*60, win: float = 30) -> List[Tuple[float, float]]:
    ## 在 [target_len-win, target_len] 区间内向左查找静默点，切分音频（避免向右找导致音频过长爆显存）
    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    audio = AudioSegment.from_file(audio_file)
    duration = float(mediainfo(audio_file)["duration"])
    if duration <= target_len:
        return [(0, duration)]
    segments, pos = [], 0.0
    safe_margin = 0.5  # 静默点前后安全边界，单位秒

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration)); break

        threshold = pos + target_len
        # 只向左找：在 [threshold-win, threshold] 范围内查找静音点
        ws, we = int((threshold - win) * 1000), int(threshold * 1000)

        # 获取完整的静默区域
        silence_regions = detect_silence(audio[ws:we], min_silence_len=int(safe_margin*1000), silence_thresh=-30)
        silence_regions = [(s/1000 + (threshold - win), e/1000 + (threshold - win)) for s, e in silence_regions]
        # 筛选长度足够（至少1秒）且位置适合的静默区域
        valid_regions = [
            (start, end) for start, end in silence_regions
            if (end - start) >= (safe_margin * 2) and (threshold - win) <= start + safe_margin <= threshold
        ]

        if valid_regions:
            # 选择最右侧的静默区域（最接近 threshold）
            start, end = valid_regions[-1]
            split_at = start + safe_margin  # 在静默区域起始点后0.5秒处切分
        else:
            rprint(f"[yellow]⚠️ No valid silence regions found for {audio_file} at {threshold}s, using threshold[/yellow]")
            split_at = threshold

        segments.append((pos, split_at)); pos = split_at

    rprint(f"[green]🎙️ Audio split completed {len(segments)} segments[/green]")
    return segments

def process_transcription(result: Dict) -> pd.DataFrame:
    all_words = []
    for segment in result['segments']:
        # Get speaker_id, if not exists, set to None
        speaker_id = segment.get('speaker_id', None)
        
        for word in segment['words']:
            # Check word length
            if len(word["word"]) > 30:
                rprint(f"[yellow]⚠️ Warning: Detected word longer than 30 characters, skipping: {word['word']}[/yellow]")
                continue
                
            # ! For French, we need to convert guillemets to empty strings
            word["word"] = word["word"].replace('»', '').replace('«', '')
            
            if 'start' not in word and 'end' not in word:
                if all_words:
                    # Assign the end time of the previous word as the start and end time of the current word
                    word_dict = {
                        'text': word["word"],
                        'start': round(all_words[-1]['end'], 2),
                        'end': round(all_words[-1]['end'], 2),
                        'speaker_id': speaker_id
                    }
                    all_words.append(word_dict)
                else:
                    # If it's the first word, look next for a timestamp then assign it to the current word
                    next_word = next((w for w in segment['words'] if 'start' in w and 'end' in w), None)
                    if next_word:
                        word_dict = {
                            'text': word["word"],
                            'start': round(next_word["start"], 2),
                            'end': round(next_word["end"], 2),
                            'speaker_id': speaker_id
                        }
                        all_words.append(word_dict)
                    else:
                        raise Exception(f"No next word with timestamp found for the current word : {word}")
            else:
                # Normal case, with start and end times
                word_dict = {
                    'text': f'{word["word"]}',
                    'start': round(word.get('start', all_words[-1]['end'] if all_words else 0), 2),
                    'end': round(word['end'], 2),
                    'speaker_id': speaker_id
                }
                
                all_words.append(word_dict)
    
    return pd.DataFrame(all_words)

def save_results(df: pd.DataFrame):
    os.makedirs('output/log', exist_ok=True)

    # Remove rows where 'text' is empty
    initial_rows = len(df)
    df = df[df['text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        rprint(f"[blue]ℹ️ Removed {removed_rows} row(s) with empty text.[/blue]")

    # Check for and remove words longer than 20 characters
    long_words = df[df['text'].str.len() > 30]
    if not long_words.empty:
        rprint(f"[yellow]⚠️ Warning: Detected {len(long_words)} word(s) longer than 30 characters. These will be removed.[/yellow]")
        df = df[df['text'].str.len() <= 30]

    df['text'] = df['text'].apply(lambda x: f'"{x}"')
    df.to_csv(_2_CLEANED_CHUNKS, index=False, encoding='utf-8-sig')
    rprint(f"[green]📊 CSV file saved to {_2_CLEANED_CHUNKS}[/green]")
