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

def split_audio(audio_file: str, target_len: float = 3*60, win: float = 60) -> List[Tuple[float, float]]:
    ## 在 [target_len-win, target_len+win] 区间内用 pydub 检测静默，切分音频
    rprint(f"[blue]🎙️ Starting audio segmentation {audio_file} {target_len} {win}[/blue]")
    audio = AudioSegment.from_file(audio_file)
    duration = float(mediainfo(audio_file)["duration"])
    if duration <= target_len + win:
        return [(0, duration)]
    segments, pos = [], 0.0

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration)); break

        threshold = pos + target_len

        # 搜索窗口：优先左边 [threshold-win, threshold]，再找右边 [threshold, threshold+win]
        search_ranges = [
            (threshold - win, threshold, "left"),
            (threshold, threshold + win, "right")
        ]

        split_at = threshold  # 默认使用阈值位置
        for ws, we, side in search_ranges:
            if ws < pos:  # 确保不超过当前起始位置
                continue

            ws_ms, we_ms = int(ws * 1000), int(we * 1000)

            # 检测静音区域（至少 0.5 秒）
            silence_regions = detect_silence(
                audio[ws_ms:we_ms],
                min_silence_len=500,
                silence_thresh=-40
            )

            if silence_regions:
                # 转换为绝对时间，筛选 >= 0.5 秒的静音
                silence_regions = [
                    (s/1000 + ws, e/1000 + ws)
                    for s, e in silence_regions
                    if (e - s) >= 500
                ]

                if silence_regions:
                    # 取最后一个静音的中点（靠后的更接近目标）
                    best_silence = silence_regions[-1]
                    split_at = (best_silence[0] + best_silence[1]) / 2
                    break

        # 只有找不到静音时才警告
        if split_at == threshold:
            rprint(f"[yellow]⚠️ No silence found, using threshold {threshold:.1f}s[/yellow]")

        segments.append((pos, split_at))
        pos = split_at

    rprint(f"[green]🎙️ Audio split completed {len(segments)} segments[/green]")
    return segments


def split_audio_by_vad(audio_file: str, max_segment_duration: float = None) -> List[Tuple[float, float]]:
    """
    使用 VAD 检测语音片段并进行切分

    Args:
        audio_file: 音频文件路径
        max_segment_duration: 单段最大时长（秒），默认使用配置值

    Returns:
        List[Tuple[float, float]]: (start, end) 列表，单位秒
    """
    from core.utils.vad_processor import get_speech_segments

    # 使用配置中的默认值
    if max_segment_duration is None:
        max_segment_duration = load_key("vad.max_segment_duration", default=120)

    # VAD 参数
    threshold = load_key("vad.threshold", default=0.5)
    min_speech_ms = load_key("vad.min_speech_ms", default=150)
    min_silence_ms = load_key("vad.min_silence_ms", default=200)
    merge_gap_ms = load_key("vad.merge_gap_ms", default=1000)

    rprint(f"[cyan]🎙️ Using VAD for audio segmentation...[/cyan]")
    rprint(f"[dim]  threshold: {threshold}, min_speech: {min_speech_ms}ms, min_silence: {min_silence_ms}ms, merge_gap: {merge_gap_ms}ms[/dim]")

    segments = get_speech_segments(
        audio_file,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        merge_gap_ms=merge_gap_ms,
        max_segment_duration=max_segment_duration
    )

    rprint(f"[green]🎙️ VAD split completed: {len(segments)} segments[/green]")
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
