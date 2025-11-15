import pandas as pd
import os
import re
import difflib
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def clean_word(word):
    """
    Standardized word cleaner (lowercase, no punctuation) for alignment.
    """
    return re.sub(r'[^\w]', '', str(word).lower())

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

def get_sentence_timestamps(df_words, df_sentences):
    """
    Get sentence timestamps using difflib-based fuzzy matching.
    This function maps LLM-modified sentences back to the original word sequence to find time ranges.
    """
    time_stamp_list = []

    # Prepare original clean word list (from df_words)
    original_words = df_words['text'].tolist()
    original_clean_words = [clean_word(w) for w in original_words]
    original_clean_words = [w for w in original_clean_words if w]  # Remove empty strings

    # Track current position to avoid searching from the beginning each time
    current_pos = 0

    # Process each sentence
    for idx, sentence in df_sentences['Source'].items():
        sentence_words = str(sentence).split()
        sentence_clean_words = [clean_word(w) for w in sentence_words]
        sentence_clean_words = [w for w in sentence_clean_words if w]  # Remove empty strings

        if not sentence_clean_words:
            # Empty sentence, skip
            time_stamp_list.append((float(df_words['start'].iloc[-1]), float(df_words['end'].iloc[-1])))
            continue

        # Use difflib to find matching blocks between original and sentence
        # Search only from the current position forward (intelligent starting position)
        remaining_original = original_clean_words[current_pos:]
        s = difflib.SequenceMatcher(None, remaining_original, sentence_clean_words, autojunk=False)
        matching_blocks = s.get_matching_blocks()

        # Find a match starting from the beginning of the sentence
        match_start_idx = -1
        match_length = 0

        for a_start, b_start, length in matching_blocks:
            if b_start == 0:  # Block starts at the beginning of sentence
                match_start_idx = a_start
                match_length = length
                break

        # If no match found, fail immediately
        if match_start_idx == -1:
            print(f"\n⚠️ Warning: No close match found for sentence: {sentence}")
            print(f"Expected: {''.join(sentence_clean_words)}")

            # Show closest matches for debugging
            close_matches = difflib.get_close_matches(
                ''.join(sentence_clean_words),
                [''.join(remaining_original[i:i+20]) for i in range(0, min(len(remaining_original), 100), 20)],
                n=3,
                cutoff=0.5
            )
            if close_matches:
                print("Close matches found:")
                for cm in close_matches:
                    print(f"  - {cm}")

            raise ValueError(f"❎ No match found for sentence: {sentence}")

        # Map to original word indices
        start_word_idx = current_pos + match_start_idx

        # Find the end index
        end_word_idx = start_word_idx + match_length - 1

        # If we only matched part of the sentence, we need to find the rest
        if match_length < len(sentence_clean_words):
            # We need to find additional matching blocks to complete the sentence
            found_words = match_length
            temp_start = match_start_idx

            # Look for subsequent matching blocks
            for a_start, b_start, length in matching_blocks:
                if a_start > temp_start:
                    # Check if this block continues from where we left off
                    if b_start == found_words:
                        # This block matches the next part of the sentence
                        additional_length = min(length, len(sentence_clean_words) - found_words)
                        end_word_idx = current_pos + a_start + additional_length - 1
                        found_words += additional_length

                        if found_words >= len(sentence_clean_words):
                            break

                    temp_start = a_start

        # Ensure we have valid indices
        start_word_idx = max(0, min(start_word_idx, len(original_words) - 1))
        end_word_idx = max(start_word_idx, min(end_word_idx, len(original_words) - 1))

        # Get timestamps
        start_time = float(df_words['start'].iloc[start_word_idx])
        end_time = float(df_words['end'].iloc[end_word_idx])

        # Verify timestamps are reasonable
        if end_time <= start_time:
            # Fallback: extend to next word
            if end_word_idx < len(df_words) - 1:
                end_time = float(df_words['end'].iloc[end_word_idx + 1])
            else:
                end_time = start_time + 1.0  # Default 1 second

        time_stamp_list.append((start_time, end_time))

        # Update current position to end of matched sentence
        # (This ensures next sentence starts searching from here)
        current_pos = end_word_idx + 1

    return time_stamp_list

def align_timestamp(df_text, df_translate, subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    # Remove gaps 🕳️
    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(lambda x: re.sub(r'[，。]', ' ', x).strip())

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        return ''.join([f"{i+1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate_for_audio, AUDIO_SUBTITLE_OUTPUT_CONFIGS, _AUDIO_DIR)
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    # 合并空字幕
    merge_empty_subtitle()

def merge_empty_subtitle():
    """合并空字幕：检查空行的字幕和前一行是否连续，如果间隔小于0.3秒则删除空字幕并更新前字幕的结束时间
    同时处理对应的中英文字幕文件"""
    import pysrt
    
    def process_srt_pair(trans_path, src_path):
        """处理中文字幕和对应的英文字幕文件对"""
        if not os.path.exists(trans_path):
            console.print(f"[yellow]⚠️ 文件不存在: {trans_path}[/yellow]")
            return
            
        if not os.path.exists(src_path):
            console.print(f"[yellow]⚠️ 文件不存在: {src_path}[/yellow]")
            return
            
        try:
            # 加载中文字幕和英文字幕
            trans_subs = pysrt.open(trans_path, encoding='utf-8')
            src_subs = pysrt.open(src_path, encoding='utf-8')
            
            if len(trans_subs) != len(src_subs):
                console.print(f"[red]❌ 字幕文件行数不匹配: {trans_path} ({len(trans_subs)}行) vs {src_path} ({len(src_subs)}行)[/red]")
                return
            
            # 找到需要合并的空字幕或重复字幕索引
            to_remove = []
            for i in range(len(trans_subs)):
                if i > 0:  # 从第二个字幕开始检查
                    current_trans = trans_subs[i]
                    prev_trans = trans_subs[i-1]
                    
                    # 检查当前中文字幕是否为空（去除空格后为空字符串）
                    current_text = current_trans.text.strip()
                    prev_text = prev_trans.text.strip()
                    
                    # 计算时间间隔（毫秒转秒）
                    time_gap = (current_trans.start.ordinal - prev_trans.end.ordinal) / 1000.0
                    
                    # 情况1: 空字幕
                    if not current_text and time_gap < 0.3:
                        # 更新中文字幕前一个字幕的结束时间
                        prev_trans.end = current_trans.end
                        
                        # 更新英文字幕前一个字幕的结束时间和内容
                        prev_src = src_subs[i-1]
                        current_src = src_subs[i]
                        prev_src.end = current_src.end
                        
                        # 合并英文字幕内容（用空格连接）
                        prev_src.text = prev_src.text.strip() + " " + current_src.text.strip()
                        
                        to_remove.append(i)
                        console.print(f"[dim]合并空字幕: 第{i+1}行 -> 第{i}行[/dim]")
                    
                    # 情况2: 译文完全重复
                    elif current_text == prev_text and time_gap < 0.3:
                        # 更新中文字幕前一个字幕的结束时间
                        prev_trans.end = current_trans.end
                        
                        # 原文不需要动（包括时间戳和内容）
                        to_remove.append(i)
                        console.print(f"[dim]合并重复字幕: 第{i+1}行 -> 第{i}行[/dim]")
            
            # 从后往前删除，避免索引问题
            for idx in reversed(to_remove):
                del trans_subs[idx]
                del src_subs[idx]
            
            # 重新编号并保存
            for i, (trans_sub, src_sub) in enumerate(zip(trans_subs, src_subs), 1):
                trans_sub.index = i
                src_sub.index = i
            
            trans_subs.save(trans_path, encoding='utf-8')
            src_subs.save(src_path, encoding='utf-8')
            console.print(f"[green]✅ 处理完成: {trans_path} 和 {src_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ 处理失败 {trans_path} 和 {src_path}: {str(e)}[/red]")
    
    # 定义字幕文件对
    subtitle_pairs = [
        (os.path.join(_OUTPUT_DIR, 'trans.srt'), os.path.join(_OUTPUT_DIR, 'src.srt')),
        (os.path.join(_AUDIO_DIR, 'trans_subs_for_audio.srt'), os.path.join(_AUDIO_DIR, 'src_subs_for_audio.srt'))
    ]
    
    console.print(Panel("[bold blue]🔍 开始检查并合并空字幕...[/bold blue]"))
    
    # 处理所有字幕对
    for trans_path, src_path in subtitle_pairs:
        process_srt_pair(trans_path, src_path)
    
    console.print(Panel("[bold green]🎉 空字幕合并完成！[/bold green]"))    

if __name__ == '__main__':
    align_timestamp_main()