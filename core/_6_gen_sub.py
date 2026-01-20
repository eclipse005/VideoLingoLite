import pandas as pd
import os
import re
import difflib
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
from core.utils.sentence_tools import clean_word
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
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

def get_sentence_timestamps(df_words, df_sentences):
    """
    Get sentence timestamps using string-level difflib matching.

    Uses character-level matching instead of word-level for better accuracy
    across all languages (CJK, space-separated, etc.).
    """
    time_stamp_list = []

    # 1. 预处理：构建字符到 chunk 的映射
    chars_list = []          # 清洗后的字符列表
    char_to_chunk_idx = []   # 每个字符对应的 chunk 索引
    char_to_start_time = []  # 每个字符对应的开始时间
    char_to_end_time = []    # 每个字符对应的结束时间

    for chunk_idx, row in df_words.iterrows():
        word = row['text']
        cleaned = clean_word(word)
        start_time = float(row['start'])
        end_time = float(row['end'])

        for char in cleaned:
            chars_list.append(char)
            char_to_chunk_idx.append(chunk_idx)
            char_to_start_time.append(start_time)
            char_to_end_time.append(end_time)

    # 拼接成字符串用于匹配
    chunks_string = ''.join(chars_list)
    total_chars = len(chars_list)

    # 当前字符位置（记录已匹配到的最后一个字符的位置）
    current_char_pos = 0

    # 遍历每一句字幕
    for idx, sentence in df_sentences['Source'].items():
        sentence = str(sentence)
        sentence_clean = clean_word(sentence)

        # 如果句子清洗后为空（全是符号），使用上一句的结束时间
        if not sentence_clean:
            if not time_stamp_list:
                time_stamp_list.append((0.0, 0.0))
            else:
                last_end = time_stamp_list[-1][1]
                time_stamp_list.append((last_end, last_end))
            continue

        # 在剩余字符串中进行模糊匹配
        remaining_string = chunks_string[current_char_pos:]

        s = difflib.SequenceMatcher(None, remaining_string, sentence_clean, autojunk=False)
        matching_blocks = s.get_matching_blocks()

        # 寻找匹配块：必须从句子开头匹配 (b_start == 0)
        match_start_rel_idx = -1
        match_length = 0

        for a_start, b_start, length in matching_blocks:
            if b_start == 0:
                match_start_rel_idx = a_start
                match_length = length
                break

        if match_start_rel_idx == -1:
            console.print(f"\n[ yellow]⚠️ Warning: No match found for sentence {idx}: {sentence}[/yellow ]")
            # 兜底策略：沿用上一句时间
            if time_stamp_list:
                start_time = time_stamp_list[-1][1]
            else:
                start_time = 0.0
            time_stamp_list.append((start_time, start_time + 1.0))
            continue

        # 计算时间戳
        abs_start_char_idx = current_char_pos + match_start_rel_idx
        abs_end_char_idx = abs_start_char_idx + match_length - 1

        # 边界检查
        abs_start_char_idx = max(0, min(abs_start_char_idx, total_chars - 1))
        abs_end_char_idx = max(abs_start_char_idx, min(abs_end_char_idx, total_chars - 1))

        # 获取时间戳
        start_time = char_to_start_time[abs_start_char_idx]
        end_time = char_to_end_time[abs_end_char_idx]

        # 时间校验：如果结束时间小于等于开始时间，强制修正
        if end_time <= start_time:
            if abs_end_char_idx < total_chars - 1:
                end_time = char_to_end_time[abs_end_char_idx + 1]
            else:
                end_time = start_time + 0.5

        time_stamp_list.append((start_time, end_time))

        # 更新位置：下一次从当前句子结束字符的下一个字符开始
        current_char_pos = abs_end_char_idx + 1

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
        result = []
        for i, row in df.iterrows():
            # Safe getter: handle NaN and non-string types
            def safe_get(col):
                val = row.get(col, '')
                return str(val).strip() if pd.notna(val) else ''

            line1 = safe_get(columns[0])
            line2 = safe_get(columns[1]) if len(columns) > 1 else ''
            result.append(f"{i+1}\n{row['timestamp']}\n{line1}\n{line2}\n\n")
        return ''.join(result).strip()

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
    df_text = safe_read_csv(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = safe_read_csv(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

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
                    if not current_text and time_gap < 0.5:
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
                    elif current_text == prev_text and time_gap < 0.5:
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
        (os.path.join(_OUTPUT_DIR, 'trans.srt'), os.path.join(_OUTPUT_DIR, 'src.srt'))
    ]
    
    console.print(Panel("[bold blue]🔍 开始检查并合并空字幕...[/bold blue]"))
    
    # 处理所有字幕对
    for trans_path, src_path in subtitle_pairs:
        process_srt_pair(trans_path, src_path)
    
    console.print(Panel("[bold green]🎉 空字幕合并完成！[/bold green]"))    

if __name__ == '__main__':
    align_timestamp_main()