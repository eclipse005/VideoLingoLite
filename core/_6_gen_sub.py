import pandas as pd
import os
import re
from typing import List
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


# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def generate_subtitles_from_sentences(sentences: List[Sentence], subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """
    直接从 Sentence 对象列表生成字幕，不需要匹配

    Args:
        sentences: Sentence 对象列表
        subtitle_output_configs: 字幕输出配置
        output_dir: 输出目录
        for_display: 是否用于显示
    """
    df_trans_time = []

    for sent in sentences:
        df_trans_time.append({
            'Source': sent.text,
            'Translation': sent.translation,
            'timestamp': (sent.start, sent.end),
            'duration': sent.duration
        })

    # 转换为 DataFrame
    df_trans_time = pd.DataFrame(df_trans_time)

    # 移除间隙
    for i in range(len(df_trans_time) - 1):
        delta_time = df_trans_time.loc[i + 1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (
                df_trans_time.loc[i, 'timestamp'][0],
                df_trans_time.loc[i + 1, 'timestamp'][0]
            )

    # 转换为 SRT 格式
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(
        lambda x: convert_to_srt_format(x[0], x[1])
    )

    # 美化字幕
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(
            lambda x: autocorrect.format(re.sub(r'[，。]', ' ', str(x).strip()).strip())
        )

    # 输出字幕
    def generate_subtitle_string(df, columns):
        result = []
        for i, row in df.iterrows():
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


def align_timestamp_main(sentences=None):
    """
    字幕生成主函数，直接从 Sentence 对象生成字幕

    Args:
        sentences: Sentence 对象列表（如果为 None，从 CSV 加载）
    """
    # 📊 显示接收到的 Sentence 对象信息
    if sentences:
        console.print(f'[cyan]📊 Received {len(sentences)} Sentence objects from Stage 4[/cyan]')
        console.print(f'[dim]Last sentence time: {sentences[-1].start:.2f}s - {sentences[-1].end:.2f}s[/dim]')
    else:
        console.print('[yellow]⚠️ No Sentence objects received, loading from CSV...[/yellow]')

    # 如果没有传入 Sentence 对象，从 CSV 加载（向后兼容）
    if sentences is None:
        from core._2_asr import load_chunks

        df_translate = safe_read_csv(_5_SPLIT_SUB)
        df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)

        src = df_translate['Source'].tolist()
        trans = df_translate['Translation'].tolist()

        # 重建 Sentence 对象
        chunks = load_chunks()
        sentences = []
        char_pos = 0
        chunk_idx = 0

        for src_text, trans_text in zip(src, trans):
            sentence_chunks = []
            text_length = len(src_text)

            while chunk_idx < len(chunks) and char_pos < text_length:
                chunk = chunks[chunk_idx]
                sentence_chunks.append(chunk)
                char_pos += len(chunk.text)
                chunk_idx += 1

            sentence = Sentence(
                chunks=sentence_chunks,
                text=src_text,
                translation=trans_text,
                start=sentence_chunks[0].start if sentence_chunks else 0.0,
                end=sentence_chunks[-1].end if sentence_chunks else 0.0,
                index=len(sentences),
                is_split=False
            )
            sentences.append(sentence)
            char_pos = 0

    # 使用新的函数直接从 Sentence 对象生成字幕
    generate_subtitles_from_sentences(sentences, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR, for_display=True)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))
    console.print(f'[green]✅ Generated subtitles from {len(sentences)} Sentence objects (no difflib matching!)[/green]')

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