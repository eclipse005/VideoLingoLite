"""
ASR Segments to Sentences Module (Stage 1)

直接从 Parakeet ASR 的 segments 创建 Sentence 对象，无需 spaCy 分句
可选择应用停顿分句（基于时间戳）

Output: split_by_nlp.txt (Stage 1 result)
"""

import json
import os
import unicodedata
import pandas as pd
from typing import List

from core.utils import rprint, load_key, timer, safe_read_csv
from core.utils.models import _3_1_SPLIT_BY_NLP, _2_CLEANED_CHUNKS, Chunk, Sentence


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


def group_chunks_into_sentences(chunks: List[Chunk]) -> List[List[Chunk]]:
    """
    根据句子结束符号将 Chunks 分组为句子。

    Args:
        chunks: List[Chunk]

    Returns:
        List of sentence groups, each group is a list of Chunks
    """
    if not chunks:
        return []

    sentences = []
    current_sentence = []

    for chunk in chunks:
        current_sentence.append(chunk)

        # 检查该词是否包含句子结束符号
        if chunk.text:
            # 检查词的任意字符是否为句子结束符
            for char in chunk.text:
                if is_sentence_terminator(char):
                    sentences.append(current_sentence)
                    current_sentence = []
                    break

    # 处理剩余的 chunk
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def load_asr_json() -> dict:
    """加载 ASR 结果 JSON"""
    asr_json_path = "output/log/asr.json"
    if not os.path.exists(asr_json_path):
        raise FileNotFoundError(f"ASR 结果不存在: {asr_json_path}")

    with open(asr_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def has_segment_text(asr_data: dict) -> bool:
    """检查 asr.json 是否包含 segment.text 字段"""
    segments = asr_data.get('segments', [])
    if not segments:
        return False
    return 'text' in segments[0] and 'start' in segments[0] and 'end' in segments[0]


def load_chunks_from_csv() -> List[Chunk]:
    """从 cleaned_chunks.csv 加载 Chunk 对象"""
    df = safe_read_csv(_2_CLEANED_CHUNKS)
    chunks = []

    for row in df.itertuples(index=True):
        speaker_id = row.speaker_id if pd.notna(row.speaker_id) and row.speaker_id else None
        # 清理 text 字段的引号
        text = row.text
        if isinstance(text, str):
            text = text.strip('"')
        chunk = Chunk(
            text=text,
            start=float(row.start),
            end=float(row.end),
            speaker_id=speaker_id,
            index=row.Index
        )
        chunks.append(chunk)

    rprint(f"[green]✅ Loaded {len(chunks)} chunks from {_2_CLEANED_CHUNKS}[/green]")
    return chunks


def create_sentences_from_chunks(chunks: List[Chunk]) -> List[Sentence]:
    """从 Chunk 列表（按标点断句）创建 Sentence 对象"""
    chunk_groups = group_chunks_into_sentences(chunks)
    sentences = []

    for idx, chunk_group in enumerate(chunk_groups):
        if not chunk_group:
            continue

        # 拼接句子文本
        text = ''.join([c.text for c in chunk_group])
        start = chunk_group[0].start
        end = chunk_group[-1].end

        sentence = Sentence(
            chunks=chunk_group,
            text=text,
            start=start,
            end=end,
            index=idx
        )
        sentences.append(sentence)

    return sentences


def create_chunks_from_words(words: List[dict], start_offset: int = 0) -> List[Chunk]:
    """
    从 words 列表创建 Chunk 对象

    Args:
        words: [{'start': float, 'end': float, 'word': str}, ...]
        start_offset: 起始时间偏移（用于调试）

    Returns:
        List[Chunk]
    """
    chunks = []
    for idx, word_info in enumerate(words):
        chunk = Chunk(
            text=word_info['word'],
            start=word_info['start'],
            end=word_info['end'],
            speaker_id=None,  # ASR 没有说话人信息
            index=idx
        )
        chunks.append(chunk)
    return chunks


def segments_to_sentences(asr_data: dict) -> List[Sentence]:
    """
    从 ASR segments 创建 Sentence 对象

    Args:
        asr_data: ASR JSON 数据，包含 segments 列表

    Returns:
        List[Sentence]
    """
    segments = asr_data.get('segments', [])
    if not segments:
        rprint("[yellow]⚠️ 未找到 ASR segments[/yellow]")
        return []

    sentences = []
    for idx, seg in enumerate(segments):
        # 从 words 创建 chunks
        words = seg.get('words', [])
        chunks = create_chunks_from_words(words)

        if not chunks:
            continue

        # 创建 Sentence 对象
        sentence = Sentence(
            chunks=chunks,
            text=seg['text'],
            start=seg['start'],
            end=seg['end'],
            index=idx
        )
        sentences.append(sentence)

    return sentences


@timer("ASR Segments 转句子")
def split_by_spacy() -> List[Sentence]:
    """
    ASR Segments 转句子主函数（Stage 1）

    优先使用 Parakeet ASR 的 segments 直接创建 Sentence 对象
    如果没有 segment.text（Custom ASR），则从 cleaned_chunks.csv 读取并按标点断句
    可选：应用停顿分句

    Returns:
        List[Sentence]: Sentence 对象列表
    """
    # 1. 加载 ASR 结果
    asr_data = load_asr_json()

    # 2. 检查是否有 segment.text，决定使用哪种方式
    if has_segment_text(asr_data):
        # 方式 A：Parakeet 格式，有 segment.text/start/end
        segment_count = len(asr_data.get('segments', []))
        rprint(f"[blue]🔍 Stage 1: ASR Segments → {segment_count} 个句子[/blue]")
        sentences = segments_to_sentences(asr_data)
    else:
        # 方式 B：Custom ASR 格式，没有 segment.text，从 CSV 读取并按标点断句
        rprint(f"[blue]🔍 Stage 1: No segment.text found, using cleaned_chunks.csv + punctuation split[/blue]")
        chunks = load_chunks_from_csv()
        sentences = create_sentences_from_chunks(chunks)
        rprint(f"[blue]   → Punctuation split: {len(chunks)} chunks → {len(sentences)} sentences[/blue]")

    if not sentences:
        rprint("[yellow]⚠️ 没有句子创建，跳过后续处理[/yellow]")
        return []

    # 3. 可选：应用停顿分句
    pause_threshold = load_key("pause_split_threshold")
    if pause_threshold and pause_threshold > 0:
        from core.spacy_utils import split_by_pause
        original_count = len(sentences)
        sentences = split_by_pause(sentences, pause_threshold)
        if len(sentences) != original_count:
            rprint(f"[cyan]  ↪ 停顿分句: {original_count} → {len(sentences)} 个[/cyan]")

    # 4. 保存到文件
    from pathlib import Path
    Path(_3_1_SPLIT_BY_NLP).parent.mkdir(parents=True, exist_ok=True)
    with open(_3_1_SPLIT_BY_NLP, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s.text + '\n')

    rprint(f'[green]✅ Stage 1 完成[/green]')
    return sentences


if __name__ == '__main__':
    split_by_spacy()
