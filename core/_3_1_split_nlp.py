"""
ASR Segments to Sentences Module (Stage 1)

直接从 Parakeet ASR 的 segments 创建 Sentence 对象，无需 spaCy 分句
可选择应用停顿分句（基于时间戳）

Output: split_by_nlp.txt (Stage 1 result)
"""

import json
import os
from typing import List

from core.utils import rprint, load_key, timer
from core.utils.models import _3_1_SPLIT_BY_NLP, Chunk, Sentence


def load_asr_json() -> dict:
    """加载 ASR 结果 JSON"""
    asr_json_path = "output/log/asr.json"
    if not os.path.exists(asr_json_path):
        raise FileNotFoundError(f"ASR 结果不存在: {asr_json_path}")

    with open(asr_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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

    使用 Parakeet ASR 的 segments 直接创建 Sentence 对象
    可选：应用停顿分句

    Returns:
        List[Sentence]: 从 segments 创建的 Sentence 对象列表
    """
    # 1. 加载 ASR 结果
    asr_data = load_asr_json()
    segment_count = len(asr_data.get('segments', []))
    rprint(f"[blue]🔍 Stage 1: ASR Segments → {segment_count} 个句子[/blue]")

    # 2. 从 segments 创建 Sentence 对象
    sentences = segments_to_sentences(asr_data)
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
