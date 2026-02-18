"""
ASR Segments to Sentences Module (Stage 1)

按标点符号将 ASR segments 分割为句子
ASR 后端（qwen/custom）返回单词级时间戳，这里按标点分句

Output: split_by_punctuation.txt (Stage 1 result)
"""

import json
import os
from typing import List

from core.utils import rprint, timer, load_key
from core.utils.models import _3_1_SPLIT_BY_PUNCTUATION, Chunk, Sentence
from core.utils.sentence_splitting import group_words_into_sentences_dicts, is_sentence_terminator
from core.utils.config_utils import get_joiner


def load_asr_json() -> dict:
    """加载 ASR 结果 JSON"""
    asr_json_path = "output/log/asr.json"
    if not os.path.exists(asr_json_path):
        raise FileNotFoundError(f"ASR 结果不存在: {asr_json_path}")

    with open(asr_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def segments_to_sentences(asr_data: dict) -> List[Sentence]:
    """
    按标点符号将 ASR segments 分割为 Sentence 对象

    ASR 后端返回单词级时间戳，这里按标点分句

    Args:
        asr_data: ASR JSON 数据，包含 segments 列表（每个 segment 有 text 和 words）

    Returns:
        List[Sentence]
    """
    segments = asr_data.get('segments', [])
    if not segments:
        rprint("[yellow]⚠️ 未找到 ASR segments[/yellow]")
        return []

    # 获取语言类型和 joiner
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)

    all_sentences = []
    sentence_idx = 0

    for seg in segments:
        seg_text = seg.get('text', '')
        seg_words = seg.get('words', [])

        if not seg_words:
            continue

        # 检查是否包含句子结束符号
        has_terminator = any(is_sentence_terminator(c) for c in seg_text)

        if has_terminator:
            # 按标点符号分句
            word_groups = group_words_into_sentences_dicts(seg_words)

            for word_group in word_groups:
                if not word_group:
                    continue

                # 创建 chunks
                chunks = []
                for word_idx, word_info in enumerate(word_group):
                    chunk = Chunk(
                        text=word_info['word'],
                        start=word_info['start'],
                        end=word_info['end'],
                        speaker_id=None,
                        index=word_idx
                    )
                    chunks.append(chunk)

                # 创建 Sentence 对象
                text = joiner.join([w['word'] for w in word_group])
                start = word_group[0]['start']
                end = word_group[-1]['end']

                sentence = Sentence(
                    chunks=chunks,
                    text=text,
                    start=start,
                    end=end,
                    index=sentence_idx
                )
                all_sentences.append(sentence)
                sentence_idx += 1
        else:
            # 没有标点符号，创建单个 Sentence
            chunks = []
            for word_idx, word_info in enumerate(seg_words):
                chunk = Chunk(
                    text=word_info['word'],
                    start=word_info['start'],
                    end=word_info['end'],
                    speaker_id=None,
                    index=word_idx
                )
                chunks.append(chunk)

            sentence = Sentence(
                chunks=chunks,
                text=seg_text,
                start=seg['start'],
                end=seg['end'],
                index=sentence_idx
            )
            all_sentences.append(sentence)
            sentence_idx += 1

    return all_sentences


@timer("ASR Segments 转句子")
def split_by_punctuation() -> List[Sentence]:
    """
    ASR Segments 转句子主函数（Stage 1）

    按标点符号将 ASR segments 分割为句子

    Returns:
        List[Sentence]: Sentence 对象列表
    """
    # 1. 加载 ASR 结果
    asr_data = load_asr_json()

    # 2. 按标点符号分句
    segments = asr_data.get('segments', [])
    total_words = sum(len(seg.get('words', [])) for seg in segments)
    rprint(f"[blue]🔍 Stage 1: ASR → {len(segments)} segment(s), {total_words} words → 按标点分句[/blue]")

    sentences = segments_to_sentences(asr_data)

    if not sentences:
        rprint("[yellow]⚠️ 没有句子创建，跳过后续处理[/yellow]")
        return []

    rprint(f"[blue]   → {len(sentences)} sentences[/blue]")

    # 3. 保存到文件
    os.makedirs(os.path.dirname(_3_1_SPLIT_BY_PUNCTUATION), exist_ok=True)
    with open(_3_1_SPLIT_BY_PUNCTUATION, 'w', encoding='utf-8') as f:
        for s in sentences:
            f.write(s.text + '\n')

    rprint(f'[green]✅ Stage 1 完成: {len(sentences)} 个句子[/green]')
    return sentences


if __name__ == '__main__':
    split_by_punctuation()
