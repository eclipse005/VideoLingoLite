"""
NLP-based Sentence Segmentation Module (Stage 1)

This module uses spaCy to perform rule-based sentence splitting:
1. Split by punctuation marks (spaCy sentence boundaries)
2. Split by commas (with linguistic analysis)
3. Split by connectors (that, which, because, but, and, etc.)
4. Split long sentences by root (dynamic programming)

Output: split_by_nlp.txt (Stage 1 result)
"""

from typing import List
from spacy.language import Language

from core.spacy_utils import *
from core.utils.models import _3_1_SPLIT_BY_NLP, Chunk, Sentence
from core.utils import rprint, load_key, get_joiner
from core._2_asr import load_chunks


# ------------
# Character Position Mapping Functions
# ------------

def build_char_to_chunk_mapping(chunks: List[Chunk], joiner: str = "") -> List[int]:
    """
    构建字符到 Chunk 索引的映射

    Args:
        chunks: Chunk 对象列表
        joiner: Chunk 之间的连接符（空格分隔语言为 " "，其他为 ""）

    Returns:
        每个字符对应的 Chunk 索引列表
    """
    char_to_chunk = []
    for chunk_idx, chunk in enumerate(chunks):
        # 添加 chunk 的每个字符
        char_to_chunk.extend([chunk_idx] * len(chunk.text))
        # 如果有空格分隔符且不是最后一个 chunk，添加空格的映射
        if joiner and chunk_idx < len(chunks) - 1:
            char_to_chunk.extend([chunk_idx] * len(joiner))
    return char_to_chunk


def nlp_split_to_sentences(chunks: List[Chunk], nlp: Language) -> List[Sentence]:
    """
    使用 spaCy 进行 NLP 分句，将 Chunk 对象组合成 Sentence 对象

    Args:
        chunks: Chunk 对象列表
        nlp: spaCy NLP 模型

    Returns:
        Sentence 对象列表
    """
    # Validate input chunks
    if not chunks:
        return []

    # 获取 ASR 语言并确定连接符（空格分隔语言使用 " "，其他使用 ""）
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)

    # 1. 拼接所有 Chunk 的文本（使用 joiner 分隔）
    full_text = joiner.join(chunk.text for chunk in chunks)
    if not full_text:
        return []

    # 2. 构建字符到 Chunk 的映射（考虑空格）
    char_to_chunk = build_char_to_chunk_mapping(chunks, joiner)

    # 3. 使用 spaCy 分句
    doc = nlp(full_text)
    sentences = []

    for sent_idx, sent in enumerate(doc.sents):
        start_char = sent.start_char
        end_char = sent.end_char

        # 边界检查 - Ensure start_char is within valid range [0, len(full_text)-1]
        if start_char >= len(full_text):
            continue  # Skip invalid sentence
        start_char = max(0, start_char)
        # Ensure end_char is at least start_char + 1 and at most len(full_text)
        end_char = max(start_char + 1, min(end_char, len(full_text)))

        # 找到对应的 Chunk 范围
        start_chunk_idx = char_to_chunk[start_char]
        end_chunk_idx = char_to_chunk[end_char - 1]

        # 提取对应的 Chunk 对象
        sentence_chunks = chunks[start_chunk_idx:end_chunk_idx + 1]

        # 创建 Sentence 对象
        sentence = Sentence(
            chunks=sentence_chunks,
            text=sent.text,
            start=sentence_chunks[0].start if sentence_chunks else 0.0,
            end=sentence_chunks[-1].end if sentence_chunks else 0.0,
            index=sent_idx
        )
        sentences.append(sentence)

    return sentences


def split_by_spacy() -> List[Sentence]:
    """
    NLP 分句主函数（Stage 1）

    使用对象化流程：从 Chunks 生成 Sentence 对象

    Returns:
        List[Sentence]: 分句后的 Sentence 对象列表
    """
    rprint("[blue]🔍 Starting NLP-based sentence segmentation (Stage 1)[/blue]")

    nlp = init_nlp()

    # 使用对象化流程生成 Sentence 对象
    sentences = split_by_nlp(nlp)

    rprint(f"[green]✅ NLP sentence segmentation completed: {_3_1_SPLIT_BY_NLP}[/green]")
    rprint(f"[cyan]📊 Generated {len(sentences)} Sentence objects[/cyan]")
    return sentences


# ------------
# New NLP Split Function with Character Position Tracking
# ------------

def split_by_nlp(nlp: Language) -> List[Sentence]:
    """
    NLP 分句主函数

    输入: cleaned_chunks.csv → List[Chunk]
    输出: List[Sentence] → 保存到 split_by_nlp.txt (文本) 和返回对象
    """
    rprint("[blue]🔍 Starting NLP sentence splitting...[/blue]")

    # 1. 加载 Chunk 对象
    chunks = load_chunks()

    # 2. NLP 分句，生成 Sentence 对象
    sentences = nlp_split_to_sentences(chunks, nlp)

    # 3. 保存文本到文件（向后兼容）
    with open(_3_1_SPLIT_BY_NLP, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent.text + '\n')

    rprint(f'[green]✅ NLP splitting complete! {len(sentences)} sentences generated[/green]')
    rprint(f'[green]💾 Saved to: {_3_1_SPLIT_BY_NLP}[/green]')

    return sentences


if __name__ == '__main__':
    split_by_spacy()
