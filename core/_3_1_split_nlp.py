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
from core.utils.models import _3_1_SPLIT_BY_NLP, _CACHE_SENTENCES_NLP, Chunk, Sentence
from core.utils import rprint, load_key, get_joiner, Timer, cache_objects
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
    修复：保证 Chunk 原子性，避免由标点符号导致的 Chunk 错误分割
    """
    # Validate input chunks
    if not chunks:
        return []

    # 获取 ASR 语言并确定连接符
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)

    # 1. 拼接所有 Chunk 的文本
    full_text = joiner.join(chunk.text for chunk in chunks)
    if not full_text:
        return []

    # 2. 构建字符到 Chunk 的映射
    char_to_chunk = build_char_to_chunk_mapping(chunks, joiner)

    # 3. 使用 spaCy 分句
    doc = nlp(full_text)
    sentences = []

    # 预先计算每个 Chunk 在 full_text 中的字符边界
    chunk_boundaries = []
    current_pos = 0
    for i, chunk in enumerate(chunks):
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk.text)
        chunk_boundaries.append((chunk_start, chunk_end, i))
        current_pos = chunk_end + len(joiner)

    # 跟踪下一个可用的 Chunk 索引 (Slice 的下界)
    next_available_chunk_idx = 0

    for sent_idx, sent in enumerate(doc.sents):
        start_char = sent.start_char
        end_char = sent.end_char

        # 边界检查
        if start_char >= len(full_text):
            continue
        start_char = max(0, start_char)
        end_char = max(start_char + 1, min(end_char, len(full_text)))

        # 找到对应的 Chunk 范围
        start_chunk_idx = char_to_chunk[start_char]
        end_chunk_idx = char_to_chunk[end_char - 1]

        # 关键修正 1：确保起始 Chunk 不会回退到已使用的 Chunk 之前
        # 如果 spaCy 识别的这个句子的开头还在上一个句子包含的 Chunk 里（例如 "，"），
        # 我们就从下一个可用 Chunk 开始算，这通常会导致 start > end，从而触发下面的跳过逻辑
        start_chunk_idx = max(start_chunk_idx, next_available_chunk_idx)

        # 关键修正 2：如果当前句子映射到的 Chunk 已经被上一个句子用光了，跳过此“残余”句子
        # 例如："终。，" 整个 Chunk 已被归入上一句，spaCy 再把 "，" 识别为新句时，这里会拦截
        if start_chunk_idx >= len(chunks):
            continue

        # 确保 end_chunk_idx 是 Slice 的上界（即包含该 Chunk，所以要 +1 对应 Python 切片语法）
        # 初始时 end_chunk_idx 指向包含 end_char 的那个 chunk 的索引
        
        # 检查分句点是否在 Chunk 内部
        if end_chunk_idx < len(chunk_boundaries):
            end_chunk_start, end_chunk_end, _ = chunk_boundaries[end_chunk_idx]
            
            # 如果 spaCy 的 end_char 在 Chunk 内部（不在边界），强制包含整个 Chunk
            # 这保证了 "终。，" 作为一个整体被归入当前句子
            if end_char > end_chunk_start and end_char < end_chunk_end:
                 # 当前 chunk 必须完整包含，所以切片上界是 index + 1
                 slice_end = end_chunk_idx + 1
            else:
                 # 正好在边界，或者跨越了，也至少要包含到当前这个 chunk
                 slice_end = end_chunk_idx + 1
        else:
            slice_end = end_chunk_idx + 1

        # 修正切片上界：必须至少等于 start
        slice_end = max(slice_end, start_chunk_idx)

        # 提取 Chunk 对象
        sentence_chunks = chunks[start_chunk_idx:slice_end]

        # 关键修正 3：忽略空的句子（当 spaCy 的句子完全落在一个已经被上一句吞并的 Chunk 里时）
        if not sentence_chunks:
            continue

        # 更新指针：下一次从当前切片的末尾开始
        next_available_chunk_idx = slice_end

        # 关键修正 4：根据 Chunk 重建文本，而不是使用 spaCy 的 sent.text
        # 这确保了 Chunk 内的文本（如 "终。，"）是完整的，不会被切断
        reconstructed_text = joiner.join(c.text for c in sentence_chunks)

        sentence = Sentence(
            chunks=sentence_chunks,
            text=reconstructed_text, # 使用重建的完整文本
            start=sentence_chunks[0].start if sentence_chunks else 0.0,
            end=sentence_chunks[-1].end if sentence_chunks else 0.0,
            index=len(sentences) # 使用列表长度作为索引，保证连续
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
    rprint("[blue]🔍 开始 NLP 分句 (Stage 1)[/blue]")

    with Timer("NLP 分句"):
        nlp = init_nlp()

        # 使用对象化流程生成 Sentence 对象
        sentences = split_by_nlp(nlp)

    rprint(f"[green]✅ NLP 分句完成: {_3_1_SPLIT_BY_NLP}[/green]")
    return sentences


# ------------
# New NLP Split Function with Character Position Tracking
# ------------

@cache_objects(_CACHE_SENTENCES_NLP, _3_1_SPLIT_BY_NLP)
def split_by_nlp(nlp: Language) -> List[Sentence]:
    """
    NLP 分句主函数

    输入: cleaned_chunks.csv → List[Chunk]
    输出: List[Sentence] → 保存到 split_by_nlp.txt (文本) 和返回对象
    """
    # 1. 加载 Chunk 对象
    chunks = load_chunks()

    # 2. NLP 分句，生成 Sentence 对象
    sentences = nlp_split_to_sentences(chunks, nlp)

    rprint(f'[green]✅ 处理完成！共 {len(sentences)} 个句子[/green]')

    return sentences


if __name__ == '__main__':
    split_by_spacy()
