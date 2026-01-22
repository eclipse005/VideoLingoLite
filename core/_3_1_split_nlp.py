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
# SudachiPy token length limit (bytes) - must match split_by_mark.py
SUDACHI_MAX_LENGTH = 40000

from core.utils.models import _3_1_SPLIT_BY_NLP, _CACHE_SENTENCES_NLP, Chunk, Sentence
from core.utils import rprint, load_key, get_joiner, timer, cache_objects
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


def _process_batch(chunks: List[Chunk], nlp: Language, joiner: str) -> List[Sentence]:
    """
    处理单个批次的 Chunk，返回 Sentence 对象列表

    这是 nlp_split_to_sentences 的核心逻辑，提取为独立函数以支持分批处理
    """
    # 1. 拼接当前批次的 Chunk 文本
    full_text = joiner.join(chunk.text for chunk in chunks)
    if not full_text:
        return []

    # 2. 构建字符到 Chunk 的映射（仅针对当前批次）
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
        start_chunk_idx = max(start_chunk_idx, next_available_chunk_idx)

        # 关键修正 2：如果当前句子映射到的 Chunk 已经被上一个句子用光了，跳过
        if start_chunk_idx >= len(chunks):
            continue

        # 确保 end_chunk_idx 是 Slice 的上界
        if end_chunk_idx < len(chunk_boundaries):
            end_chunk_start, end_chunk_end, _ = chunk_boundaries[end_chunk_idx]

            if end_char > end_chunk_start and end_char < end_chunk_end:
                 slice_end = end_chunk_idx + 1
            else:
                 slice_end = end_chunk_idx + 1
        else:
            slice_end = end_chunk_idx + 1

        # 修正切片上界：必须至少等于 start
        slice_end = max(slice_end, start_chunk_idx)

        # 提取 Chunk 对象
        sentence_chunks = chunks[start_chunk_idx:slice_end]

        # 关键修正 3：忽略空的句子
        if not sentence_chunks:
            continue

        # 更新指针：下一次从当前切片的末尾开始
        next_available_chunk_idx = slice_end

        # 关键修正 4：根据 Chunk 重建文本
        reconstructed_text = joiner.join(c.text for c in sentence_chunks)

        sentence = Sentence(
            chunks=sentence_chunks,
            text=reconstructed_text,
            start=sentence_chunks[0].start if sentence_chunks else 0.0,
            end=sentence_chunks[-1].end if sentence_chunks else 0.0,
            index=len(sentences)
        )
        sentences.append(sentence)

    return sentences


def _process_japanese_in_batches(chunks: List[Chunk], nlp: Language, joiner: str) -> List[Sentence]:
    """
    对日语进行分批处理，避免 SudachiPy 字节限制

    策略：
    1. 按 Chunk 边界累积字节大小
    2. 接近 SUDACHI_MAX_LENGTH 时，向前查找结束符号（。！？）切分
    3. 找不到结束符号则按字节强制切分
    4. 每批独立处理，最后合并结果

    注意：静默处理，不打印详细日志
    """
    all_sentences = []

    # 计算每个 Chunk 的字节大小和累积字节
    chunk_bytes = [len(chunk.text.encode('utf-8')) for chunk in chunks]
    cumulative_bytes = []
    total = 0
    for b in chunk_bytes:
        total += b
        cumulative_bytes.append(total)

    # 分批处理
    batch_start = 0
    batch_count = 0

    while batch_start < len(chunks):
        # 找到当前批次的结束位置
        batch_end = batch_start
        batch_bytes = 0

        while batch_end < len(chunks):
            next_chunk_bytes = chunk_bytes[batch_end]

            # 如果添加此 Chunk 会超过限制
            if batch_bytes + next_chunk_bytes > SUDACHI_MAX_LENGTH and batch_end > batch_start:
                # 向前查找最近的结束符号（。！？）
                sentence_end_found = False
                for look_back in range(min(50, batch_end - batch_start)):  # 最多回溯50个chunk
                    check_idx = batch_end - 1 - look_back
                    if check_idx < batch_start:
                        break
                    chunk_text = chunks[check_idx].text
                    # 检查是否包含日语结束符号
                    if any(punct in chunk_text for punct in ['。', '！', '？', '．', '!', '?']):
                        batch_end = check_idx + 1  # 在包含结束符号的chunk之后切分
                        sentence_end_found = True
                        break

                if sentence_end_found:
                    break  # 找到结束符号，在此切分
                # 否则继续尝试添加下一个chunk（向下面的逻辑继续）

            batch_bytes += next_chunk_bytes
            batch_end += 1

        batch_chunks = chunks[batch_start:batch_end]
        batch_count += 1

        # 处理当前批次
        batch_sentences = _process_batch(batch_chunks, nlp, joiner)
        all_sentences.extend(batch_sentences)

        # 移动到下一批次
        batch_start = batch_end

    return all_sentences


def nlp_split_to_sentences(chunks: List[Chunk], nlp: Language) -> List[Sentence]:
    """
    使用 spaCy 进行 NLP 分句，将 Chunk 对象组合成 Sentence 对象
    修复：保证 Chunk 原子性，避免由标点符号导致的 Chunk 错误分割

    对于日语：分批处理以避免 SudachiPy 字节限制
    """
    # Validate input chunks
    if not chunks:
        return []

    # 获取 ASR 语言并确定连接符
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)

    # 日语需要分批处理
    if asr_language == 'ja':
        return _process_japanese_in_batches(chunks, nlp, joiner)

    # 其他语言使用原有逻辑
    return _process_batch(chunks, nlp, joiner)


@timer("NLP 分句")
def split_by_spacy() -> List[Sentence]:
    """
    NLP 分句主函数（Stage 1）

    使用对象化流程：从 Chunks 生成 Sentence 对象

    Returns:
        List[Sentence]: 分句后的 Sentence 对象列表
    """
    rprint("[blue]🔍 开始 NLP 分句 (Stage 1)[/blue]")

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
