"""
LLM-based Sentence Segmentation Module (difflib-aligned)

This module uses an LLM to find *natural sentence boundaries* and then
uses difflib to map these boundaries back to the original, unmodified
word sequence.

This guarantees 100% preservation of the original ASR words while
leveraging the LLM's semantic grouping capabilities.
"""

import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.prompts import get_sentence_segmentation_prompt
from core.utils import (
    check_file_exists, get_language_length_limit, get_joiner,
    get_llm_words_and_splits, map_llm_splits_to_original, reconstruct_sentences
)
from core.utils.sentence_tools import clean_word
from core.utils.models import _2_CLEANED_CHUNKS, _3_2_SPLIT_BY_MEANING_RAW
from core.utils.config_utils import load_key
from core.utils.ask_gpt import ask_gpt
from rich.console import Console

console = Console()

# ================================================================
# 辅助函数
# ================================================================

def has_ending_punctuation(word):
    """检查词是否以句子结束标点结尾"""
    endings = ('.', '!', '?', '。', '！', '？', '...', '…')
    return str(word).endswith(endings)

def has_pause_before(word_idx, word_to_chunk_idx, chunks, pause_threshold):
    """检查指定位置之前是否有停顿"""
    if word_idx == 0:
        return False

    chunk_i = word_to_chunk_idx[word_idx]
    chunk_prev = word_to_chunk_idx[word_idx - 1]

    if chunk_i > chunk_prev:
        time_gap = chunks.iloc[chunk_i].start - chunks.iloc[chunk_prev].end
        return time_gap > pause_threshold
    return False

def should_skip_llm(words, pause_after, pause_gap, batch_start_idx, word_to_chunk_idx, chunks, pause_threshold, max_length):
    """判断是否可以跳过 LLM 处理

    跳过条件：
    1. 极短片段 (1-2 个词) + 前后都有停顿
    2. 完整句子（以标点结尾）+ 后面有停顿 + 不超过 max_length
    3. 短片段 (3-10 个词) + 前后都有长停顿
    """
    word_count = len(words)

    # 场景 1: 极短片段 (1-2 个词) + 前后都有停顿
    if word_count <= 2 and pause_after and pause_gap > pause_threshold:
        if has_pause_before(batch_start_idx, word_to_chunk_idx, chunks, pause_threshold):
            return True

    # 场景 2: 完整句子（以标点结尾）+ 后面有停顿 + 不超过 max_length
    # 注意：必须检查词数，否则长句子（200+ 词）会被错误跳过
    if word_count > 0:
        last_word = words[-1]
        if has_ending_punctuation(last_word) and pause_after and word_count <= max_length:
            return True

    # 场景 3: 短片段 (3-10 个词) + 前后都有长停顿
    if 3 <= word_count <= 10 and pause_after and pause_gap > pause_threshold:
        if has_pause_before(batch_start_idx, word_to_chunk_idx, chunks, pause_threshold):
            return True

    return False

# ================================================================
# 主要功能：LLM组句处理
# ================================================================

def validate_segmentation_response(response_data):
    if not isinstance(response_data, dict):
        return {"status": "error", "message": "Response is not a dictionary"}
    sentences = response_data.get("sentences")
    if not isinstance(sentences, list) or len(sentences) == 0:
        return {"status": "error", "message": "Invalid or empty 'sentences'"}
    return {"status": "success", "message": "Validation completed"}

def _process_batch_threaded(batch_info):
    batch_count, batch_words, batch_text, max_length, batch_start_idx, skip_llm = batch_info
    try:
        sentences = process_sentence_chunk(batch_words, max_length, batch_start_idx, skip_llm)
        console.print(f"[green]✅ Batch {batch_count} completed[/green]")
        return (batch_count, sentences)
    except Exception as e:
        console.print(f"[red]❌ Batch {batch_count} failed: {e}[/red]")
        return (batch_count, None)

def process_sentence_chunk(batch_words, max_length, batch_start_idx, skip_llm=False):
    """Process a chunk of words into sentences using LLM and difflib alignment."""
    if skip_llm:
        # 跳过 LLM，直接作为一句输出
        return [' '.join(batch_words)]

    # 转换为文本格式
    words_text = ' '.join(batch_words)
    original_batch_words = batch_words
    original_batch_clean_words = [clean_word(w) for w in original_batch_words]

    prompt = get_sentence_segmentation_prompt(original_batch_words, max_length)

    response_data = ask_gpt(
        prompt,
        resp_type='json',
        valid_def=validate_segmentation_response,
        log_title='llm_sentence_segmentation'
    )

    if response_data.get("status") == "error":
        console.print(f"[red]LLM 响应无效: {response_data.get('message')}[/red]")
        raise RuntimeError(f"LLM 响应无效: {response_data.get('message')}")

    llm_sentences = response_data.get("sentences", [words_text])

    # Get LLM's clean words and split points
    llm_clean_words, llm_split_indices = get_llm_words_and_splits(llm_sentences)

    # Map split points back to original list
    original_split_indices = map_llm_splits_to_original(
        original_batch_clean_words,
        llm_clean_words,
        llm_split_indices
    )

    # Reconstruct sentences preserving original words
    final_sentences = reconstruct_sentences(original_batch_words, original_split_indices)

    if not final_sentences:
        console.print("[yellow]警告: difflib 对齐未产生任何句子，将返回原始文本块[/yellow]")
        return [words_text]

    return final_sentences

@check_file_exists(_3_2_SPLIT_BY_MEANING_RAW)
def llm_sentence_split():
    """
    Main function for LLM-based sentence segmentation.

    Output: split_by_meaning_raw.txt (原始LLM组句结果)
    Skip: If split_by_meaning_raw.txt exists (Parakeet segments 或 LLM 已生成)
    """

    console.print("[blue]🔍 Starting LLM sentence segmentation (difflib-aligned)[/blue]")
    console.print(f"[cyan]📖 Reading input from: {_2_CLEANED_CHUNKS}[/cyan]")

    chunks = pd.read_excel(_2_CLEANED_CHUNKS)
    chunks.text = chunks.text.apply(lambda x: str(x).strip('"').strip())
    words_text = ' '.join(chunks.text.to_list())
    original_total_words_list = words_text.split()

    console.print(f"[green]✅ Loaded {len(chunks)} chunks, {len(original_total_words_list)} words total[/green]")

    # Get max_length from origin_length config
    asr_language = load_key("asr.language")
    max_length = get_language_length_limit(asr_language, 'origin')
    max_workers = load_key("max_workers")
    pause_threshold = load_key("pause_split_threshold")

    # Build mapping: words_list position -> chunks row index
    word_to_chunk_idx = []
    for chunk_idx, text in enumerate(chunks.text):
        words_in_cell = text.split()
        for _ in words_in_cell:
            word_to_chunk_idx.append(chunk_idx)

    # Prepare batches
    words_list = original_total_words_list
    batch_size = 300
    batches = []
    skipped_batches = []  # 存储 skip LLM 的批次
    batch_count = 0
    batch_idx = 0

    while batch_idx < len(words_list):
        batch_count += 1
        batch_end = min(batch_idx + batch_size, len(words_list))
        end_pos = batch_end
        pause_gap = 0
        pause_found = False
        split_reason = "max_length"  # 默认：达到 300 词上限

        # 步骤 1: 正向找第一个停顿（在 300 词范围内）
        if pause_threshold and pause_threshold > 0:
            for i in range(batch_idx, batch_end):
                chunk_i = word_to_chunk_idx[i]
                chunk_next = word_to_chunk_idx[min(i + 1, len(word_to_chunk_idx) - 1)]

                if chunk_next > chunk_i:
                    # Calculate time gap between chunks
                    time_gap = chunks.iloc[chunk_next].start - chunks.iloc[chunk_i].end
                    if time_gap > pause_threshold:
                        end_pos = i + 1
                        pause_gap = time_gap
                        pause_found = True
                        split_reason = f"pause {pause_gap:.1f}s"
                        break

        # 步骤 2: 如果没找到停顿，往回找最后一个句号
        if not pause_found:
            for i in range(batch_end - 1, batch_idx, -1):
                if has_ending_punctuation(words_list[i]):
                    end_pos = i + 1
                    split_reason = "punctuation"
                    break

        batch_words = words_list[batch_idx:end_pos]
        if len(batch_words) == 0:
            batch_idx = end_pos
            continue

        batch_text = ' '.join(batch_words)

        # 步骤 3: 检查是否可以跳过 LLM
        skip_llm = should_skip_llm(
            batch_words, pause_found, pause_gap, batch_idx,
            word_to_chunk_idx, chunks, pause_threshold or 0, max_length
        )

        if skip_llm:
            skipped_batches.append((batch_count, batch_words))
            console.print(f"[yellow]⚡ Batch {batch_count} ({len(batch_words)} words, {split_reason}) - skipping LLM[/yellow]")
        else:
            batches.append((batch_count, batch_words, batch_text, max_length, batch_idx, False))
            console.print(f"[cyan]📦 Batch {batch_count} ({len(batch_words)} words, {split_reason})[/cyan]")

        batch_idx = end_pos

    console.print(f"[cyan]📦 Created {len(batches)} batches for LLM processing[/cyan]")
    console.print(f"[cyan]⚡ Skipped {len(skipped_batches)} batches (no LLM needed)[/cyan]")

    # Process batches in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(_process_batch_threaded, batch_info): batch_info for batch_info in batches}

        for future in as_completed(future_to_batch):
            batch_info = future_to_batch[future]
            try:
                batch_result = future.result()
                if batch_result[1] is not None:
                    results.append(batch_result)
            except Exception as e:
                console.print(f"[red]❌ Batch {batch_info[0]} failed in thread: {e}[/red]")

    # Sort results by batch count
    results.sort(key=lambda x: x[0])

    # Combine all sentences from all batches (including skipped ones)
    all_sentences = []

    # 首先添加跳过 LLM 的批次（按顺序）
    skipped_idx = 0
    result_idx = 0
    current_batch = 1

    while current_batch <= batch_count:
        # 检查是否是跳过的批次
        if skipped_idx < len(skipped_batches) and skipped_batches[skipped_idx][0] == current_batch:
            # 跳过的批次，直接添加
            all_sentences.extend([' '.join(skipped_batches[skipped_idx][1])])
            skipped_idx += 1
        elif result_idx < len(results) and results[result_idx][0] == current_batch:
            # LLM 处理的批次
            all_sentences.extend(results[result_idx][1])
            result_idx += 1
        current_batch += 1

    console.print(f"[green]✅ Generated {len(all_sentences)} sentences total[/green]")

    # Save raw LLM output (no further processing yet)
    with open(_3_2_SPLIT_BY_MEANING_RAW, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_sentences))

    console.print(f"[green]💾 Saved raw LLM segmentation to: {_3_2_SPLIT_BY_MEANING_RAW}[/green]")
    return all_sentences

if __name__ == '__main__':
    llm_sentence_split()
