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
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.prompts import get_sentence_segmentation_prompt, get_split_prompt
from core.utils import check_file_exists
from core.utils.models import _2_CLEANED_CHUNKS, _3_2_SPLIT_BY_MEANING_RAW, _3_2_SPLIT_BY_MEANING
from core.utils.config_utils import load_key
from core.utils.ask_gpt import ask_gpt
from rich.console import Console

console = Console()

# ================================================================
# Alignment Helper Functions (difflib)
# ================================================================

def clean_word(word):
    """Standardized word cleaner (lowercase, no punctuation) for alignment."""
    word = str(word).lower()
    # Keep letters, numbers, and CJK characters (中文日文韩文)
    cleaned = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', word)
    return cleaned

def get_llm_words_and_splits(segmented_sentences):
    """
    Parses the LLM's sentence list to get its clean words and split indices.
    Returns: (llm_clean_words, llm_split_indices)

    Adaptive tokenization based on language type:
    - Space-delimited languages (English): split by spaces
    - CJK languages (Chinese, Japanese, Korean): smart tokenization
    """
    # Get language setting for adaptive tokenization
    asr_language = load_key("asr.language")
    is_cjk = asr_language.lower() in ['zh', 'chinese', 'ja', 'japanese', 'ko', 'korean']

    llm_clean_words = []
    llm_split_indices = []
    current_word_count = 0

    for sentence in segmented_sentences:
        if is_cjk:
            # For CJK languages: smart tokenization
            # Keep English+number compound words intact, split Chinese by characters
            # Pattern explanation:
            # [a-zA-Z]+(?:[0-9]+\.?[0-9]*|[.-][0-9]+)*  matches English+digits (e.g., Python3.9, GPT-4, v2.0.1)
            # [0-9]+(?:\.?[0-9]*)?  matches pure numbers (e.g., 2023, 99, 3.14)
            # [^\s]  matches other characters (mainly Chinese single characters)
            pattern = r'[a-zA-Z]+(?:[0-9]+\.?[0-9]*|[.-][0-9]+)*|[0-9]+(?:\.?[0-9]*)?|[^\s]'
            words = re.findall(pattern, sentence.replace(' ', ''))
        else:
            # For space-delimited languages: split by spaces (original logic)
            words = str(sentence).split()

        if not words:
            continue

        clean_words_in_sentence = [clean_word(w) for w in words]
        clean_words_in_sentence = [w for w in clean_words_in_sentence if w]

        current_word_count += len(clean_words_in_sentence)
        llm_clean_words.extend(clean_words_in_sentence)
        llm_split_indices.append(current_word_count)

    return llm_clean_words, llm_split_indices

def map_llm_splits_to_original(original_clean_words, llm_clean_words, llm_split_indices):
    """
    Maps LLM's split indices back to the original word list indices using difflib.
    Strategy: 'Next Block Start'. If a split occurs in a mismatch region (e.g., LLM changed words),
    we snap the split to the *start* of the next matching block.
    """
    s = difflib.SequenceMatcher(None, original_clean_words, llm_clean_words, autojunk=False)
    matching_blocks = s.get_matching_blocks()
    
    original_split_indices = []
    
    for llm_split_idx in llm_split_indices:
        mapped_idx = -1
        
        # 1. Check if the split is strictly INSIDE a matching block
        for a_start, b_start, length in matching_blocks:
            b_end = b_start + length
            if b_start < llm_split_idx < b_end:
                b_offset = llm_split_idx - b_start
                mapped_idx = a_start + b_offset
                break
        
        if mapped_idx != -1:
            original_split_indices.append(mapped_idx)
            continue

        # 2. If not inside, find the START of the NEXT matching block
        next_block_start_idx = None
        for a_start, b_start, length in matching_blocks:
            if length == 0: continue
            if b_start >= llm_split_idx:
                next_block_start_idx = a_start
                break
        
        if next_block_start_idx is not None:
            original_split_indices.append(next_block_start_idx)
        else:
            # Split is after all matches
            original_split_indices.append(len(original_clean_words))

    original_split_indices = sorted(list(set(original_split_indices)))
    
    # Safety: Ensure we don't miss the final end if LLM implies it
    if llm_split_indices and llm_split_indices[-1] == len(llm_clean_words):
        if len(original_clean_words) not in original_split_indices:
            original_split_indices.append(len(original_clean_words))

    return original_split_indices

def reconstruct_sentences(original_words, original_split_indices):
    """Rebuilds sentences using the *original* word list and the *mapped* split indices."""
    final_sentences = []
    last_idx = 0
    for idx in original_split_indices:
        if idx > last_idx:
            sentence_words = original_words[last_idx:idx]
            sentence_text = ' '.join(sentence_words)
            final_sentences.append(sentence_text)
        last_idx = idx
    return final_sentences

# ================================================================
# Processing Functions
# ================================================================

def validate_segmentation_response(response_data):
    if not isinstance(response_data, dict):
        return {"status": "error", "message": "Response is not a dictionary"}
    sentences = response_data.get("sentences")
    if not isinstance(sentences, list) or len(sentences) == 0:
        return {"status": "error", "message": "Invalid or empty 'sentences'"}
    return {"status": "success", "message": "Validation completed"}

def _process_batch_threaded(batch_info):
    batch_count, batch_words, batch_text, max_length, batch_start_idx = batch_info
    try:
        sentences = process_sentence_chunk(batch_text, max_length, batch_start_idx)
        console.print(f"[green]✅ Batch {batch_count} completed[/green]")
        return (batch_count, sentences)
    except Exception as e:
        console.print(f"[red]❌ Batch {batch_count} failed: {e}[/red]")
        return (batch_count, None)

def process_sentence_chunk(words_text, max_length, batch_start_idx):
    """Process a chunk of words into sentences using LLM and difflib alignment."""
    original_batch_words = words_text.split()
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

def map_br_to_original_sentence(original_sentence, llm_sentence_with_br):
    """
    将 LLM 返回句中的 [br] 映射回原句。
    特性：
    1. 修正 Difflib 对齐偏差导致的单词内切分 (解决 range [br] s 问题)。
    2. 保护 CJK (中日韩) 语言的字符间切分。
    3. 跳过标点符号。
    """
    if '[br]' not in llm_sentence_with_br:
        return original_sentence

    # --- 辅助函数 ---
    def is_cjk(char):
        """判断字符是否为中日韩字符"""
        code = ord(char)
        return (
            (0x4E00 <= code <= 0x9FFF) or  # 汉字
            (0x3040 <= code <= 0x309F) or  # 平假名
            (0x30A0 <= code <= 0x30FF) or  # 片假名
            (0xAC00 <= code <= 0xD7AF)     # 韩文
        )

    def get_clean_chars(text):
        """只提取字母、数字、汉字等有效字符用于对齐"""
        return [c for c in text if re.match(r'\w', c)]

    # --- 1. 数据准备 ---
    orig_clean_chars = get_clean_chars(original_sentence)
    orig_clean_str = "".join(orig_clean_chars)

    llm_text_no_br = llm_sentence_with_br.replace('[br]', '')
    llm_clean_chars = get_clean_chars(llm_text_no_br)
    llm_clean_str = "".join(llm_clean_chars)

    # --- 2. 确定 LLM 中的切分点 (第 N 个有效字符后) ---
    br_anchor_indices = [] 
    parts = llm_sentence_with_br.split('[br]')
    current_valid_char_count = 0
    
    for part in parts[:-1]:
        part_clean_count = len(get_clean_chars(part))
        current_valid_char_count += part_clean_count
        br_anchor_indices.append(current_valid_char_count)

    # --- 3. 使用 Diff 算法映射索引 ---
    matcher = difflib.SequenceMatcher(None, orig_clean_str, llm_clean_str, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()
    
    mapped_clean_indices = []
    
    for llm_idx in br_anchor_indices:
        best_match_idx = -1
        # 尝试找到包含该 LLM 索引的匹配块
        for a_start, b_start, length in matching_blocks:
            b_end = b_start + length
            if b_start <= llm_idx <= b_end:
                offset = llm_idx - b_start
                best_match_idx = a_start + offset
                break
        
        # 如果落在差异区（比如 LLM 加了词导致索引偏移），找最近的前一个匹配块末尾
        if best_match_idx == -1:
            closest_preceding_match_end = 0
            for a_start, b_start, length in matching_blocks:
                b_end = b_start + length
                if b_end <= llm_idx:
                    closest_preceding_match_end = a_start + length
                else:
                    break 
            best_match_idx = closest_preceding_match_end
            
        best_match_idx = min(best_match_idx, len(orig_clean_str))
        mapped_clean_indices.append(best_match_idx)

    # --- 4. 将 Clean 索引转回 Raw 索引 ---
    raw_insert_positions = []
    for target_clean_idx in mapped_clean_indices:
        raw_idx = 0
        clean_counter = 0
        found = False
        
        for char in original_sentence:
            if re.match(r'\w', char):
                clean_counter += 1
            raw_idx += 1
            if clean_counter == target_clean_idx:
                raw_insert_positions.append(raw_idx)
                found = True
                break
        
        if not found:
             raw_insert_positions.append(len(original_sentence))

    # --- 5. 插入标签并执行“边界保护” (关键步骤) ---
    final_chars = list(original_sentence)
    # 倒序插入，防止索引错乱
    raw_insert_positions = sorted(list(set(raw_insert_positions)), reverse=True)
    
    for pos in raw_insert_positions:
        current_pos = pos
        
        # === 核心修复逻辑 ===
        # 检查当前插入点左右是否都是“单词字符”，且都不是 CJK。
        # 如果是，说明切到了英文单词内部 (如 range|s)，必须强制向后滑。
        while (current_pos > 0 and current_pos < len(final_chars)):
            prev_char = final_chars[current_pos-1]
            curr_char = final_chars[current_pos]
            
            is_prev_word = re.match(r'\w', prev_char) is not None
            is_curr_word = re.match(r'\w', curr_char) is not None
            
            # 1. 如果遇到非单词字符（空格、标点），说明单词结束了，停止滑动
            if not (is_prev_word and is_curr_word):
                break
            
            # 2. 如果遇到了中文/日文/韩文，说明这是正常的紧凑语言，停止滑动
            if is_cjk(prev_char) or is_cjk(curr_char):
                break
            
            # 3. 既是单词字符，又不是CJK -> 说明切断了英文/拉丁文单词 -> 向后移！
            current_pos += 1
            
        # === 标点跳过逻辑 ===
        # 不要插在标点前面，比如 "word, [br]" 而不是 "word [br] ,"
        while current_pos < len(final_chars) and final_chars[current_pos] in ",.!?;:，。！？；：":
            current_pos += 1
            
        final_chars.insert(current_pos, " [br] ")

    result = "".join(final_chars)
    # 清理可能产生的多余空格
    result = re.sub(r'\s*\[br\]\s*', ' [br] ', result)
    return result.strip()

def split_sentence(sentence, num_parts=2, word_limit=20, index=-1):
    """Split a single sentence into parts using LLM."""
    split_prompt = get_split_prompt(sentence, num_parts, word_limit)

    def valid_split(response_data):
        choice = response_data.get("choice", "1")
        split_key = f'split{choice}'
        if split_key not in response_data:
            return {"status": "error", "message": f"Missing required key: `{split_key}`"}
        if "[br]" not in response_data[split_key]:
            return {"status": "success", "message": "Split not required by LLM."}
        return {"status": "success", "message": "Split completed"}

    response_data = ask_gpt(
        split_prompt,
        resp_type='json',
        valid_def=valid_split,
        log_title='split_single_sentence'
    )

    choice = response_data.get("choice", "1")
    best_split = response_data.get(f"split{choice}", sentence)

    if '[br]' in best_split:
        mapped_split = map_br_to_original_sentence(sentence, best_split)
        if best_split.count('[br]') == mapped_split.count('[br]'):
            best_split = mapped_split

    # 不需要替换为 \n，直接返回带 [br] 的结果
    if index != -1 and '[br]' in best_split:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')

    return best_split

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def llm_sentence_split():
    """
    Main function for LLM-based sentence segmentation.

    Output: split_by_meaning_raw.txt (原始LLM组句结果)
    Skip: If split_by_meaning.txt exists (最终结果已生成) 或 Parakeet 模式
    """
    # Check if final result already exists
    import os
    if os.path.exists(_3_2_SPLIT_BY_MEANING):
        with open(_3_2_SPLIT_BY_MEANING, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        if sentences:
            console.print(f"[yellow]⏭️ Skipping LLM sentence splitting (final result already exists: {len(sentences)} sentences)[/yellow]")
            return sentences

    # Check if Parakeet mode - if so, skip LLM processing
    asr_runtime = load_key("asr.runtime")
    if asr_runtime == "parakeet":
        console.print(f"[yellow]⏭️ Skipping LLM sentence splitting (Parakeet mode, using raw segments from ASR)[/yellow]")
        # Return empty list to indicate we're using Parakeet's raw output
        return []

    console.print("[blue]🔍 Starting LLM sentence segmentation (difflib-aligned)[/blue]")
    console.print(f"[cyan]📖 Reading input from: {_2_CLEANED_CHUNKS}[/cyan]")

    chunks = pd.read_excel(_2_CLEANED_CHUNKS)
    chunks.text = chunks.text.apply(lambda x: str(x).strip('"').strip())
    words_text = ' '.join(chunks.text.to_list())
    original_total_words_list = words_text.split()

    console.print(f"[green]✅ Loaded {len(chunks)} chunks, {len(original_total_words_list)} words total[/green]")

    max_length = load_key("max_split_length")
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
    batch_count = 0
    batch_idx = 0

    while batch_idx < len(words_list):
        batch_count += 1
        batch_end = min(batch_idx + batch_size, len(words_list))
        end_pos = batch_end

        # Check for pause-based split (time gap > threshold)
        if pause_threshold and pause_threshold > 0:
            for i in range(batch_end - 1, max(batch_idx, batch_end - 50), -1):
                chunk_i = word_to_chunk_idx[i]
                chunk_next = word_to_chunk_idx[min(i + 1, len(word_to_chunk_idx) - 1)]
                gap = float(chunks.iloc[chunk_next]['start']) - float(chunks.iloc[chunk_i]['end'])
                if gap > pause_threshold:
                    end_pos = i + 1
                    console.print(f"[cyan]📍 Pause split at word {i} ({end_pos - batch_idx} words): {gap:.1f}s gap[/cyan]")
                    break
            else:
                # No pause split found, use sentence terminator logic
                for i in range(batch_end, min(batch_end + 50, len(words_list))):
                    if '.' in words_list[i] or '。' in words_list[i]:
                        end_pos = i + 1
                        break
        else:
            # Original logic: adjust to sentence terminator
            for i in range(batch_end, min(batch_end + 50, len(words_list))):
                if '.' in words_list[i] or '。' in words_list[i]:
                    end_pos = i + 1
                    break

        batch_words = words_list[batch_idx:end_pos]
        batch_text = ' '.join(batch_words)
        batches.append((batch_count, batch_words, batch_text, max_length, batch_idx))
        batch_idx = end_pos

    console.print(f"[cyan]🚀 Submitting {len(batches)} batches to thread pool (pause_threshold={pause_threshold})[/cyan]")

    all_sentences = [None] * len(batches)
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(_process_batch_threaded, batch_info): batch_info[0]
            for batch_info in batches
        }

        for future in as_completed(future_to_batch):
            batch_count_result, sentences = future.result()

            if sentences:
                all_sentences[batch_count_result - 1] = sentences
                completed_batches += 1
                if completed_batches % 10 == 0 or completed_batches == len(batches):
                    console.print(f"[green]✅ Completed {completed_batches}/{len(batches)} batches[/green]")
            else:
                console.print(f"[red]❌ Batch {batch_count_result} failed - aborting[/red]")
                for f in future_to_batch:
                    if not f.done(): f.cancel()
                raise RuntimeError(f"Batch {batch_count_result} processing failed")

    final_sentences = []
    for sentences in all_sentences:
        if sentences:
            final_sentences.extend(sentences)

    console.print(f"[green]✅ All {len(batches)} batches processed successfully.[/green]")
    console.print(f"[cyan]💾 Saving results to: {_3_2_SPLIT_BY_MEANING_RAW}[/cyan]")

    with open(_3_2_SPLIT_BY_MEANING_RAW, 'w', encoding='utf-8') as f:
        for sentence in final_sentences:
            f.write(sentence + '\n')

    return final_sentences

if __name__ == '__main__':
    llm_sentence_split()