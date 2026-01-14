import re
import difflib
import math
from typing import List, Tuple

from core.utils import load_key
from core.utils.ask_gpt import ask_gpt
from core.prompts import get_split_prompt
from rich.console import Console

console = Console()

# ================================================================
# 基础功能：分词和长度计算
# ================================================================

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

def clean_word(word):
    """Standardized word cleaner (lowercase, no punctuation) for alignment."""
    word = str(word).lower()
    # Keep letters, numbers, and CJK characters
    cleaned = re.sub(r'[^a-z0-9\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uac00-\ud7af]', '', word)
    return cleaned

def get_word_count(sentence):
    """
    获取句子词数，兼容 CJK 语言
    - 空格分隔语言 (en, es, fr, etc.): 按空格计数
    - CJK 语言 (zh, ja, ko): 英文单词按空格，中文字符按字符
    """
    asr_language = load_key("asr.language")
    is_cjk = asr_language.lower() in ['zh', 'chinese', 'ja', 'japanese', 'ko', 'korean']

    if is_cjk:
        # CJK: 保持英文+数字复合词完整，中文按字符分割
        words = re.findall(r'[a-zA-Z]+(?:[0-9]+\.?[0-9]*|[.-][0-9]+)*|[^\s]',
                          sentence.replace(' ', ''))
        return len(words)
    else:
        # 空格分隔语言: 按空格计数
        return len(sentence.split())

def tokenize_sentence(sentence):
    """简单分词，替代 spacy"""
    asr_language = load_key("asr.language")
    is_cjk = asr_language.lower() in ['zh', 'chinese', 'ja', 'japanese', 'ko', 'korean']

    if is_cjk:
        # CJK: 英文单词 + 中文字符
        words = re.findall(r'[a-zA-Z]+(?:[0-9]+\.?[0-9]*|[.-][0-9]+)*|[^\s]',
                          sentence.replace(' ', ''))
    else:
        # 空格分隔语言: 按空格分割
        words = sentence.split()

    return words

# ================================================================
# 核心功能：BR标记映射与句子切分
# ================================================================

def map_br_to_original_sentence(original_sentence, llm_sentence_with_br):
    """
    将 LLM 返回句中的 [br] 映射回原句。
    特性：
    1. 修正 Difflib 对齐偏差导致的单词内切分
    2. 保护 CJK (中日韩) 语言的字符间切分
    3. 跳过标点符号
    """
    if '[br]' not in llm_sentence_with_br:
        return original_sentence

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
        
        # 如果落在差异区，找最近的前一个匹配块末尾
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

    # --- 5. 插入标签并执行“边界保护” ---
    final_chars = list(original_sentence)
    # 倒序插入，防止索引错乱
    raw_insert_positions = sorted(list(set(raw_insert_positions)), reverse=True)
    
    for pos in raw_insert_positions:
        current_pos = pos
        
        # 核心修复逻辑：避免英文单词内切分
        while (current_pos > 0 and current_pos < len(final_chars)):
            prev_char = final_chars[current_pos-1]
            curr_char = final_chars[current_pos]
            
            is_prev_word = re.match(r'\w', prev_char) is not None
            is_curr_word = re.match(r'\w', curr_char) is not None
            
            # 遇到非单词字符（空格、标点），停止滑动
            if not (is_prev_word and is_curr_word):
                break
            
            # 遇到了中文/日文/韩文，停止滑动
            if is_cjk(prev_char) or is_cjk(curr_char):
                break
            
            # 切断了英文/拉丁文单词 -> 向后移
            current_pos += 1
            
        # 标点跳过逻辑：不要插在标点前面
        while current_pos < len(final_chars) and final_chars[current_pos] in ",.!?;:，。！？；：":
            current_pos += 1
            
        final_chars.insert(current_pos, " [br] ")

    result = "".join(final_chars)
    # 清理可能产生的多余空格
    result = re.sub(r'\s*\[br\]\s*', ' [br] ', result)
    return result.strip()

def split_sentence(sentence: str, num_parts: int = 2, word_limit: int = 20, index: int = -1) -> str:
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

    if index != -1 and '[br]' in best_split:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')

    return best_split

# ================================================================
# 辅助功能：并行处理和批量操作
# ================================================================

def get_llm_words_and_splits(segmented_sentences: List[str]) -> Tuple[List[str], List[int]]:
    """
    Parses the LLM's sentence list to get its clean words and split indices.
    Returns: (llm_clean_words, llm_split_indices)
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
            pattern = r'[a-zA-Z]+(?:[0-9]+\.?[0-9]*|[.-][0-9]+)*|[0-9]+(?:\.?[0-9]*)?|[^\s]'
            words = re.findall(pattern, sentence.replace(' ', ''))
        else:
            # For space-delimited languages: split by spaces
            words = str(sentence).split()

        if not words:
            continue

        clean_words_in_sentence = [clean_word(w) for w in words]
        clean_words_in_sentence = [w for w in clean_words_in_sentence if w]

        current_word_count += len(clean_words_in_sentence)
        llm_clean_words.extend(clean_words_in_sentence)
        llm_split_indices.append(current_word_count)

    return llm_clean_words, llm_split_indices

def map_llm_splits_to_original(original_clean_words: List[str], llm_clean_words: List[str], llm_split_indices: List[int]) -> List[int]:
    """
    Maps LLM's split indices back to the original word list indices using difflib.
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

def reconstruct_sentences(original_words: List[str], original_split_indices: List[int]) -> List[str]:
    """Rebuilds sentences using the *original* word list and the *mapped* split indices."""
    from core.utils import get_joiner
    
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)

    final_sentences = []
    last_idx = 0
    for idx in original_split_indices:
        if idx > last_idx:
            sentence_words = original_words[last_idx:idx]
            sentence_text = joiner.join(sentence_words)
            final_sentences.append(sentence_text)
        last_idx = idx
    return final_sentences
