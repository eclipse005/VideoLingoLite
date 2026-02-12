import re
import difflib
import math
import unicodedata
import time
from typing import List, Tuple

from core.utils import load_key, get_joiner
from core.utils.ask_gpt import ask_gpt
from core.prompts import get_split_prompt
from rich.console import Console

console = Console()


# ================================================================
# 时间格式化工具
# ================================================================

def format_duration(seconds: float) -> str:
    """
    将秒数格式化为 时:分:秒 格式

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串，如 "1:23:45" 或 "23:45" 或 "45s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    elif minutes > 0:
        return f"{minutes}:{secs:02d}"
    else:
        return f"{secs}s"


class Timer:
    """简单的计时器上下文管理器"""

    def __init__(self, name: str = "操作"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        duration_str = format_duration(elapsed)
        console.print(f"[dim]⏱️ {self.name}耗时: {duration_str}[/dim]")


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
    """Extract only letters, numbers, and valid characters for alignment"""
    # Use unicodedata instead of regex, consistent with clean_word()
    result = []
    JAPANESE_MARKS = '\u30fc\u309a\u309b\u309c\u3099\u309d'
    for char in text:
        category = unicodedata.category(char)
        # Keep letters (L*) and numbers (N*)
        if category.startswith('L') or category.startswith('N'):
            # Exclude Japanese modifier marks
            if char not in JAPANESE_MARKS:
                result.append(char)
    return result

def clean_word(word):
    """
    Standardized word cleaner using unicodedata.
    Keeps letters and numbers, removes all punctuation and symbols.
    """
    word = str(word).lower()
    result = []

    # 日文修饰符：长音符号、半浊点、浊点等
    JAPANESE_MARKS = '\u30fc\u309a\u309b\u309c\u3099\u309d'

    for char in word:
        category = unicodedata.category(char)

        # 保留字母(L*)、数字(N*)
        # 去除标点(P*)、符号(S*)、分隔符(Z*)、控制字符(C*)
        if category.startswith('L') or category.startswith('N'):
            # 额外排除日文修饰符
            if char not in JAPANESE_MARKS:
                result.append(char)

    return ''.join(result)

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

    # 返回结果（不在子线程打印日志，由主线程打印）
    return best_split


