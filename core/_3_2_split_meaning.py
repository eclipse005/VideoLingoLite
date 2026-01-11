import concurrent.futures
from difflib import SequenceMatcher
import math
import os
import shutil
import re
from core.prompts import get_split_prompt
from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_2_SPLIT_BY_MEANING_RAW, _3_2_SPLIT_BY_MEANING
console = Console()


# ================================================================
# Tokenization (替代 spacy，支持 CJK)
# ================================================================

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
        # [a-zA-Z]+(?:[0-9]+\.?[0-9]*|[.-][0-9]+)*  保持英文+digits (如 GPT-4, Python3.9)
        # [^\s]  其他字符（主要是中文单字符）
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
# [br] 映射: 借用 _3_llm_sentence_split.py 的 map_br_to_original_sentence
# ================================================================

def map_br_to_raw_positions(original_sentence, llm_sentence_with_br):
    """
    将 LLM 返回句中的 [br] 映射回原句的字符位置。

    借用自 _3_llm_sentence_split.py:201-333 的逻辑，简化为纯位置返回。

    特性：
    1. 修正 difflib 对齐偏差导致的单词内切分
    2. 保护 CJK (中日韩) 语言的字符间切分
    3. 跳过标点符号
    """
    if '[br]' not in llm_sentence_with_br:
        return []

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
    matcher = SequenceMatcher(None, orig_clean_str, llm_clean_str, autojunk=False)
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

    # --- 5. 边界保护: 避免切在英文单词中间 ---
    final_positions = []
    for pos in raw_insert_positions:
        current_pos = pos

        # 检查当前插入点左右是否都是"单词字符"，且都不是 CJK
        while (current_pos > 0 and current_pos < len(original_sentence)):
            prev_char = original_sentence[current_pos - 1]
            curr_char = original_sentence[current_pos] if current_pos < len(original_sentence) else ''

            is_prev_word = re.match(r'\w', prev_char) is not None
            is_curr_word = re.match(r'\w', curr_char) is not None if curr_char else False

            # 遇到非单词字符（空格、标点），停止滑动
            if not (is_prev_word and is_curr_word):
                break

            # 遇到 CJK，停止滑动
            if is_cjk(prev_char) or is_cjk(curr_char):
                break

            # 切断英文单词 → 向后移
            current_pos += 1

        # 标点跳过: 不要插在标点前面
        while current_pos < len(original_sentence) and original_sentence[current_pos] in ",.!?;:，。！？；：":
            current_pos += 1

        final_positions.append(current_pos)

    return final_positions

def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0):
    """Split a long sentence using LLM and return the result as a string."""
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
        log_title='split_long_sentence'
    )

    choice = response_data.get("choice", "1")
    best_split = response_data.get(f"split{choice}", sentence)

    # 使用新的 map_br_to_raw_positions 函数
    if '[br]' in best_split:
        split_positions = map_br_to_raw_positions(sentence, best_split)

        # 按切分点重建句子
        if split_positions:
            result_parts = []
            last_pos = 0
            for pos in split_positions:
                if pos > last_pos:
                    result_parts.append(sentence[last_pos:pos].strip())
                    last_pos = pos
            if last_pos < len(sentence):
                result_parts.append(sentence[last_pos:].strip())

            best_split = '\n'.join(result_parts)

            if index != -1:
                console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')

            # 显示切分结果
            table = Table(title="")
            table.add_column("Type", style="cyan")
            table.add_column("Sentence")
            table.add_row("Original", sentence, style="yellow")
            table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
            console.print(table)

    return best_split

def parallel_split_sentences(sentences, max_length, max_workers, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    asr_language = load_key("asr.language")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # 使用有效长度判断
            effective_length = get_effective_length(sentence, asr_language)
            num_parts = math.ceil(effective_length / max_length)
            if check_length_exceeds(sentence, max_length, asr_language):
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index, retry_attempt=retry_attempt)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                split_lines = split_result.strip().split('\n')
                new_sentences[index] = [line.strip() for line in split_lines]
            else:
                new_sentences[index] = [sentence]

    return [sentence for sublist in new_sentences for sentence in sublist]

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """
    主函数：切分长句

    输入: split_by_meaning_raw.txt (由 LLM 组句或 Parakeet segments 生成)
    输出: split_by_meaning.txt (切分长句后的最终结果)
    """
    # 读取输入句子 (raw)
    with open(_3_2_SPLIT_BY_MEANING_RAW, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]

    console.print(f'[cyan]📖 Loaded {len(sentences)} sentences from {_3_2_SPLIT_BY_MEANING_RAW}[/cyan]')

    # 统计需要切分的句子
    asr_language = load_key("asr.language")
    soft_limit = get_language_length_limit(asr_language, 'origin')
    hard_limit = get_hard_limit(soft_limit, asr_language)
    long_sentences = [s for s in sentences if check_length_exceeds(s, soft_limit, asr_language)]

    if long_sentences:
        console.print(f'[yellow]⚠️ Found {len(long_sentences)} long sentences (> {hard_limit})[/yellow]')
    else:
        console.print(f'[green]✅ No long sentences found, all sentences are within limit.[/green]')
        # 直接复制到最终文件
        shutil.copy(_3_2_SPLIT_BY_MEANING_RAW, _3_2_SPLIT_BY_MEANING)
        console.print(f'[green]💾 Copied to: {_3_2_SPLIT_BY_MEANING}[/green]')
        return sentences

    # 🔄 多轮处理确保所有长句都被切分
    for retry_attempt in range(3):
        console.print(f'[cyan]🔄 Round {retry_attempt + 1}/3: Processing sentences...[/cyan]')
        sentences = parallel_split_sentences(
            sentences,
            max_length=soft_limit,
            max_workers=load_key("max_workers"),
            retry_attempt=retry_attempt
        )

    # 💾 保存结果到最终文件
    with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))

    console.print(f'[green]✅ All sentences processed! Final count: {len(sentences)}[/green]')
    console.print(f'[green]💾 Saved to: {_3_2_SPLIT_BY_MEANING}[/green]')

    return sentences

if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()