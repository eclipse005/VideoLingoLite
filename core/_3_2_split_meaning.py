import concurrent.futures
import math
from typing import List

from core.utils import *
from rich.console import Console
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING, Sentence

console = Console()


def parse_br_positions(llm_output: str) -> List[int]:
    """
    解析 LLM 输出中 [br] 标记的位置

    Args:
        llm_output: LLM 返回的带 [br] 标记的文本

    Returns:
        [br] 在 LLM 输出中的字符位置列表
    """
    import re
    return [m.start() for m in re.finditer(r'\[br\]', llm_output)]


def find_br_positions_in_original(llm_output: str, original_text: str) -> List[int]:
    """
    使用 difflib 找到 [br] 在原始句子中的字符位置

    Args:
        llm_output: LLM 返回的带 [br] 标记的文本
        original_text: 原始句子文本

    Returns:
        [br] 在原始文本中的字符位置列表
    """
    import difflib
    from core.utils.sentence_tools import clean_word

    # 1. 找到 [br] 在 LLM 输出中的位置
    br_positions = parse_br_positions(llm_output)

    if not br_positions:
        return []

    # 2. 清洗文本用于匹配
    llm_clean = clean_word(llm_output.replace('[br]', ''))
    original_clean = clean_word(original_text)

    # 3. 使用 difflib 匹配
    s = difflib.SequenceMatcher(None, llm_clean, original_clean, autojunk=False)
    matching_blocks = s.get_matching_blocks()

    # 4. 建立 LLM 输出位置到原始文本位置的映射
    llm_to_original = {}
    for llm_start, orig_start, length in matching_blocks:
        if length == 0:
            continue
        for i in range(length):
            llm_to_original[llm_start + i] = orig_start + i

    # 5. 找到每个 [br] 对应的原始位置
    original_br_positions = []
    for br_pos in br_positions:
        # [br] 在清洗后文本中的位置（需要去除之前的 [br]）
        llm_before_br = llm_output[:br_pos]
        llm_clean_before_br = clean_word(llm_before_br.replace('[br]', ''))

        if llm_clean_before_br in llm_to_original:
            original_pos = llm_to_original[llm_clean_before_br]
            original_br_positions.append(original_pos)

    return original_br_positions


def split_sentence_by_br(sentence: Sentence, llm_output: str) -> List[Sentence]:
    """
    根据 LLM 返回的 [br] 标记拆分 Sentence

    Args:
        sentence: 原始 Sentence 对象
        llm_output: LLM 返回的带 [br] 标记的文本

    Returns:
        拆分后的 Sentence 对象列表
    """
    from core.utils.sentence_tools import clean_word

    # 1. 找到 [br] 在原始句子中的位置
    br_positions = find_br_positions_in_original(llm_output, sentence.text)

    if not br_positions:
        # 没有需要拆分的地方
        return [sentence]

    # 2. 构建清洗后文本到 Chunk 的映射（关键修复：使用清洗后的文本）
    char_to_chunk = []
    for chunk_idx, chunk in enumerate(sentence.chunks):
        cleaned_chunk_text = clean_word(chunk.text)
        char_to_chunk.extend([chunk_idx] * len(cleaned_chunk_text))

    # 3. 根据 [br] 位置确定 Chunk 拆分点
    split_points = [0]  # 起始点
    for br_pos in br_positions:
        if br_pos < len(char_to_chunk):
            chunk_idx = char_to_chunk[br_pos]
            if chunk_idx not in split_points:
                split_points.append(chunk_idx)
    split_points.append(len(sentence.chunks))  # 结束点
    split_points = sorted(set(split_points))

    # 4. 拆分 Chunks，创建新的 Sentence 对象
    new_sentences = []
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]

        if start_idx >= end_idx:
            continue

        sub_chunks = sentence.chunks[start_idx:end_idx]

        new_sentence = Sentence(
            chunks=sub_chunks,
            text="".join(c.text for c in sub_chunks),
            start=sub_chunks[0].start,
            end=sub_chunks[-1].end,
            index=sentence.index + i,
            is_split=True
        )
        new_sentences.append(new_sentence)

    return new_sentences


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
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                # 处理 [br] 标记为换行
                if '[br]' in split_result:
                    split_lines = [part.strip() for part in split_result.split('[br]')]
                    split_lines = [line for line in split_lines if line]
                else:
                    # 如果没有切分，保持原样
                    split_lines = [split_result.strip()]
                new_sentences[index] = split_lines
            else:
                new_sentences[index] = [sentence]

    return [sentence for sublist in new_sentences for sentence in sublist]

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """
    主函数：切分长句 (Stage 2)

    输入: split_by_nlp.txt (Stage 1 NLP 分句结果)
    输出: split_by_meaning.txt (LLM 切分长句后的最终结果)
    """
    console.print("[blue]🔍 Starting LLM sentence segmentation (Stage 2)[/blue]")

    # 读取 Stage 1 的输出
    with open(_3_1_SPLIT_BY_NLP, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]

    console.print(f'[cyan]📖 Loaded {len(sentences)} sentences from Stage 1 ({_3_1_SPLIT_BY_NLP})[/cyan]')

    # 统计需要切分的句子
    asr_language = load_key("asr.language")
    soft_limit = get_language_length_limit(asr_language, 'origin')
    long_sentences = [s for s in sentences if check_length_exceeds(s, soft_limit, asr_language)]

    if long_sentences:
        console.print(f'[yellow]⚠️ Found {len(long_sentences)} long sentences that need LLM splitting[/yellow]')
    else:
        console.print(f'[green]✅ No long sentences found, all sentences are within limit.[/green]')

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