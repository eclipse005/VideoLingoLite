import concurrent.futures
import math
from typing import List

from core.utils import *
from core._2_asr import load_chunks
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING, _CACHE_SENTENCES_SPLIT, Sentence
import time

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

        # 使用清洗后文本的长度（字符位置）来查找映射
        clean_pos = len(llm_clean_before_br)
        if clean_pos in llm_to_original:
            original_pos = llm_to_original[clean_pos]
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
    # 注意：这里不需要添加空格，因为 find_br_positions_in_original 返回的位置
    # 是基于清洗后的文本（clean_word 去除了空格），所以 char_to_chunk 也应该不包含空格
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

        # 使用 joiner 拼接子句文本
        asr_language = load_key("asr.language")
        joiner = get_joiner(asr_language)
        sub_text = joiner.join(c.text for c in sub_chunks)

        new_sentence = Sentence(
            chunks=sub_chunks,
            text=sub_text,
            start=sub_chunks[0].start,
            end=sub_chunks[-1].end,
            index=sentence.index + i,
            is_split=True
        )
        new_sentences.append(new_sentence)

    return new_sentences


def parallel_split_sentences(sentences: List[Sentence], max_length: int, max_workers: int, retry_attempt: int = 0) -> List[Sentence]:
    """
    Split sentences in parallel using a thread pool.

    Args:
        sentences: List of Sentence objects to process
        max_length: Maximum effective length per sentence
        max_workers: Number of parallel workers
        retry_attempt: Retry attempt number (for logging)

    Returns:
        Flattened list of Sentence objects (split as needed)
    """
    new_sentences = [None] * len(sentences)
    futures = []

    asr_language = load_key("asr.language")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # Skip if already split
            if sentence.is_split:
                new_sentences[index] = [sentence]
                continue

            # Use effective length for Sentence object
            effective_length = get_effective_length(sentence.text, asr_language)
            num_parts = math.ceil(effective_length / max_length)

            if check_length_exceeds(sentence.text, max_length, asr_language):
                # Submit LLM task with sentence.text
                future = executor.submit(split_sentence, sentence.text, num_parts, max_length, index=index)
                futures.append((future, index, sentence))
            else:
                new_sentences[index] = [sentence]

        total = len(futures)
        # 创建 future 到 (index, sentence) 的映射
        future_to_data = {future: (index, sentence) for future, index, sentence in futures}

        # 只有在有任务时才显示进度条
        if total > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                transient=False,
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]第 {retry_attempt + 1} 轮: 切分句子...", total=total)

                for future in concurrent.futures.as_completed(future_to_data.keys()):
                    index, sentence = future_to_data[future]
                    split_result = future.result()

                    if split_result and '[br]' in split_result:
                        # Use split_sentence_by_br to split the Sentence object
                        split_sentence_list = split_sentence_by_br(sentence, split_result)
                        new_sentences[index] = split_sentence_list
                    else:
                        # No splitting occurred, keep original
                        new_sentences[index] = [sentence]

                    progress.update(task, advance=1)
        else:
            # 没有需要切分的句子，直接跳过
            pass

    # Flatten the list of lists
    return [s for sublist in new_sentences for s in sublist]

@cache_objects(_CACHE_SENTENCES_SPLIT, _3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning(sentences: List[Sentence]) -> List[Sentence]:
    """
    主函数：切分长句 (Stage 2)

    Args:
        sentences: Sentence 对象列表

    Returns:
        List[Sentence]: 切分后的 Sentence 对象列表
    """
    console.print("[blue]🔍 开始 LLM 长句切分 (Stage 2)[/blue]")

    start_time = time.time()

    # 统计需要切分的句子
    asr_language = load_key("asr.language")
    soft_limit = get_language_length_limit(asr_language, 'origin')
    long_sentences = [s for s in sentences if check_length_exceeds(s.text, soft_limit, asr_language)]

    if long_sentences:
        console.print(f'[yellow]⚠️ 发现 {len(long_sentences)} 个需要拆分的长句[/yellow]')
    else:
        console.print(f'[green]✅ 所有句子长度符合要求，无需拆分[/green]')

    # 🔄 多轮处理确保所有长句都被切分
    for retry_attempt in range(3):
        # 检查是否还有需要切分的句子
        remaining_long = [s for s in sentences if check_length_exceeds(s.text, soft_limit, asr_language)]
        if not remaining_long:
            if retry_attempt > 0:
                console.print(f'[green]✅ 所有句子已符合长度限制[/green]')
            break

        # 只有在有需要切分的句子时才调用处理函数
        sentences = parallel_split_sentences(
            sentences,
            max_length=soft_limit,
            max_workers=load_key("max_workers"),
            retry_attempt=retry_attempt
        )

    elapsed = time.time() - start_time
    console.print(f"[dim]⏱️ LLM 长句切分耗时: {format_duration(elapsed)}[/dim]")

    console.print(f'[green]✅ 处理完成！最终句子数: {len(sentences)}[/green]')

    return sentences

def load_sentences() -> List[Sentence]:
    """
    从缓存加载 Sentence 对象

    Returns:
        List[Sentence]: Sentence 对象列表
    """
    import pickle
    import os
    if os.path.exists(_CACHE_SENTENCES_SPLIT):
        with open(_CACHE_SENTENCES_SPLIT, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"缓存文件不存在: {_CACHE_SENTENCES_SPLIT}")


if __name__ == '__main__':
    split_sentences_by_meaning()