import concurrent.futures
import math
import shutil
import re

from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_2_SPLIT_BY_MEANING_RAW, _3_2_SPLIT_BY_MEANING

console = Console()

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