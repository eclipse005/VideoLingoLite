import pandas as pd
import json
import concurrent.futures
import unicodedata
from typing import List
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core.utils import *
from core.utils.models import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher

console = Console()


def split_chunks_by_chars(chunk_size: int, max_i: int, texts: List[str]) -> List[str]:
    """
    Split text into chunks based on character count

    Args:
        chunk_size: Maximum characters per chunk
        max_i: Maximum sentences per chunk
        texts: List of texts

    Returns:
        List of multi-line text chunks
    """
    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in texts:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks


def get_previous_content(chunks: List[str], chunk_index: int) -> List[str] | None:
    """Get previous content for context"""
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:]


def get_after_content(chunks: List[str], chunk_index: int) -> List[str] | None:
    """Get after content for context"""
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2]


def translate_chunk(chunk: str, chunks: List[str], theme_prompt: str, i: int):
    """Translate a single chunk with context"""
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation


def similar(a: str, b: str) -> float:
    """Calculate similarity between two strings using unicodedata normalization"""
    a_norm = unicodedata.normalize('NFC', a.lower())
    b_norm = unicodedata.normalize('NFC', b.lower())
    return SequenceMatcher(None, a_norm, b_norm).ratio()


@timer("翻译")
@cache_objects(_CACHE_SENTENCES_TRANSLATED)
def translate_all(sentences: List[Sentence]) -> List[Sentence]:
    """
    翻译所有句子并填充 Sentence.translation 字段

    Args:
        sentences: Sentence 对象列表

    Returns:
        List[Sentence]: 带翻译的 Sentence 对象列表
    """
    # 准备翻译块（从 Sentence 对象提取文本）
    sentence_texts = [sent.text for sent in sentences]
    chunks = split_chunks_by_chars(chunk_size=1500, max_i=20, texts=sentence_texts)

    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    console.print(f'[cyan]📊 开始翻译 {len(chunks)} 个批次[/cyan]')

    # 🔄 Use concurrent execution for translation
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
            futures.append(future)
        results = []
        total = len(futures)
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            results.append(future.result())
            # 每 20% 显示一次进度
            if total > 0 and (i + 1) % max(1, total // 5) == 0:
                console.print(f'[dim]📊 已翻译 {i + 1}/{total} 个批次 ({(i + 1) * 100 // total}%)[/dim]')

    results.sort(key=lambda x: x[0])  # Sort results based on original order

    # 💾 将翻译结果填充到 Sentence.translation 字段
    sent_idx = 0  # 当前处理到的句子索引
    src_text, trans_text = [], []

    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)

        # Calculate similarity between current chunk and translation results
        chunk_text = ''.join(chunk_lines).lower()
        matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text))
                          for r in results]
        best_match = max(matching_results, key=lambda x: x[1])

        # Check similarity and handle exceptions
        if best_match[1] < 0.9:
            console.print(f"[yellow]Warning: No matching translation found for chunk {i}[/yellow]")
            raise ValueError(f"Translation matching failed (chunk {i})")
        elif best_match[1] < 1.0:
            console.print(f"[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]")

        chunk_translations = best_match[0][2].split('\n')
        trans_text.extend(chunk_translations)

        # 将翻译填充到 Sentence 对象
        for trans in chunk_translations:
            if sent_idx < len(sentences):
                sentences[sent_idx].translation = trans
                sent_idx += 1

    # Save translation results to CSV
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    df_translate.to_csv(_4_2_TRANSLATION, index=False, encoding='utf-8-sig')

    console.print("[bold green]✅ 翻译完成并已保存[/bold green]")

    return sentences


if __name__ == '__main__':
    print("This module requires Sentence objects as input.")
