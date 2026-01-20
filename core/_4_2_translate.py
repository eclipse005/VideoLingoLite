import pandas as pd
import json
import concurrent.futures
import unicodedata
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.models import *
console = Console()

# Function to split text into chunks
def split_chunks_by_chars(chunk_size, max_i, texts=None):
    """
    Split text into chunks based on character count

    Args:
        chunk_size: Maximum characters per chunk
        max_i: Maximum sentences per chunk
        texts: List of texts (if None, load from file)

    Returns:
        List of multi-line text chunks
    """
    if texts is None:
        with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
            sentences = file.read().strip().split('\n')
    else:
        sentences = texts

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:] # Get last 3 lines
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2] # Get first 2 lines

# 🔍 Translate a single chunk
def translate_chunk(chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation

# Add similarity calculation function
def similar(a, b):
    # Use unicodedata normalization to handle composed characters, improving matching accuracy
    a_norm = unicodedata.normalize('NFC', a.lower())
    b_norm = unicodedata.normalize('NFC', b.lower())
    return SequenceMatcher(None, a_norm, b_norm).ratio()

# 🚀 Main function to translate all chunks
@check_file_exists(_4_2_TRANSLATION)
def translate_all(sentences=None):
    """
    翻译所有句子并填充 Sentence.translation 字段

    Args:
        sentences: Sentence 对象列表（如果为 None，从文本文件加载）

    Returns:
        List[Sentence]: 带翻译的 Sentence 对象列表
    """
    console.print("[bold green]Start Translating All...[/bold green]")

    # 如果没有传入 Sentence 对象，从文本文件加载（向后兼容）
    if sentences is None:
        from core._2_asr import load_chunks
        from core._3_1_split_nlp import build_char_to_chunk_mapping
        import spacy

        # 从 split_by_meaning.txt 加载文本
        with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
            text_lines = [line.strip() for line in file.readlines() if line.strip()]

        # 重建 Sentence 对象
        chunks = load_chunks()
        sentences = []
        char_pos = 0
        chunk_idx = 0

        for text_line in text_lines:
            sentence_chunks = []
            text_length = len(text_line)

            while chunk_idx < len(chunks) and char_pos < text_length:
                chunk = chunks[chunk_idx]
                sentence_chunks.append(chunk)
                char_pos += len(chunk.text)
                chunk_idx += 1

            sentence = Sentence(
                chunks=sentence_chunks,
                text=text_line,
                start=sentence_chunks[0].start if sentence_chunks else 0.0,
                end=sentence_chunks[-1].end if sentence_chunks else 0.0,
                index=len(sentences),
                is_split=False
            )
            sentences.append(sentence)
            char_pos = 0

    # 准备翻译块（从 Sentence 对象提取文本）
    sentence_texts = [sent.text for sent in sentences]
    chunks = split_chunks_by_chars(chunk_size=1500, max_i=20, texts=sentence_texts)

    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    # 🔄 Use concurrent execution for translation
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
                futures.append(future)
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(task, advance=1)

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

    # Save translation results to CSV (向后兼容)
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    df_translate.to_csv(_4_2_TRANSLATION, index=False, encoding='utf-8-sig')
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

    return sentences

if __name__ == '__main__':
    translate_all()