import os
import pandas as pd
import warnings
from typing import List
from spacy.language import Language
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils.config_utils import load_key, get_joiner
from core.utils import safe_read_csv, rprint
from core.utils.models import Chunk, Sentence

warnings.filterwarnings("ignore", category=FutureWarning)

# SudachiPy token length limit (bytes)
SUDACHI_MAX_LENGTH = 40000


# ------------
# Original file-based function (deprecated)
# ------------

def split_by_mark_main(nlp):
    """旧的文件流版本（已弃用）"""
    language = load_key("asr.language")
    joiner = get_joiner(language)
    rprint(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")
    chunks = safe_read_csv("output/log/cleaned_chunks.csv")
    chunks.text = chunks.text.apply(lambda x: x.strip('"').strip(""))

    chunks_list = chunks.text.to_list()

    if language == 'ja':
        rprint("[yellow]⚠️  Japanese detected, processing in chunks to avoid SudachiPy limit...[/yellow]")
        sentences_by_mark = _process_japanese_in_chunks(nlp, chunks_list)
    else:
        input_text = joiner.join(chunks_list)
        sentences_by_mark = _process_batch_text(nlp, input_text)

    # Write to file
    from core.spacy_utils.load_nlp_model import SPLIT_BY_MARK_FILE
    with open(SPLIT_BY_MARK_FILE, "w", encoding="utf-8") as output_file:
        for sentence in sentences_by_mark:
            output_file.write(sentence + "\n")

    rprint(f"[green]💾 Sentences split by punctuation marks saved to →  `{SPLIT_BY_MARK_FILE}`[/green]")

def _process_japanese_in_chunks(nlp, chunks_list):
    """
    Process Japanese text in chunks to avoid SudachiPy length limit.
    Splits at chunk boundaries (natural time-based segments) rather than mid-sentence.
    """
    all_sentences = []

    # Calculate cumulative byte lengths to find safe split points
    chunk_bytes = [len(chunk.encode('utf-8')) for chunk in chunks_list]
    cumulative_bytes = []
    total = 0
    for b in chunk_bytes:
        total += b
        cumulative_bytes.append(total)

    # Find split points: process chunks until approaching limit, then split
    batch_start = 0
    batch_count = 0

    while batch_start < len(chunks_list):
        # Find how many chunks we can safely include in this batch
        batch_end = batch_start
        batch_bytes = 0

        while batch_end < len(chunks_list):
            next_chunk_bytes = chunk_bytes[batch_end]

            # If adding this chunk would exceed limit, stop here
            if batch_bytes + next_chunk_bytes > SUDACHI_MAX_LENGTH and batch_end > batch_start:
                break

            batch_bytes += next_chunk_bytes
            batch_end += 1

        # Process this batch
        batch_chunks = chunks_list[batch_start:batch_end]
        batch_text = ''.join(batch_chunks)

        batch_count += 1
        rprint(f"[dim]  Processing batch {batch_count}: {len(batch_chunks)} chunks, {batch_bytes} bytes[/dim]")

        try:
            sentences = _process_batch_text(nlp, batch_text)
            all_sentences.extend(sentences)
        except Exception as e:
            # If still fails, reduce batch size by half and retry
            if len(batch_chunks) > 1:
                rprint(f"[yellow]  Batch too large, splitting further...[/yellow]")
                mid = len(batch_chunks) // 2
                # Process first half
                batch_text1 = ''.join(batch_chunks[:mid])
                sentences1 = _process_batch_text(nlp, batch_text1)
                all_sentences.extend(sentences1)
                # Process second half
                batch_text2 = ''.join(batch_chunks[mid:])
                sentences2 = _process_batch_text(nlp, batch_text2)
                all_sentences.extend(sentences2)
            else:
                raise e

        # Move to next batch
        batch_start = batch_end

    return all_sentences

def _process_batch_text(nlp, text):
    """Process a batch of text through spaCy and split by punctuation marks"""
    language = load_key("asr.language")
    joiner = get_joiner(language)

    doc = nlp(text)
    assert doc.has_annotation("SENT_START")

    # skip - and ...
    sentences_by_mark = []
    current_sentence = []

    # iterate all sentences
    for sent in doc.sents:
        sent_text = sent.text.strip()

        # check if the current sentence ends with - or ...
        if current_sentence and (
            sent_text.startswith('-') or
            sent_text.startswith('...') or
            current_sentence[-1].endswith('-') or
            current_sentence[-1].endswith('...')
        ):
            current_sentence.append(sent_text)
        else:
            if current_sentence:
                sentences_by_mark.append(joiner.join(current_sentence))
                current_sentence = []
            current_sentence.append(sent_text)

    # add the last sentence
    if current_sentence:
        sentences_by_mark.append(joiner.join(current_sentence))

    # FIX: Merge standalone punctuation marks with previous sentence
    # This handles spaCy bug where it splits "text。" into "text" and "。"
    # Also handles Japanese quotes 「 and 」
    i = 0
    while i < len(sentences_by_mark):
        if i > 0 and sentences_by_mark[i] in [',', '.', '，', '。', '？', '！']:
            # Merge with previous sentence
            sentences_by_mark[i-1] += sentences_by_mark[i]
            sentences_by_mark.pop(i)
        else:
            i += 1

    return sentences_by_mark


# ------------
# New Object-based Function
# ------------

def split_by_mark(chunks: List[Chunk], nlp: Language) -> List[Sentence]:
    """
    按标点分句（对象化版本）

    Args:
        chunks: Chunk 对象列表
        nlp: spaCy NLP 模型

    Returns:
        List[Sentence]: 分句后的 Sentence 对象列表
    """
    from core._3_1_split_nlp import nlp_split_to_sentences
    return nlp_split_to_sentences(chunks, nlp)


if __name__ == "__main__":
    nlp = init_nlp()
    split_by_mark(nlp)
