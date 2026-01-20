import os
import pandas as pd
import difflib
import unicodedata
from core.utils.config_utils import load_key, get_joiner
from core.utils import safe_read_csv
from core.spacy_utils.load_nlp_model import SPLIT_BY_NLP_FILE, SPLIT_BY_PAUSE_FILE
from core.utils.sentence_tools import clean_word
from rich import print as rprint
from core.utils.models import _2_CLEANED_CHUNKS


def is_latin_text(text):
    """检测文本是否是拉丁字母或数字（包括拉丁字符集）"""
    if not text:
        return False

    for char in text:
        code = ord(char)
        # Basic Latin (0x0020-0x007F): 空格, a-z, A-Z, 0-9, 常见标点
        # Latin-1 Supplement (0x0080-0x00FF): è, é, ê, ë, à, â, ä, 等
        # Latin Extended-A (0x0100-0x017F): Ā, ā, Ă, ă, Ą, ą, 等
        # Latin Extended-B (0x0180-0x024F): ƀ, Ƀ, ɂ, 等
        if not (0x0020 <= code <= 0x024F):
            return False
    return True


def smart_join(chunks):
    """智能连接 chunks：如果相邻两个都是拉丁字母，用空格连接"""
    if not chunks:
        return ""

    result = str(chunks[0]).strip('"')
    for i in range(1, len(chunks)):
        prev = result
        curr = str(chunks[i]).strip('"')

        # 检测前一个文本的结尾和当前文本的开头是否都是拉丁字母
        if prev and curr:
            prev_last = prev[-1] if prev else ""
            curr_first = curr[0] if curr else ""

            if is_latin_text(prev_last) and is_latin_text(curr_first):
                result += " " + curr
            else:
                result += curr
        else:
            result += curr

    return result


def split_by_pause():
    # Input/output paths
    input_nlp_path = SPLIT_BY_NLP_FILE
    chunks_path = _2_CLEANED_CHUNKS
    output_pause_path = SPLIT_BY_NLP_FILE

    # Get pause threshold from config (0 or null means disabled)
    pause_threshold = load_key("pause_split_threshold")
    if pause_threshold is None or pause_threshold == 0:
        rprint("[yellow]⏭️  Pause-based splitting is disabled (pause_split_threshold=0 or null)[/yellow]")
        return

    # Get language and joiner from config
    language = load_key("asr.language")
    joiner = get_joiner(language)
    rprint(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")
    rprint(f"[blue]🔍 Pause threshold: {pause_threshold}s[/blue]")

    print(f"🚀 Starting pause-based splitting...")

    if not os.path.exists(input_nlp_path) or not os.path.exists(chunks_path):
        print(f"❌ Error: Files not found. Please ensure:\n- {input_nlp_path}\n- {chunks_path}")
        return

    # Load data
    chunks_df = safe_read_csv(chunks_path)
    with open(input_nlp_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    print(f"📊 Loaded {len(raw_lines)} lines of text and {len(chunks_df)} word chunks with timestamps.")

    # Build character-to-chunk mapping
    chunk_char_map = []
    for idx, row in chunks_df.iterrows():
        chunk_text = str(row['text']).strip('"')
        if joiner == " ":
            cleaned = unicodedata.normalize('NFC', "".join(chunk_text.split()).lower())
        else:
            cleaned = unicodedata.normalize('NFC', chunk_text.lower())
        for char in cleaned:
            chunk_char_map.append({'char': char, 'chunk_idx': idx})

    # Extract pure character list for difflib matching
    all_chars = [item['char'] for item in chunk_char_map]

    final_sentences = []
    current_char_pos = 0

    # Process each sentence
    for sentence in raw_lines:
        if not sentence.strip():
            continue

        # Prepare sentence for matching
        if joiner == " ":
            sentence_chars = list(unicodedata.normalize('NFC', "".join(sentence.split()).lower()))
        else:
            sentence_chars = list(unicodedata.normalize('NFC', sentence.lower()))

        if not sentence_chars:
            continue

        # Match using difflib
        remaining_chars = all_chars[current_char_pos:]
        s = difflib.SequenceMatcher(None, remaining_chars, sentence_chars, autojunk=False)
        matching_blocks = s.get_matching_blocks()

        # Find matching block that starts at sentence beginning
        match_start_rel_idx = -1
        match_length = 0
        for a_start, b_start, length in matching_blocks:
            if b_start == 0:
                match_start_rel_idx = a_start
                match_length = length
                break

        if match_start_rel_idx == -1:
            print(f"\n⚠️ Warning: No match found for sentence: {sentence}")
            final_sentences.append(sentence)
            continue

        # Calculate character positions
        absolute_start_char_idx = current_char_pos + match_start_rel_idx
        absolute_end_char_idx = absolute_start_char_idx + match_length - 1

        # Boundary checks
        max_char_idx = len(chunk_char_map) - 1
        absolute_end_char_idx = min(absolute_end_char_idx, max_char_idx)
        absolute_start_char_idx = min(absolute_start_char_idx, max_char_idx)
        if absolute_end_char_idx < absolute_start_char_idx:
            absolute_end_char_idx = absolute_start_char_idx

        # Get chunk range
        start_chunk_idx = chunk_char_map[absolute_start_char_idx]['chunk_idx']
        end_chunk_idx = chunk_char_map[absolute_end_char_idx]['chunk_idx']

        # Get chunks for this sentence
        sentence_chunks = chunks_df.iloc[start_chunk_idx:end_chunk_idx + 1]

        # If sentence has less than 2 chunks, cannot check for pauses
        if len(sentence_chunks) < 2:
            final_sentences.append(sentence)
            current_char_pos = absolute_end_char_idx + 1
            if current_char_pos >= len(all_chars):
                current_char_pos = len(all_chars) - 1
            continue

        # Check for physical pauses and split if needed
        temp_group = []
        last_chunk = sentence_chunks.iloc[0]
        temp_group.append(str(last_chunk['text']).strip('"'))

        for i in range(1, len(sentence_chunks)):
            current_chunk = sentence_chunks.iloc[i]

            # Calculate gap
            gap = current_chunk['start'] - last_chunk['end']

            if gap > pause_threshold:
                # Gap too large, split
                final_sentences.append(smart_join(temp_group))
                print(f"✂️  Detected pause {gap:.2f}s, split after '{last_chunk['text']}'")
                temp_group = [str(current_chunk['text']).strip('"')]
            else:
                temp_group.append(str(current_chunk['text']).strip('"'))

            last_chunk = current_chunk

        if temp_group:
            final_sentences.append(smart_join(temp_group))

        # Update position
        current_char_pos = absolute_end_char_idx + 1
        if current_char_pos >= len(all_chars):
            current_char_pos = len(all_chars) - 1

    # Write output
    with open(output_pause_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(final_sentences))

    print(f"\n✨ Processing complete!")
    print(f"📝 Original lines: {len(raw_lines)} -> After split: {len(final_sentences)}")
    print(f"💾 Saved to: {output_pause_path}")


if __name__ == '__main__':
    split_by_pause()
