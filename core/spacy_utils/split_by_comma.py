import itertools
import os
import warnings
from typing import List
from core.utils import rprint, load_key, get_joiner
from core.spacy_utils.load_nlp_model import init_nlp, SPLIT_BY_COMMA_FILE
from core.utils.models import Sentence
from core.utils.sentence_tools import map_char_positions_to_chunks, should_split_by_origin_length

warnings.filterwarnings("ignore", category=FutureWarning)

def is_valid_phrase(phrase):
    # 🔍 Check for subject and verb
    has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] or token.pos_ == "PRON" for token in phrase)
    has_verb = any((token.pos_ == "VERB" or token.pos_ == 'AUX') for token in phrase)
    return (has_subject and has_verb)

def analyze_comma(start, doc, token):
    left_phrase = doc[max(start, token.i - 9):token.i]
    right_phrase = doc[token.i + 1:min(len(doc), token.i + 10)]

    suitable_for_splitting = is_valid_phrase(right_phrase)

    # 🚫 Remove punctuation and check word count
    left_words = [t for t in left_phrase if not t.is_punct]
    right_words = list(itertools.takewhile(lambda t: not t.is_punct, right_phrase))

    if len(left_words) <= 3 or len(right_words) <= 3:
        suitable_for_splitting = False

    return suitable_for_splitting

def split_by_comma(text, nlp):
    doc = nlp(text)
    sentences = []
    start = 0

    for i, token in enumerate(doc):
        if token.text == "," or token.text == "，":
            suitable_for_splitting = analyze_comma(start, doc, token)

            if suitable_for_splitting:
                sentences.append(doc[start:token.i].text.strip())
                rprint(f"[yellow]✂️  Split at comma: {doc[start:token.i][-4:]},| {doc[token.i + 1:][:4]}[/yellow]")
                start = token.i + 1

    sentences.append(doc[start:].text.strip())
    return sentences


# ------------
# Original file-based function (deprecated)
# ------------

def split_by_comma_main(nlp):
    """旧的文件流版本（已弃用）"""
    with open(SPLIT_BY_COMMA_FILE, "r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        split_sentences = split_by_comma(sentence.strip(), nlp)
        all_split_sentences.extend(split_sentences)

    with open(SPLIT_BY_COMMA_FILE, "w", encoding="utf-8") as output_file:
        for sentence in all_split_sentences:
            output_file.write(sentence + "\n")

    rprint(f"[green]💾 Sentences split by commas saved to →  `{SPLIT_BY_COMMA_FILE}`[/green]")


# ------------
# New Object-based Function
# ------------

def split_by_comma(sentences: List[Sentence], nlp) -> List[Sentence]:
    """
    按逗号分句（对象化版本）

    使用与 _3_1_split_nlp.py 相同的字符位置映射逻辑。

    Args:
        sentences: Sentence 对象列表
        nlp: spaCy NLP 模型

    Returns:
        List[Sentence]: 分句后的 Sentence 对象列表
    """
    result = []
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)

    for sentence in sentences:
        if len(sentence.chunks) <= 1:
            result.append(sentence)
            continue

        if not should_split_by_origin_length(sentence.text):
            result.append(sentence)
            continue

        doc = nlp(sentence.text)
        split_char_positions = []

        # 查找所有适合切分的逗号位置（收集原始文本中的字符位置）
        for token in doc:
            if token.text == "," or token.text == "，":
                left_start = max(0, token.i - 9)
                left_phrase = doc[left_start:token.i]
                right_phrase = doc[token.i + 1:min(len(doc), token.i + 10)]

                if is_valid_phrase(right_phrase):
                    left_words = [t for t in left_phrase if not t.is_punct]
                    right_words = list(itertools.takewhile(lambda t: not t.is_punct, right_phrase))

                    if len(left_words) > 3 and len(right_words) > 3:
                        # 使用 token.idx（原始文本中的字符位置）
                        split_char_positions.append(token.idx)
                        rprint(f"[yellow]✂️  Split at comma: {left_phrase[-4:]}| {right_phrase[:4]}[/yellow]")

        if not split_char_positions:
            result.append(sentence)
        else:
            # 使用公共函数映射到 Chunk 索引
            chunk_split_indices = map_char_positions_to_chunks(sentence, split_char_positions)

            chunk_split_indices = [idx + 1 for idx in chunk_split_indices]

            # 按切分点分割 chunks
            split_points = [0] + chunk_split_indices + [len(sentence.chunks)]
            split_points = sorted(set(split_points))

            for i in range(len(split_points) - 1):
                start_idx = split_points[i]
                end_idx = split_points[i + 1]

                if start_idx >= end_idx:
                    continue

                sub_chunks = sentence.chunks[start_idx:end_idx]
                if sub_chunks:
                    new_sentence = Sentence(
                        chunks=sub_chunks,
                        text=joiner.join(c.text for c in sub_chunks),
                        start=sub_chunks[0].start,
                        end=sub_chunks[-1].end,
                        is_split=True
                    )
                    result.append(new_sentence)

    return result


if __name__ == "__main__":
    nlp = init_nlp()
    split_by_comma_main(nlp)
