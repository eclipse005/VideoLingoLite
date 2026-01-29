import os
import warnings
from typing import List
from core.utils import rprint, load_key, get_joiner
from core.spacy_utils.load_nlp_model import init_nlp, SPLIT_BY_CONNECTOR_FILE
from core.utils.models import Sentence
from core.utils.sentence_tools import map_char_positions_to_chunks, should_split_by_origin_length

warnings.filterwarnings("ignore", category=FutureWarning)

def analyze_connectors(doc, token):
    """
    Analyze whether a token is a connector that should trigger a sentence split.
    """
    lang = load_key("asr.language")
    if lang == "en":
        connectors = ["that", "which", "where", "when", "because", "but", "and", "or"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "zh":
        connectors = ["因为", "所以", "但是", "而且", "虽然", "如果", "即使", "尽管"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "ja":
        connectors = ["けれども", "しかし", "だから", "それで", "ので", "のに", "ため"]
        mark_dep = "mark"
        det_pron_deps = ["case"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "ko":
        connectors = ["하지만", "그러나", "그리고", "그래서", "왜냐하면", "그런데", "또는", "즉", "게다가"]
        mark_dep = "mark"
        det_pron_deps = ["det"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "fr":
        connectors = ["que", "qui", "où", "quand", "parce que", "mais", "et", "ou"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "ru":
        connectors = ["что", "который", "где", "когда", "потому что", "но", "и", "или"]
        mark_dep = "mark"
        det_pron_deps = ["det"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "es":
        connectors = ["que", "cual", "donde", "cuando", "porque", "pero", "y", "o"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "de":
        connectors = ["dass", "welche", "wo", "wann", "weil", "aber", "und", "oder"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    elif lang == "it":
        connectors = ["che", "quale", "dove", "quando", "perché", "ma", "e", "o"]
        mark_dep = "mark"
        det_pron_deps = ["det", "pron"]
        verb_pos = "VERB"
        noun_pos = ["NOUN", "PROPN"]
    else:
        return False, False

    if token.text.lower() not in connectors:
        return False, False

    if lang == "en" and token.text.lower() == "that":
        if token.dep_ == mark_dep and token.head.pos_ == verb_pos:
            return True, False
        else:
            return False, False
    elif token.dep_ in det_pron_deps and token.head.pos_ in noun_pos:
        return False, False
    else:
        return True, False

def split_by_connectors(text, context_words=5, nlp=None):
    doc = nlp(text)
    sentences = [doc.text]

    lang = load_key("asr.language")

    while True:
        split_occurred = False
        new_sentences = []

        for sent in sentences:
            doc = nlp(sent)
            start = 0

            for i, token in enumerate(doc):
                split_before, _ = analyze_connectors(doc, token)

                if lang == "en" and i + 1 < len(doc) and doc[i + 1].text in ["'s", "'re", "'ve", "'ll", "'d"]:
                    continue

                left_words = doc[max(0, token.i - context_words):token.i]
                right_words = doc[token.i+1:min(len(doc), token.i + context_words + 1)]

                left_words = [word.text for word in left_words if not word.is_punct]
                right_words = [word.text for word in right_words if not word.is_punct]

                if len(left_words) >= context_words and len(right_words) >= context_words and split_before:
                    rprint(f"[yellow]✂️  Split before '{token.text}': {' '.join(left_words)}| {token.text} {' '.join(right_words)}[/yellow]")
                    new_sentences.append(doc[start:token.i].text.strip())
                    start = token.i
                    split_occurred = True
                    break

            if start < len(doc):
                new_sentences.append(doc[start:].text.strip())

        if not split_occurred:
            break

        sentences = new_sentences

    return sentences


# ------------
# Original file-based function (deprecated)
# ------------

def split_sentences_main(nlp):
    """旧的文件流版本（已弃用）"""
    from core.spacy_utils.load_nlp_model import SPLIT_BY_COMMA_FILE

    with open(SPLIT_BY_COMMA_FILE, "r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        split_sentences = split_by_connectors(sentence.strip(), nlp=nlp)
        all_split_sentences.extend(split_sentences)

    with open(SPLIT_BY_CONNECTOR_FILE, "w+", encoding="utf-8") as output_file:
        for sentence in all_split_sentences:
            output_file.write(sentence + "\n")
        output_file.seek(output_file.tell() - 1, os.SEEK_SET)
        output_file.truncate()

    rprint(f"[green]💾 Sentences split by connectors saved to →  `{SPLIT_BY_CONNECTOR_FILE}`[/green]")


# ------------
# New Object-based Function
# ------------

def split_by_connector(sentences: List[Sentence], nlp) -> List[Sentence]:
    """
    按连接词分句（对象化版本）

    使用与 _3_1_split_nlp.py 相同的字符位置映射逻辑。

    Args:
        sentences: Sentence 对象列表
        nlp: spaCy NLP 模型

    Returns:
        List[Sentence]: 分句后的 Sentence 对象列表
    """
    lang = load_key("asr.language")
    context_words = 5
    result = []

    for sentence in sentences:
        # 对每个句子进行可能的多次切分
        current_sentences = [sentence]

        while True:
            split_occurred = False
            new_sentences = []

            for sent in current_sentences:
                if len(sent.chunks) <= 1:
                    new_sentences.append(sent)
                    continue

                if not should_split_by_origin_length(sent.text):
                    new_sentences.append(sent)
                    continue

                doc = nlp(sent.text)
                split_char_pos = None

                for i, token in enumerate(doc):
                    # 检查是否为连接词
                    is_connector, _ = analyze_connectors(doc, token)

                    # 跳过英语缩写后的连接词
                    if lang == "en" and i + 1 < len(doc) and doc[i + 1].text in ["'s", "'re", "'ve", "'ll", "'d"]:
                        continue

                    left_words = doc[max(0, token.i - context_words):token.i]
                    right_words = doc[token.i+1:min(len(doc), token.i + context_words + 1)]

                    left_words_list = [word.text for word in left_words if not word.is_punct]
                    right_words_list = [word.text for word in right_words if not word.is_punct]

                    if (len(left_words_list) >= context_words and
                        len(right_words_list) >= context_words and
                        is_connector):

                        # 使用 token.idx（原始文本中的字符位置）
                        split_char_pos = token.idx
                        rprint(f"[yellow]✂️  Split before '{token.text}': {' '.join(left_words_list[-4:])}| {token.text} {' '.join(right_words_list[:4])}[/yellow]")
                        split_occurred = True
                        break

                if split_char_pos is not None:
                    # 使用公共函数映射到 Chunk 索引
                    chunk_split_indices = map_char_positions_to_chunks(sent, [split_char_pos])

                    if chunk_split_indices and 0 < chunk_split_indices[0] < len(sent.chunks):
                        split_idx = chunk_split_indices[0]
                        left_chunks = sent.chunks[:split_idx]
                        right_chunks = sent.chunks[split_idx:]

                        if left_chunks and right_chunks:
                            joiner = get_joiner(lang)
                            left_sentence = Sentence(
                                chunks=left_chunks,
                                text=joiner.join(c.text for c in left_chunks),
                                start=left_chunks[0].start,
                                end=left_chunks[-1].end,
                                is_split=True
                            )
                            right_sentence = Sentence(
                                chunks=right_chunks,
                                text=joiner.join(c.text for c in right_chunks),
                                start=right_chunks[0].start,
                                end=right_chunks[-1].end,
                                is_split=True
                            )
                            new_sentences.extend([left_sentence, right_sentence])
                        else:
                            new_sentences.append(sent)
                    else:
                        new_sentences.append(sent)
                else:
                    new_sentences.append(sent)

            if not split_occurred:
                break

            current_sentences = new_sentences

        result.extend(current_sentences)

    return result


if __name__ == "__main__":
    nlp = init_nlp()
    split_sentences_main(nlp)
