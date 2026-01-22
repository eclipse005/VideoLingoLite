import os
import string
import warnings
from typing import List, Tuple
from core.utils import rprint, load_key, get_joiner
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils.models import _3_1_SPLIT_BY_NLP, Sentence
from core.utils.sentence_tools import map_char_positions_to_chunks

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------
# Helper Functions
# ------------

def split_long_sentence(doc):
    """
    使用动态规划切分长句

    Returns:
        List[int]: 切分点的 token 索引列表（升序）
    """
    tokens = [token.text for token in doc]
    n = len(tokens)

    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    prev = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(max(0, i - 100), i):
            if i - j >= 30:
                token = doc[i-1]
                if j == 0 or (token.is_sent_end or token.pos_ in ['VERB', 'AUX'] or token.dep_ == 'ROOT'):
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j

    # 回溯获取切分点
    split_points = []
    i = n
    while i > 0:
        j = prev[i]
        if j > 0:
            split_points.append(j)
        i = j

    return split_points[::-1]  # 返回升序

def split_extremely_long_sentence(doc):
    tokens = [token.text for token in doc]
    n = len(tokens)

    num_parts = (n + 59) // 60
    part_length = n // num_parts

    sentences = []
    language = load_key("asr.language")
    joiner = get_joiner(language)
    for i in range(num_parts):
        start = i * part_length
        end = start + part_length if i < num_parts - 1 else n
        sentence = joiner.join(tokens[start:end])
        sentences.append(sentence)

    return sentences


# ------------
# Original file-based function (deprecated)
# ------------

def split_long_by_root_main(nlp):
    """旧的文件流版本（已弃用）"""
    from core.spacy_utils.load_nlp_model import SPLIT_BY_CONNECTOR_FILE

    with open(SPLIT_BY_CONNECTOR_FILE, "r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        doc = nlp(sentence.strip())
        if len(doc) > 60:
            split_sentences = split_long_sentence(doc)
            if any(len(nlp(sent)) > 60 for sent in split_sentences):
                split_sentences = [subsent for sent in split_sentences for subsent in split_extremely_long_sentence(nlp(sent))]
            all_split_sentences.extend(split_sentences)
            rprint(f"[yellow]✂️  Splitting long sentences by root: {sentence[:30]}...[/yellow]")
        else:
            all_split_sentences.append(sentence.strip())

    punctuation = string.punctuation + "'" + '"'

    with open(_3_1_SPLIT_BY_NLP, "w", encoding="utf-8") as output_file:
        for i, sentence in enumerate(all_split_sentences):
            stripped_sentence = sentence.strip()
            if not stripped_sentence or all(char in punctuation for char in stripped_sentence):
                rprint(f"[yellow]⚠️  Warning: Empty or punctuation-only line detected at index {i}[/yellow]")
                continue
            output_file.write(sentence + "\n")

    rprint(f"[green]💾 Long sentences split by root saved to →  {_3_1_SPLIT_BY_NLP}[/green]")


# ------------
# New Object-based Function
# ------------

def split_by_root(sentences: List[Sentence], nlp) -> List[Sentence]:
    """
    按词根切分长句（对象化版本）

    使用与 _3_1_split_nlp.py 相同的字符位置映射逻辑。

    Args:
        sentences: Sentence 对象列表
        nlp: spaCy NLP 模型

    Returns:
        List[Sentence]: 分句后的 Sentence 对象列表
    """
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)
    result = []

    for sentence in sentences:
        if len(sentence.chunks) <= 1:
            result.append(sentence)
            continue

        doc = nlp(sentence.text)

        # 只处理超过 60 个 token 的句子
        if len(doc) <= 60:
            result.append(sentence)
            continue

        # 使用 DP 算法找到切分点（token 索引）
        token_split_indices = split_long_sentence(doc)

        if not token_split_indices:
            result.append(sentence)
            continue

        # 将 token 索引转换为字符位置
        char_positions = []
        for token_idx in token_split_indices:
            if token_idx < len(doc):
                char_positions.append(doc[token_idx].idx)

        # 使用公共函数映射到 Chunk 索引
        chunk_split_indices = map_char_positions_to_chunks(sentence, char_positions)

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

        rprint(f"[yellow]✂️  Splitting long sentence by root: {sentence.text[:30]}...[/yellow]")

    return result


if __name__ == "__main__":
    nlp = init_nlp()
    split_long_by_root_main(nlp)
