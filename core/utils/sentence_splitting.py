"""
句子拆分工具函数

用于 ASR 后端按标点符号分句
"""

import unicodedata
from typing import List, Tuple, Dict


def is_sentence_terminator(char: str) -> bool:
    """
    判断字符是否为句子结束符号（使用 Unicode 类别）

    涵盖多语言：
    - 中文/日文：。！？
    - 英文：.!?
    - 其他语言的句子结束符号
    """
    if not char:
        return False

    # 常见句子结束符号
    terminators = {'.', '!', '?', '。', '！', '？', '‼', '⁇', '⁈', '⁉'}
    if char in terminators:
        return True

    # 使用 Unicode 类别判断
    # Po: Other punctuation（包含大多数标点符号）
    category = unicodedata.category(char)
    if category == 'Po':
        # 进一步过滤，只保留句子结束类的标点
        # 排除逗号、顿号、引号等非句子结束符号
        non_terminators = {
            ',', '，', '、', ';', '；', ':', '：',
            '"', "'", '「', '」', '『', '』',
            '（', '）', '(', ')', '[', ']', '{', '}',
            '・', '·', '•'
        }
        if char not in non_terminators:
            return True

    return False


def group_words_into_sentences_tuples(word_response: List[Tuple]) -> List[List[Tuple]]:
    """
    根据句子结束符号将 words 分组为句子（tuple 格式）

    Args:
        word_response: [(word, start, end), ...]

    Returns:
        List of sentence groups, each group is a list of (word, start, end)
    """
    if not word_response:
        return []

    sentences = []
    current_sentence = []

    for word_info in word_response:
        word = word_info[0]  # tuple: (word, start, end)
        current_sentence.append(word_info)

        # 检查该词是否包含句子结束符号
        if word:
            for char in word:
                if is_sentence_terminator(char):
                    sentences.append(current_sentence)
                    current_sentence = []
                    break

    # 处理剩余的词
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


def group_words_into_sentences_dicts(word_response: List[Dict]) -> List[List[Dict]]:
    """
    根据句子结束符号将 words 分组为句子（dict 格式）

    Args:
        word_response: [{'word': str, 'start': float, 'end': float}, ...]

    Returns:
        List of sentence groups, each group is a list of words
    """
    if not word_response:
        return []

    sentences = []
    current_sentence = []

    for word_info in word_response:
        word = word_info['word']
        current_sentence.append(word_info)

        # 检查该词是否包含句子结束符号
        if word:
            for char in word:
                if is_sentence_terminator(char):
                    sentences.append(current_sentence)
                    current_sentence = []
                    break

    # 处理剩余的词
    if current_sentence:
        sentences.append(current_sentence)

    return sentences
