"""
停顿分句模块（对象化版本）
"""
import pandas as pd
import warnings
from typing import List
from core.utils.config_utils import load_key
from core.utils import rprint
from core.utils.models import Sentence, Chunk

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------
# Fix Abnormal Word Timestamps
# ------------

def fix_abnormal_words_in_sentence(sentence: Sentence, max_duration: float = 2.0) -> Sentence:
    """
    修正句子中首尾的异常长词

    问题：ASR 可能将静音段算到词的时间戳中，导致某些词时长异常（如 >2 秒）
    解决：只处理句首/句尾的异常词，用同句其他词的平均时长作为基准

    Args:
        sentence: Sentence 对象
        max_duration: 最大合理时长（秒），超过此值认为是异常

    Returns:
        修正后的 Sentence 对象
    """
    if not sentence.chunks:
        return sentence

    # 计算这句话所有词的平均时长（排除异常词后再算）
    normal_durations = []
    for chunk in sentence.chunks:
        duration = chunk.end - chunk.start
        if duration <= max_duration:
            normal_durations.append(duration)

    # 计算平均时长，如果都是异常词则使用默认值
    if normal_durations:
        avg_duration = sum(normal_durations) / len(normal_durations)
    else:
        avg_duration = 0.5  # 默认值

    # 修正异常长词
    for i, chunk in enumerate(sentence.chunks):
        duration = chunk.end - chunk.start

        if duration > max_duration:
            is_first = (i == 0)
            is_last = (i == len(sentence.chunks) - 1)

            if is_first:
                # 句首词：start 被前拖（包含前面的静音），修正 start
                original_start = chunk.start
                chunk.start = chunk.end - avg_duration
                rprint(f"[yellow]修正句首词 '{chunk.text}': "
                       f"{duration:.2f}s → {avg_duration:.2f}s "
                       f"(start {original_start:.2f}s → {chunk.start:.2f}s)[/yellow]")

            elif is_last:
                # 句尾词：end 被后拖（包含后面的静音），修正 end
                original_end = chunk.end
                chunk.end = chunk.start + avg_duration
                rprint(f"[yellow]修正句尾词 '{chunk.text}': "
                       f"{duration:.2f}s → {avg_duration:.2f}s "
                       f"(end {original_end:.2f}s → {chunk.end:.2f}s)[/yellow]")

            else:
                # 句中词：情况复杂，暂不处理，仅记录
                rprint(f"[dim]⚠️ 句中异常词 '{chunk.text}': {duration:.2f}s (跳过修正)[/dim]")

    # 更新句子的时间戳
    sentence.update_timestamps()
    return sentence


def fix_abnormal_words(sentences: List[Sentence], max_duration: float = 2.0) -> List[Sentence]:
    """
    批量修正句子中的异常长词

    Args:
        sentences: Sentence 对象列表
        max_duration: 最大合理时长（秒）

    Returns:
        修正后的 Sentence 对象列表
    """
    result = []
    fixed_count = 0

    for sentence in sentences:
        fixed = fix_abnormal_words_in_sentence(sentence, max_duration)
        result.append(fixed)

        # 统计修正数量
        for chunk in sentence.chunks:
            if chunk.end - chunk.start > max_duration:
                fixed_count += 1

    if fixed_count > 0:
        rprint(f"[cyan]🔧 共修正 {fixed_count} 个异常长词[/cyan]")

    return result


# ------------
# New Object-based Function
# ------------

def split_by_pause(sentences: List[Sentence], pause_threshold: float) -> List[Sentence]:
    """
    按停顿切分（对象化版本）

    检查每个句子内部 chunks 之间的停顿，如果停顿超过阈值则切分句子

    Args:
        sentences: Sentence 对象列表
        pause_threshold: 停顿阈值（秒），超过此值则切分

    Returns:
        List[Sentence]: 切分后的 Sentence 对象列表
    """
    # Step 1: 先修复异常长词（始终执行）
    sentences = fix_abnormal_words(sentences, max_duration=2.0)

    # Step 2: 如果停顿切分未启用，直接返回
    if pause_threshold <= 0:
        return sentences

    # Step 3: 检查句子内部的停顿并切分
    result = []
    split_count = 0

    for sentence in sentences:
        # 单词句子，无法检查间隔，直接保留
        if len(sentence.chunks) < 2:
            result.append(sentence)
            continue

        # 找到需要切分的位置
        split_positions = []
        for i in range(1, len(sentence.chunks)):
            current_chunk = sentence.chunks[i]
            last_chunk = sentence.chunks[i - 1]
            gap = current_chunk.start - last_chunk.end

            if gap > pause_threshold:
                split_positions.append(i)
                rprint(f"[cyan]✂️ 检测到停顿 {gap:.2f}s > {pause_threshold}s，"
                       f"在 '{last_chunk.text}' 后切分[/cyan]")

        # 无需切分
        if not split_positions:
            result.append(sentence)
            continue

        # 执行切分
        start_idx = 0
        for split_idx in split_positions:
            # 创建新句子（从 start_idx 到 split_idx-1）
            new_chunks = sentence.chunks[start_idx:split_idx]
            new_sentence = Sentence(
                chunks=new_chunks,
                text=''.join(c.text for c in new_chunks),
                start=new_chunks[0].start,
                end=new_chunks[-1].end,
                translation=sentence.translation,
                is_split=True
            )
            result.append(new_sentence)
            split_count += 1
            start_idx = split_idx

        # 添加最后一部分
        if start_idx < len(sentence.chunks):
            new_chunks = sentence.chunks[start_idx:]
            new_sentence = Sentence(
                chunks=new_chunks,
                text=''.join(c.text for c in new_chunks),
                start=new_chunks[0].start,
                end=new_chunks[-1].end,
                translation=sentence.translation,
                is_split=True
            )
            result.append(new_sentence)

    if split_count > 0:
        rprint(f"[green]✅ 停顿切分完成：{split_count} 个句子被切分[/green]")

    return result
