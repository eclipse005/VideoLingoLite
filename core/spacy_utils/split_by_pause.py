"""
停顿分句模块（对象化版本）
"""
import pandas as pd
import warnings
from typing import List
from core.utils.config_utils import load_key
from core.utils import rprint
from core.utils.models import Sentence

warnings.filterwarnings("ignore", category=FutureWarning)


# ------------
# Split by Pause Function
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
    # 如果停顿切分未启用，直接返回
    if pause_threshold <= 0:
        return sentences

    # 检查句子内部的停顿并切分
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
