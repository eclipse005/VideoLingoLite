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
# New Object-based Function
# ------------

def split_by_pause(sentences: List[Sentence], pause_threshold: float) -> List[Sentence]:
    """
    按停顿切分（对象化版本）

    根据句子之间的停顿时间进行分批处理

    Args:
        sentences: Sentence 对象列表
        pause_threshold: 停顿阈值（秒），超过此值则分批

    Returns:
        List[Sentence]: 分批后的 Sentence 对象列表
    """
    if pause_threshold <= 0:
        return sentences

    result = []
    current_batch = []

    for sent in sentences:
        if not current_batch:
            current_batch.append(sent)
            continue

        # 计算当前句子与上一个句子之间的停顿
        last_sent = current_batch[-1]
        gap = sent.start - last_sent.end

        if gap > pause_threshold:
            # 停顿超过阈值，结束当前批次
            result.extend(current_batch)
            current_batch = [sent]
        else:
            current_batch.append(sent)

    if current_batch:
        result.extend(current_batch)

    return result
