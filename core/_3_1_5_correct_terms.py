"""
ASR 术语矫正模块 (Stage 1.5)

在 NLP 分句后进行，使用 LLM Agent 智能识别并矫正 ASR 错误

输入: List[Sentence] (来自 _3_1_split_nlp.py)
输出: List[Sentence] (矫正后，传递给 _3_2_split_meaning.py)
"""

import json
import re
from typing import List, Tuple, Dict, Any, Optional

from core.utils import load_key
from core.utils.models import Sentence
from core.utils.sentence_tools import clean_word


class SentenceToolExecutor:
    """Sentence 对象的工具执行器"""

    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences
        self.changes: List[Dict] = []

    def read_sentences(self, start_idx: int = None, end_idx: int = None) -> str:
        """读取句子内容，返回全部或指定索引范围"""
        total = len(self.sentences)

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = total

        # 限制最多显示 100 句
        MAX_DISPLAY = 100
        truncated = False
        actual_end = end_idx

        if actual_end - start_idx > MAX_DISPLAY:
            actual_end = start_idx + MAX_DISPLAY
            truncated = True

        result = []
        for i in range(start_idx, min(actual_end, total)):
            result.append(f"第{i}句: {self.sentences[i].text}")

        output = "\n".join(result)

        if truncated:
            remaining = total - actual_end
            output += f"\n\n[共 {total} 句，已显示到第 {actual_end} 句，剩余 {remaining} 句。"
            output += f"请调用 read_sentences(start_idx={actual_end}) 继续查看]"

        return output

    def get_context(self, sentence_idx: int, context_count: int = 2) -> str:
        """获取某句子的上下文"""
        start = max(0, sentence_idx - context_count)
        end = min(len(self.sentences), sentence_idx + context_count + 1)

        result = []
        for i in range(start, end):
            marker = ">>> " if i == sentence_idx else "    "
            result.append(f"{marker}第{i}句: {self.sentences[i].text}")

        return "\n".join(result)

    def batch_replace(self, replacements: list) -> str:
        """批量替换术语（只修改文本，不动 chunks 和时间戳）"""
        results = []
        total_changes = 0

        for replacement in replacements:
            old_text = replacement.get("old_text", "")
            new_text = replacement.get("new_text", "")

            if not old_text:
                results.append({"error": "old_text 不能为空"})
                continue

            # 在所有句子中查找并替换
            for sent_idx, sentence in enumerate(self.sentences):
                changes_count = self._replace_in_sentence(
                    sentence, sent_idx, old_text, new_text
                )
                total_changes += changes_count

            results.append({
                "old_text": old_text,
                "new_text": new_text,
                "count": total_changes
            })

        return json.dumps({
            "success": True,
            "total_changes": total_changes,
            "details": results
        }, ensure_ascii=False)

    def _replace_in_sentence(
        self, sentence: Sentence, sent_idx: int,
        old_text: str, new_text: str
    ) -> int:
        """在单个句子中查找并替换（只修改文本）"""
        # 清洗文本用于匹配
        sent_clean = clean_word(sentence.text)
        old_clean = clean_word(old_text)

        # 查找所有匹配位置
        matches = list(re.finditer(re.escape(old_clean), sent_clean))
        if not matches:
            return 0

        # 在原始文本中进行替换（从后往前避免位置偏移）
        changes_count = 0
        for match in reversed(matches):
            # 在原始文本中找到对应位置
            original_start, original_end = self._find_original_position(
                sentence.text, sent_clean, match.start(), match.end()
            )

            if original_start is not None:
                # 执行替换
                sentence.text = (
                    sentence.text[:original_start] +
                    new_text +
                    sentence.text[original_end:]
                )
                changes_count += 1

                # 记录修改
                self.changes.append({
                    "sentence_idx": sent_idx,
                    "old_text": old_text,
                    "new_text": new_text
                })

        return changes_count

    def _find_original_position(
        self, original_text: str, cleaned_text: str,
        clean_start: int, clean_end: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        在原始文本中找到清洗后文本对应的位置

        使用滑动窗口匹配，找到最可能的位置
        """
        # 清洗前后的文本长度可能不同
        # 使用滑动窗口在原始文本中查找匹配
        window_size = clean_end - clean_start

        # 提取清洗后窗口内容
        cleaned_window = cleaned_text[clean_start:clean_end]

        # 在原始文本中搜索
        best_match = None
        best_score = 0

        for i in range(len(original_text) - window_size + 1):
            window = original_text[i:i + window_size]
            # 清洗窗口内容进行对比
            from core.utils.sentence_tools import clean_word
            if clean_word(window) == cleaned_window:
                # 精确匹配
                return i, i + window_size

        # 如果没有精确匹配，返回 None
        return None, None

    def finish(self, summary: str) -> str:
        """完成矫正任务"""
        return json.dumps({
            "changes_count": len(self.changes),
            "summary": summary,
            "is_finish": True
        }, ensure_ascii=False)
