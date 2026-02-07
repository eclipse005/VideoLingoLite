"""
ASR 术语矫正模块 (Stage 1.5)

在 NLP 分句后进行，使用 LLM Agent 智能识别并矫正 ASR 错误

输入: List[Sentence] (来自 _3_1_split_nlp.py)
输出: List[Sentence] (矫正后，传递给 _3_2_split_meaning.py)
"""

import json
import re
from typing import List, Tuple, Dict, Any
from difflib import SequenceMatcher

from core.utils import rprint, load_key, get_joiner, timer
from core.utils.models import Sentence, Chunk
from core.utils.sentence_tools import clean_word
from core.utils.ask_gpt import ask_gpt_with_tools


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
        """批量替换术语，智能重新分配时间戳"""
        results = []
        total_changes = 0
        asr_language = load_key("asr.language")
        joiner = get_joiner(asr_language)

        for replacement in replacements:
            old_text = replacement.get("old_text", "")
            new_text = replacement.get("new_text", "")

            if not old_text:
                results.append({"error": "old_text 不能为空"})
                continue

            # 在所有句子中查找并替换
            for sent_idx, sentence in enumerate(self.sentences):
                # 使用 _find_and_replace_in_sentence 处理单个句子
                changes_count = self._find_and_replace_in_sentence(
                    sentence, sent_idx, old_text, new_text, joiner
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

    def _find_and_replace_in_sentence(
        self, sentence: Sentence, sent_idx: int,
        old_text: str, new_text: str, joiner: str
    ) -> int:
        """在单个句子中查找并替换"""
        # 清洗文本用于匹配
        sent_clean = clean_word(sentence.text)
        old_clean = clean_word(old_text)

        # 查找所有匹配位置
        matches = list(re.finditer(re.escape(old_clean), sent_clean))
        if not matches:
            return 0

        # 字符位置到 Chunk 索引的映射
        char_to_chunk = []
        for chunk_idx, chunk in enumerate(sentence.chunks):
            chunk_clean = clean_word(chunk.text)
            char_to_chunk.extend([chunk_idx] * len(chunk_clean))

        changes_count = 0

        # 从后往前替换，避免位置偏移
        for match in reversed(matches):
            start_char = match.start()
            end_char = match.end()

            if start_char >= len(char_to_chunk) or end_char > len(char_to_chunk):
                continue

            start_chunk_idx = char_to_chunk[start_char]
            end_chunk_idx = char_to_chunk[end_char - 1]

            # 提取旧 Chunk
            old_chunks = sentence.chunks[start_chunk_idx:end_chunk_idx + 1]

            # 创建新 Chunk
            new_chunks = self._create_new_chunks(new_text, old_chunks)

            # 重新分配时间戳
            new_chunks = self._redistribute_timestamps(old_chunks, new_chunks)

            # 替换
            sentence.chunks = (
                sentence.chunks[:start_chunk_idx] +
                new_chunks +
                sentence.chunks[end_chunk_idx + 1:]
            )

            # 更新句子文本
            sentence.text = joiner.join(c.text for c in sentence.chunks)

            # 记录修改
            self.changes.append({
                "sentence_idx": sent_idx,
                "old_text": old_text,
                "new_text": new_text,
                "old_count": len(old_chunks),
                "new_count": len(new_chunks)
            })
            changes_count += 1

        return changes_count

    def _create_new_chunks(self, text: str, source_chunks: List[Chunk]) -> List[Chunk]:
        """创建新的 Chunk 对象"""
        # 简化实现：将新文本作为一个 Chunk
        # TODO: 如果需要保留 speaker_id，可以从 source_chunks[0] 获取
        new_chunk = Chunk(
            text=text,
            start=0.0,  # 时间戳会在 _redistribute_timestamps 中设置
            end=0.0,
            speaker_id=source_chunks[0].speaker_id if source_chunks else None,
            index=source_chunks[0].index if source_chunks else 0
        )
        return [new_chunk]

    def _redistribute_timestamps(
        self, old_chunks: List[Chunk], new_chunks: List[Chunk]
    ) -> List[Chunk]:
        """重新分配时间戳"""
        old_count = len(old_chunks)
        new_count = len(new_chunks)

        # 边界永远不变
        boundary_start = old_chunks[0].start
        boundary_end = old_chunks[-1].end
        total_duration = boundary_end - boundary_start

        if old_count == new_count:
            # 数量相同，1对1复制
            for i, new_chunk in enumerate(new_chunks):
                new_chunk.start = old_chunks[i].start
                new_chunk.end = old_chunks[i].end
        else:
            # 数量不同，平均分配
            avg_duration = total_duration / new_count
            current_time = boundary_start

            for new_chunk in new_chunks:
                new_chunk.start = current_time
                new_chunk.end = current_time + avg_duration
                current_time = new_chunk.end

        return new_chunks

    def finish(self, summary: str) -> str:
        """完成矫正任务"""
        return json.dumps({
            "changes_count": len(self.changes),
            "summary": summary,
            "is_finish": True
        }, ensure_ascii=False)
