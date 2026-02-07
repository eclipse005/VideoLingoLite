"""
ASR 热词矫正模块 (Stage 2)

在 NLP 分句后进行，使用 LLM Agent 智能识别并矫正 ASR 错误

输入: List[Sentence] (来自 _3_1_split_nlp.py)
输出: List[Sentence] (矫正后，传递给 _3_3_split_meaning.py)
"""

import json
import os
import re
import unicodedata
from typing import List, Tuple, Dict, Any, Optional

from core.utils import rprint, load_key, timer
from core.utils.ask_gpt import ask_gpt_with_tools
from core.utils.models import Sentence, Chunk
from core.utils.sentence_tools import get_joiner


# ==================== 工具定义 ====================

SENTENCES_TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "read_sentences",
            "description": "读取句子内容，返回全部或指定索引范围",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_idx": {"type": "integer", "description": "起始索引（可选，从0开始）"},
                    "end_idx": {"type": "integer", "description": "结束索引（可选）"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "获取某句子的上下文（前后各N句）",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentence_idx": {"type": "integer"},
                    "context_count": {"type": "integer", "default": 2}
                },
                "required": ["sentence_idx"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "batch_replace",
            "description": "批量执行多处替换。在所有句子中查找并替换多种错误形式",
            "parameters": {
                "type": "object",
                "properties": {
                    "replacements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string", "description": "要查找的错误文本"},
                                "new_text": {"type": "string", "description": "替换后的正确术语"}
                            },
                            "required": ["old_text", "new_text"]
                        }
                    }
                },
                "required": ["replacements"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "完成矫正任务",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "修改总结"}
                },
                "required": ["summary"]
            }
        }
    }
]


# ==================== System Prompt ====================

def build_system_prompt(terms_with_meanings: List[dict]) -> str:
    """构建 System Prompt（完全复用 agent_correct.py）"""
    terms_info = "\n".join([
        f"  - {t['name']}" + (f": {t['meaning']}" if t['meaning'] else "")
        for t in terms_with_meanings
    ])

    return f"""你是一个专业的 ASR（自动语音识别）术语矫正专家。

## 任务
矫正文本中被语音识别错误的专业术语。

## 术语列表及含义
{terms_info}

## ⚠️ 重要限制（必须遵守）
**只能矫正上述术语列表中的术语**，不要矫正其他内容。
- 如果文本中没有上述术语的错误形式，直接调用 finish
- 不要自创术语或矫正不在列表中的内容
- 只处理指定的 {len(terms_with_meanings)} 个术语
- **复数形式也允许**：根据上下文判断（如 APIs, URLs）

## 可用工具
1. read_sentences - 读取句子内容
2. get_context - 查看某句的上下文（前后几句）
3. batch_replace - 批量替换（在所有句子中执行多个替换规则）
4. finish - 完成矫正

## 高效工作流程（重要！）

1. **先全局扫描**：用 read_sentences 查看句子内容
2. **记录所有疑似错误**：在心中/草稿中记录所有发现的错误位置和形式
3. **批量处理**：阅读完成后，使用 batch_replace 一次性完成所有替换
4. **最后调用 finish**

### 批量操作示例

**推荐方式：使用 batch_replace**
如果发现多种错误模式，用一次 batch_replace 完成所有替换：
```
batch_replace(replacements=[
  {{"old_text": "L L M", "new_text": "LLM"}},
  {{"old_text": "j son", "new_text": "JSON"}},
  {{"old_text": "A P I", "new_text": "API"}}
])
```

### 使用 batch_replace 的关键原则（重要！）

**一次提交所有规则，包括复数形式**：
- 如果术语在文本中有复数出现，必须为单数和复数各写一条规则
- 不要先替换再回头修复复数，一次到位

示例：假设术语是 "ABC"，文本中发现错误形式 "A B C" 和 "A B Cs"
```
batch_replace(replacements=[
  {{"old_text": "A B C", "new_text": "ABC"}},
  {{"old_text": "A B Cs", "new_text": "ABCs"}}
])
```

错误做法：
- 只提交 {{"old_text": "A B Cs", "new_text": "ABC"}}（丢失复数）
- 先提交单数规则，后续再补充复数规则（浪费轮次）

## 重要原则

### 复数形式
- 根据上下文判断是否需要复数形式
- 如果语境是复数（these/those/all/multiple/several 等标记），替换时使用复数形式

示例：
- "these L L M models" → 替换为 "these LLM models"（复数）
- "a L L M model" → 替换为 "a LLM model"（单数）

### 谨慎修正
- 只修改明确是 ASR 误识别的情况
- 如果某个词在当前上下文中是合理的，不要修改

示例：
- "I can see the point" → "see" 是正确的，不要改成 "C" 或 "sea"
- "We use API calls" → "API" 是正确的，不需要修改

## 常见 ASR 错误模式参考
- 空格插入：字母间被插入空格（L L M → LLM）
- 大小写错误：首字母未大写或全小写（json → JSON）
- 同音近音：发音相似的错误替换

请开始工作，记住：批量处理，不要逐个处理。
"""


def _parse_terms(terms_config: List[str]) -> List[dict]:
    """解析术语列表，支持 '术语 : 含义' 格式"""
    parsed = []
    for term in terms_config:
        if ' : ' in term:
            parts = term.split(' : ', 1)
        elif ': ' in term:
            parts = term.split(': ', 1)
        elif '：' in term:
            parts = term.split('：', 1)
        else:
            parsed.append({'name': term.strip(), 'meaning': None})
            continue

        parsed.append({
            'name': parts[0].strip(),
            'meaning': parts[1].strip() if len(parts) > 1 else None
        })
    return parsed


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
        """
        在单个句子中查找并替换（只修改文本）

        使用多语言单词边界检查，支持：中、日、韩、英、德、俄等所有语言
        """
        changes_count = 0

        # 在原始文本中查找所有匹配位置
        for match in re.finditer(re.escape(old_text), sentence.text):
            start, end = match.span()

            # 检查是否为单词边界
            if self._is_word_boundary(sentence.text, start, end - start):
                # 执行替换
                sentence.text = (
                    sentence.text[:start] +
                    new_text +
                    sentence.text[end:]
                )
                changes_count += 1

                # 记录修改
                self.changes.append({
                    "sentence_idx": sent_idx,
                    "old_text": old_text,
                    "new_text": new_text
                })

        return changes_count

    def _is_word_boundary(self, text: str, pos: int, length: int) -> bool:
        """
        检查指定位置是否为单词边界（多语言通用）

        策略：检查匹配位置前后是否为字母（L* 类别）
             如果前后都不是字母，则为单词边界

        支持：中、日、韩、英、德、俄、法、意、西等所有语言

        Args:
            text: 完整文本
            pos: 匹配起始位置
            length: 匹配长度

        Returns:
            bool: True 表示是单词边界，False 表示不是
        """
        # 检查前一个字符
        if pos > 0:
            prev_char = text[pos - 1]
            if unicodedata.category(prev_char).startswith('L'):  # 前面是字母
                return False

        # 检查后一个字符
        end_pos = pos + length
        if end_pos < len(text):
            next_char = text[end_pos]
            if unicodedata.category(next_char).startswith('L'):  # 后面是字母
                return False

        return True  # 前后都不是字母，是单词边界

    def finish(self, summary: str) -> str:
        """完成矫正任务"""
        return json.dumps({
            "changes_count": len(self.changes),
            "summary": summary,
            "is_finish": True
        }, ensure_ascii=False)


# ==================== 主入口函数 ====================

def _sync_chunks_to_text(sentences: List[Sentence], original_texts: List[str]) -> None:
    """
    同步 chunks 到矫正后的文本（使用 difflib 对比）

    策略：
    - 使用 difflib 对比矫正前后的文本
    - 未修改部分 → 保持原 chunks
    - 修改部分 → 重建 chunks（切分 + 重新分配时间戳）

    Args:
        sentences: Sentence 对象列表（会就地修改）
        original_texts: 矫正前的原始文本列表
    """
    from difflib import SequenceMatcher

    for sent_idx, sentence in enumerate(sentences):
        original_text = original_texts[sent_idx]

        # 文本未修改，保持原 chunks
        if original_text == sentence.text:
            continue

        # 使用 difflib 对比
        matcher = SequenceMatcher(None, original_text, sentence.text)

        new_chunks = []
        current_orig_chunk_idx = 0
        orig_chunks = sentence.chunks
        asr_language = load_key("asr.language")
        joiner = get_joiner(asr_language)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # 未修改部分：保持原 chunks
                # 通过字符位置找到对应的 chunks
                char_pos = 0
                for chunk in orig_chunks:
                    chunk_end = char_pos + len(chunk.text)
                    # 检查 chunk 是否完全在 [i1, i2) 范围内
                    if char_pos >= i1 and chunk_end <= i2:
                        new_chunks.append(chunk)
                    elif chunk_end > i2:
                        # 超出当前 equal 范围，停止处理
                        break
                    char_pos = chunk_end

            elif tag == 'replace':
                # 修改部分：重建 chunks
                # 获取修改部分的原始时间范围
                orig_start_time = None
                orig_end_time = None

                # 找到原始文本中对应位置的时间戳
                char_pos = 0
                for chunk in orig_chunks:
                    chunk_text = chunk.text
                    chunk_end = char_pos + len(chunk_text)

                    if char_pos <= i1 and chunk_end >= i2:
                        # chunk 完全在修改范围内
                        if orig_start_time is None:
                            orig_start_time = chunk.start
                        orig_end_time = chunk.end
                    elif char_pos <= i1 < chunk_end:
                        # chunk 跨越修改开始边界
                        if orig_start_time is None:
                            orig_start_time = chunk.start
                        orig_end_time = chunk.end
                    elif i1 < char_pos <= i2:
                        # chunk 在修改范围内
                        orig_end_time = chunk.end

                    char_pos += len(chunk_text)

                # 如果找不到时间戳，使用句子的边界
                if orig_start_time is None:
                    orig_start_time = sentence.start
                if orig_end_time is None:
                    orig_end_time = sentence.end

                # 切分新文本
                new_text = sentence.text[j1:j2]
                rebuilt_chunks = _split_text_into_chunks(
                    new_text, orig_start_time, orig_end_time,
                    asr_language, joiner,
                    speaker_id=orig_chunks[0].speaker_id if orig_chunks else None
                )
                new_chunks.extend(rebuilt_chunks)

            elif tag == 'delete':
                # 删除部分：跳过
                pass

            elif tag == 'insert':
                # 插入部分：使用原位置的时间戳
                # 在 j1-j2 位置插入，使用附近的时间戳
                new_chunks.extend(_split_text_into_chunks(
                    sentence.text[j1:j2],
                    sentence.start, sentence.end,
                    asr_language, joiner,
                    speaker_id=orig_chunks[0].speaker_id if orig_chunks else None
                ))

        # 更新 sentence 的 chunks
        sentence.chunks = new_chunks

        # 确保 Sentence.start/end 与 chunks 一致
        if new_chunks:
            sentence.start = new_chunks[0].start
            sentence.end = new_chunks[-1].end


def _split_text_into_chunks(
    text: str, start_time: float, end_time: float,
    asr_language: str, joiner: str,
    speaker_id: Optional[str]
) -> List[Chunk]:
    """
    将文本切分为 chunks 并分配时间戳

    策略：
    - 空格分隔语言：按空格切分
    - CJK 语言：按字符切分
    - 时间戳：平均分配，边界不变
    """
    # 根据语言选择切分方式
    if asr_language in ['zh', 'ja', 'ko']:
        # CJK 语言：按字符切分
        parts = list(text)
    else:
        # 空格分隔语言：按空格切分
        parts = text.split()
        if joiner:
            # 如果有空格连接符，重新添加以便正确分配时间
            text_with_joiner = joiner.join(parts)
            parts = text_with_joiner.split(joiner) if joiner else parts

    if not parts:
        return []

    # 分配时间戳
    total_duration = end_time - start_time
    chunk_duration = total_duration / len(parts)

    chunks = []
    current_time = start_time

    for i, part in enumerate(parts):
        chunk_start = current_time
        chunk_end = current_time + chunk_duration

        # 最后一个 chunk 使用 end_time（避免浮点误差）
        if i == len(parts) - 1:
            chunk_end = end_time

        chunks.append(Chunk(
            text=part,
            start=chunk_start,
            end=chunk_end,
            speaker_id=speaker_id,
            index=i
        ))

        current_time = chunk_end

    return chunks


def _save_checkpoint(sentences: List[Sentence], path: str) -> None:
    """保存矫正后的句子到断点文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence.text + '\n')
    rprint(f"[dim]💾 断点已保存: {path}[/dim]")


def _load_checkpoint(sentences: List[Sentence], path: str) -> List[Sentence]:
    """从断点文件加载矫正后的句子"""
    with open(path, 'r', encoding='utf-8') as f:
        corrected_texts = [line.rstrip('\n') for line in f]

    # 验证数量一致
    if len(corrected_texts) != len(sentences):
        rprint(f"[yellow]⚠️ 断点文件句子数量不匹配 ({len(corrected_texts)} vs {len(sentences)})，重新执行矫正[/yellow]")
        os.remove(path)
        return sentences

    # 保存原始文本（用于 chunk 同步）
    original_texts = [s.text for s in sentences]

    # 更新句子的文本
    for sent, corrected_text in zip(sentences, corrected_texts):
        sent.text = corrected_text

    # 同步 chunks（矫正前 -> 矫正后）
    _sync_chunks_to_text(sentences, original_texts)

    return sentences


@timer("ASR 术语矫正")
def correct_terms_in_sentences(sentences: List[Sentence]) -> List[Sentence]:
    """
    主函数：对句子列表进行术语矫正

    Args:
        sentences: NLP 分句后的 Sentence 对象列表

    Returns:
        List[Sentence]: 矫正后的 Sentence 对象列表
    """
    # 1. 检查开关
    enabled = load_key("asr_term_correction.enabled")
    if not enabled:
        rprint("[yellow]⏭️ 术语矫正已禁用，跳过[/yellow]")
        return sentences

    # 2. 加载术语配置
    terms_config = load_key("asr_term_correction.terms")
    if not terms_config:
        rprint("[yellow]⏭️ 未配置术语，跳过矫正[/yellow]")
        return sentences

    # 3. 检查断点文件
    checkpoint_path = "output/log/hotword_correct.txt"
    if os.path.exists(checkpoint_path):
        rprint(f"[blue]📂 发现断点文件，直接加载矫正结果[/blue]")
        return _load_checkpoint(sentences, checkpoint_path)

    # 4. 解析术语列表
    terms_with_meanings = _parse_terms(terms_config)
    rprint(f"[blue]🔍 开始术语矫正，共 {len(terms_with_meanings)} 个术语[/blue]")
    for t in terms_with_meanings:
        rprint(f"  - {t['name']}" + (f": {t['meaning']}" if t['meaning'] else ""))

    # 5. 创建工具执行器
    tool_executor = SentenceToolExecutor(sentences)

    # 6. 保存原始文本（用于后续 chunk 同步）
    original_texts = [s.text for s in sentences]

    # 7. 构建系统提示词
    system_prompt = build_system_prompt(terms_with_meanings)

    # 8. 调用 LLM Agent
    user_task = f"请矫正这 {len(sentences)} 个句子中的术语错误"

    result = ask_gpt_with_tools(
        system_prompt=system_prompt,
        prompt=user_task,
        tools=SENTENCES_TOOLS_DEFINITION,
        tool_executor=tool_executor,
        max_rounds=20,
        log_title="hotword_correction"
    )

    # 9. 输出统计
    if result and result.get("is_finish"):
        changes_count = result.get("changes_count", 0)
        if changes_count > 0:
            rprint(f"[green]✅ 矫正完成: {changes_count} 处修改[/green]")

            # 显示修改明细
            from collections import Counter
            stats = Counter(f"{c['old_text']} → {c['new_text']}" for c in tool_executor.changes)
            for change, count in stats.most_common():
                rprint(f"  {change}: {count} 处")
        else:
            rprint("[green]✅ 未发现需要矫正的错误[/green]")
    else:
        rprint("[yellow]⚠️ LLM 未正常完成，返回原始句子[/yellow]")

    # 10. 同步 chunks（使用矫正前的文本进行对比）
    _sync_chunks_to_text(sentences, original_texts)

    # 11. 保存断点文件
    _save_checkpoint(sentences, checkpoint_path)

    return sentences
