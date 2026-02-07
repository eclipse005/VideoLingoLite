"""
ASR 术语矫正模块 (Stage 1.5)

在 NLP 分句后进行，使用 LLM Agent 智能识别并矫正 ASR 错误

输入: List[Sentence] (来自 _3_1_split_nlp.py)
输出: List[Sentence] (矫正后，传递给 _3_2_split_meaning.py)
"""

import json
import re
from typing import List, Tuple, Dict, Any, Optional

from core.utils import rprint, load_key, timer
from core.utils.ask_gpt import ask_gpt_with_tools
from core.utils.models import Sentence
from core.utils.sentence_tools import clean_word


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


# ==================== 主入口函数 ====================

def _sync_chunks_to_text(sentences: List[Sentence]) -> None:
    """
    同步 chunks 到矫正后的文本

    将矫正后的 sentence.text 作为一个整体 chunk，
    时间戳使用原句子的 start/end。

    Args:
        sentences: Sentence 对象列表（会就地修改）
    """
    for sentence in sentences:
        # 如果 sentence.text 被修改了，重建 chunks
        # 简化策略：将整个 text 作为一个 chunk
        new_chunk = Chunk(
            text=sentence.text,
            start=sentence.start,
            end=sentence.end,
            speaker_id=sentence.chunks[0].speaker_id if sentence.chunks else None,
            index=0
        )
        sentence.chunks = [new_chunk]

        # 确保 Sentence.start/end 与 chunks 一致
        if sentence.chunks:
            sentence.start = sentence.chunks[0].start
            sentence.end = sentence.chunks[-1].end


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
    enabled = load_key("asr_term_correction.enabled", default=False)
    if not enabled:
        rprint("[yellow]⏭️ 术语矫正已禁用，跳过[/yellow]")
        return sentences

    # 2. 加载术语配置
    terms_config = load_key("asr_term_correction.terms", default=[])
    if not terms_config:
        rprint("[yellow]⏭️ 未配置术语，跳过矫正[/yellow]")
        return sentences

    # 3. 解析术语列表
    terms_with_meanings = _parse_terms(terms_config)
    rprint(f"[blue]🔍 开始术语矫正，共 {len(terms_with_meanings)} 个术语[/blue]")
    for t in terms_with_meanings:
        rprint(f"  - {t['name']}" + (f": {t['meaning']}" if t['meaning'] else ""))

    # 4. 创建工具执行器
    tool_executor = SentenceToolExecutor(sentences)

    # 5. 构建系统提示词
    system_prompt = build_system_prompt(terms_with_meanings)

    # 6. 调用 LLM Agent
    user_task = f"请矫正这 {len(sentences)} 个句子中的术语错误"

    result = ask_gpt_with_tools(
        system_prompt=system_prompt,
        prompt=user_task,
        tools=SENTENCES_TOOLS_DEFINITION,
        tool_executor=tool_executor,
        max_rounds=20,
        log_title="asr_term_correction"
    )

    # 7. 输出统计
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

        rprint(f"[dim]LLM 总结: {result.get('summary', 'N/A')}[/dim]")
    else:
        rprint("[yellow]⚠️ LLM 未正常完成，返回原始句子[/yellow]")

    # 8. 同步 chunks 到矫正后的文本
    _sync_chunks_to_text(sentences)

    return sentences
