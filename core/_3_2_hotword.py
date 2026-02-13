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
from core.utils.models import Sentence, Chunk, Correction
from core.utils.sentence_tools import get_joiner
from core.utils.cache_utils import cache


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
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_frame",
            "description": "分析视频在指定时间戳的画面内容，识别屏幕上显示的文字（如图表标题、界面标签、字幕等）。当你难以判断正确的术语时，可以调用此工具查看画面辅助决策。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "number",
                        "description": "视频时间戳（秒）"
                    }
                },
                "required": ["timestamp"]
            }
        }
    }
]


# ==================== System Prompt ====================

def build_system_prompt(terms_with_meanings: List[dict], asr_language: str) -> str:
    """构建通用的 System Prompt"""
    terms_info = "\n".join([
        f"  - {t['name']}" + (f"（{t['meaning']}）" if t['meaning'] else "")
        for t in terms_with_meanings
    ])

    return f"""你是一个专业的 ASR（自动语音识别）术语矫正专家。

## 任务
音频转录中常因口音、杂音将专业术语误识别为发音相近的普通词汇。你需要通过上下文分析和视觉辅助，将这些错误还原为术语列表中的正确表达。

## 音频语言
{asr_language}

## 术语库
{terms_info}

## 核心判断方法

对每个可疑词，问自己：**快速把这个词读出来，听起来像不像术语库中的某个术语？**

如果发音相似，且上下文中该术语出现是合理的，就替换。

## ⚠️ 重要原则

1. **只矫正术语库中的术语**，不要矫正其他内容
2. **必须发音相似且上下文合理才替换**
3. **不确定时不替换**
4. **如果没有发现错误，直接调用 finish**

## 视觉辅助 (analyze_frame) 调用准则

当判断困难时，可调用 analyze_frame(timestamp) 查看画面辅助决策。

- timestamp：该术语所在句子的时间戳
- 画面内容可能包含相关信息时才使用
- 如果仅凭音频上下文就能确定，无需调用视觉辅助

## 工作流程

1. 用 read_sentences 读取所有句子（注意每句都有时间戳 [start-end]）
2. 通读全文，理解整体在讨论什么领域
3. 逐个术语思考：这个术语的发音，在文中有没有被错误识别的形式？
4. 如果判断困难，可调用 analyze_frame(timestamp) 查看画面辅助决策
5. 收集所有发现的错误对，用一次 batch_replace 完成替换
6. 调用 finish

### 批量操作示例

假设术语库中有某技术术语和某产品名称，在转录文本中发现：
- "错误形式1" 读起来像 "术语1"，且上下文在讨论相关技术
- "错误形式2" 读起来像 "术语2"，且上下文在讨论相关产品

```
batch_replace(replacements=[
  {{"old_text": "错误形式1", "new_text": "术语1"}},
  {{"old_text": "错误形式2", "new_text": "术语2"}}
])
```

### 复数形式
如果术语有复数形式出现，单数和复数各写一条规则，一次到位。
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

    def __init__(self, sentences: List[Sentence], video_path: Optional[str] = None):
        self.sentences = sentences
        self.changes: List[Dict] = []
        self.video_path = video_path
        self.vision_calls: List[Dict] = []  # 记录视觉辅助调用
        self.last_vision_call: Optional[Dict] = None  # 最后一次视觉辅助调用（用于标记后续替换）

    def read_sentences(self, start_idx: int = None, end_idx: int = None) -> str:
        """读取句子内容，返回全部或指定索引范围"""
        total = len(self.sentences)

        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = total

        # 限制最多显示 500 句
        MAX_DISPLAY = 500
        truncated = False
        actual_end = end_idx

        if actual_end - start_idx > MAX_DISPLAY:
            actual_end = start_idx + MAX_DISPLAY
            truncated = True

        result = []
        for i in range(start_idx, min(actual_end, total)):
            s = self.sentences[i]
            # 显示时间戳，方便 LLM 调用 analyze_frame
            result.append(f"第{i}句 [{s.start:.1f}s-{s.end:.1f}s]: {s.text}")

        output = "\n".join(result)
        if truncated:
            remaining = total - actual_end
            output += f"\n\n[共 {total} 句，已显示到第 {actual_end} 句，剩余 {remaining} 句。"
            output += f"请调用 read_sentences(start_idx={actual_end}) 继续查看]"

        return output


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
            rule_changes = 0
            for sent_idx, sentence in enumerate(self.sentences):
                changes_count = self._replace_in_sentence(
                    sentence, sent_idx, old_text, new_text
                )
                rule_changes += changes_count
            total_changes += rule_changes

            results.append({
                "old_text": old_text,
                "new_text": new_text,
                "count": rule_changes
            })

        # 清除视觉辅助标记（只标记紧跟在视觉辅助后的替换）
        self.last_vision_call = None

        return json.dumps({
            "success": True,
            "total_changes": total_changes,
            "details": results
        }, ensure_ascii=False)

    def _replace_in_sentence(
        self, sentence: Sentence, sent_idx: int,
        old_text: str, new_text: str
    ) -> int:
        """在单个句子中查找并替换（记录 Correction 并修改文本）"""
        # 先收集所有匹配，避免在循环中修改文本导致位置漂移
        matches = [
            m for m in re.finditer(re.escape(old_text), sentence.text)
            if self._is_word_boundary(sentence.text, m.start(), m.end() - m.start())
        ]

        # 从后往前替换，不影响前面的位置
        for match in reversed(matches):
            start, end = match.span()

            # 记录矫正（在修改文本之前记录位置）
            sentence.corrections.append(Correction(
                old_text=old_text,
                new_text=new_text,
                start_idx=start,
                end_idx=end
            ))

            # 执行替换
            sentence.text = (
                sentence.text[:start] +
                new_text +
                sentence.text[end:]
            )

            # 保留旧的 changes 记录（用于统计输出）
            change_record = {
                "sentence_idx": sent_idx,
                "old_text": old_text,
                "new_text": new_text
            }
            # 如果紧跟在视觉辅助调用之后，标记为使用了视觉辅助
            if self.last_vision_call:
                change_record["vision_assisted"] = True
                change_record["vision_timestamp"] = self.last_vision_call["timestamp"]
                change_record["vision_result"] = self.last_vision_call["result"]
            self.changes.append(change_record)

        return len(matches)

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

    def analyze_frame(self, timestamp: float) -> str:
        """
        分析视频指定时间戳的画面内容

        Args:
            timestamp: 视频时间戳（秒）

        Returns:
            画面中识别到的内容，或错误信息（JSON格式）
        """
        from core.utils.ask_gpt import ask_gpt_vision
        import subprocess
        from pathlib import Path

        # 验证 video_path
        if not self.video_path:
            return json.dumps({
                "error": "视频文件路径未设置",
                "suggestion": "请确保视频文件已正确加载"
            }, ensure_ascii=False)

        # 验证 timestamp
        if timestamp < 0:
            return json.dumps({
                "error": f"无效的时间戳: {timestamp}",
                "suggestion": "时间戳必须 >= 0"
            }, ensure_ascii=False)

        video_path = self.video_path

        # 创建输出目录
        output_dir = Path("output/log/pic")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 文件名: 帧时间戳.png
        output_path = output_dir / f"frame_{timestamp:.1f}s.png"

        # 用 ffmpeg 提取单帧
        try:
            subprocess.run([
                "ffmpeg", "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                "-y", str(output_path)
            ], check=True, capture_output=True, timeout=60)
        except FileNotFoundError:
            return json.dumps({
                "error": "ffmpeg 未安装或不在 PATH 中",
                "suggestion": "请安装 ffmpeg 并确保其在系统 PATH 中"
            }, ensure_ascii=False)
        except subprocess.CalledProcessError as e:
            return json.dumps({
                "error": f"ffmpeg 执行失败",
                "suggestion": "请检查视频文件是否存在且格式有效",
                "stderr": e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            }, ensure_ascii=False)
        except subprocess.TimeoutExpired:
            return json.dumps({
                "error": "ffmpeg 执行超时",
                "suggestion": "视频文件可能损坏或时间戳超出视频长度"
            }, ensure_ascii=False)

        # 调用 Vision API
        try:
            result = ask_gpt_vision(str(output_path), "请分析这张图片")
            # 记录成功的视觉辅助调用
            vision_call = {
                "timestamp": timestamp,
                "result": result[:100] + "..." if len(result) > 100 else result
            }
            self.vision_calls.append(vision_call)
            # 设置标记，用于关联后续的术语替换
            self.last_vision_call = vision_call
            return result
        except Exception as e:
            return json.dumps({
                "error": f"Vision API 调用失败: {str(e)}",
                "suggestion": "请检查 API 密钥配置或网络连接"
            }, ensure_ascii=False)


# ==================== 主入口函数 ====================

def _rebuild_chunks_from_corrections(sentences: List[Sentence]) -> None:
    """基于 sentence.corrections 重建 chunks（锁定时间边界）"""
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)

    for sentence in sentences:
        if not sentence.corrections:
            continue

        orig_chunks = sentence.chunks
        if not orig_chunks:
            continue

        # 构建 chunk 位置映射：字符位置 -> Chunk
        chunk_map = []
        curr_idx = 0
        for chunk in orig_chunks:
            start_pos = sentence.original_text.find(chunk.text, curr_idx)
            if start_pos == -1:
                start_pos = curr_idx
            end_pos = start_pos + len(chunk.text)
            chunk_map.append({
                "chunk": chunk,
                "start_idx": start_pos,
                "end_idx": end_pos
            })
            curr_idx = end_pos

        # 按位置排序 corrections（从后往前处理，避免位置偏移）
        sorted_corrections = sorted(
            sentence.corrections,
            key=lambda c: c.start_idx,
            reverse=True
        )

        final_chunks = []
        skip_until = 0

        for item in chunk_map:
            chunk = item["chunk"]
            chunk_start = item["start_idx"]
            chunk_end = item["end_idx"]

            if chunk_start < skip_until:
                continue

            # 检查是否有 correction 涉及这个 chunk
            correction = None
            for corr in sorted_corrections:
                if not (chunk_end <= corr.start_idx or chunk_start >= corr.end_idx):
                    correction = corr
                    break

            if correction:
                # 找到所有被这个 correction 影响的 chunks
                affected_items = [
                    item for item in chunk_map
                    if not (item["end_idx"] <= correction.start_idx or
                           item["start_idx"] >= correction.end_idx)
                ]

                if affected_items:
                    # 锁定时间边界
                    fixed_start = affected_items[0]["chunk"].start
                    fixed_end = affected_items[-1]["chunk"].end
                    spk_id = affected_items[0]["chunk"].speaker_id
                else:
                    fixed_start = final_chunks[-1].end if final_chunks else sentence.start
                    fixed_end = fixed_start
                    spk_id = orig_chunks[0].speaker_id

                # 用 new_text 切分并分配时间
                new_chunks = _split_text_into_chunks(
                    correction.new_text, fixed_start, fixed_end,
                    asr_language, joiner, spk_id
                )
                final_chunks.extend(new_chunks)

                skip_until = correction.end_idx
            else:
                final_chunks.append(chunk)

        sentence.chunks = final_chunks
        if final_chunks:
            sentence.start = final_chunks[0].start
            sentence.end = final_chunks[-1].end


def _split_text_into_chunks(
    text: str, start_time: float, end_time: float,
    asr_language: str, joiner: str,
    speaker_id: Optional[str]
) -> List[Chunk]:
    """将文本切分为 chunks 并在时间范围内平均分配"""
    # 根据语言选择切分方式
    if asr_language in ['zh', 'ja', 'ko']:
        parts = list(text)
    else:
        parts = text.split()
        if joiner:
            text_with_joiner = joiner.join(parts)
            parts = text_with_joiner.split(joiner) if joiner else parts

    if not parts:
        return []

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


@timer("ASR 术语矫正")
@cache()
def correct_terms_in_sentences(sentences: List[Sentence]) -> List[Sentence]:
    """
    主函数：对句子列表进行术语矫正

    Args:
        sentences: NLP 分句后的 Sentence 对象列表

    Returns:
        List[Sentence]: 矫正后的 Sentence 对象列表
    """
    # 0. 初始化热词分组（如果需要）
    from core.utils.config_utils import init_hotword_groups
    init_hotword_groups()

    # 1. 检查开关
    enabled = load_key("asr_term_correction.enabled")
    if not enabled:
        rprint("[yellow]⏭️ 术语矫正已禁用，跳过[/yellow]")
        return sentences

    # 2. 加载激活分组的热词配置
    active_group_id = load_key("asr_term_correction.active_group_id")
    groups = load_key("asr_term_correction.groups") or []

    # 找到激活分组
    active_group = next((g for g in groups if g["id"] == active_group_id), None)

    if not active_group:
        rprint("[yellow]⚠️ 未找到激活的热词分组，跳过矫正[/yellow]")
        return sentences

    terms_config = active_group.get("keyterms", [])
    if not terms_config:
        rprint(f"[yellow]⏭️ 分组 '{active_group['name']}' 中未配置热词，跳过矫正[/yellow]")
        return sentences

    # 3. 解析术语列表
    terms_with_meanings = _parse_terms(terms_config)
    rprint(f"[blue]🔍 开始术语矫正，共 {len(terms_with_meanings)} 个术语[/blue]")
    for t in terms_with_meanings:
        rprint(f"  - {t['name']}" + (f": {t['meaning']}" if t['meaning'] else ""))

    # 4. 保存原始文本（用于 chunk 重建）
    for sent in sentences:
        sent.original_text = sent.text

    # 5. 获取视频文件路径
    from core._1_ytdlp import find_video_files
    video_path = find_video_files()

    # 6. 创建工具执行器
    tool_executor = SentenceToolExecutor(sentences, video_path=video_path)

    # 7. 构建系统提示词（传入 asr_language）
    asr_language = load_key("asr.language")
    system_prompt = build_system_prompt(terms_with_meanings, asr_language)

    # 8. 调用 LLM Agent
    term_names = ", ".join(t['name'] for t in terms_with_meanings)
    user_task = f"请检查这 {len(sentences)} 个句子，找出以下术语的语音识别错误并矫正：{term_names}"

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

            # 保存矫正记录到文件（与控制台输出一致）
            _save_correction_log(tool_executor.changes, changes_count)
        else:
            rprint("[green]✅ 未发现需要矫正的错误[/green]")
    else:
        rprint("[yellow]⚠️ LLM 未正常完成，返回原始句子[/yellow]")

    # 10. 同步 chunks（基于 corrections 重建）
    _rebuild_chunks_from_corrections(sentences)

    return sentences


def _save_correction_log(changes, changes_count):
    """保存矫正日志到 output/log/hotword_correct.txt"""
    from collections import Counter

    log_path = "output/log/hotword_correct.txt"

    with open(log_path, 'w', encoding='utf-8') as f:
        if changes_count > 0:
            f.write(f"✅ 矫正完成: {changes_count} 处修改\n")

            # 构建统计信息
            stats = Counter(f"{c['old_text']} → {c['new_text']}" for c in changes)

            # 分离有/无视觉辅助的矫正
            vision_assisted_changes = [c for c in changes if c.get('vision_assisted')]
            normal_changes = [c for c in changes if not c.get('vision_assisted')]

            # 显示所有矫正（带视觉辅助标记）
            f.write("\n")
            for change, count in stats.most_common():
                # 检查这个矫正是否使用了视觉辅助
                vision_item = next((c for c in vision_assisted_changes
                                   if c['old_text'] == change.split(' → ')[0]
                                   and c['new_text'] == change.split(' → ')[1]), None)
                if vision_item:
                    f.write(f"  {change}: {count} 处 📷\n")
                else:
                    f.write(f"  {change}: {count} 处\n")

            # 视觉辅助详情
            if vision_assisted_changes:
                f.write(f"\n📷 视觉辅助详情:\n")
                vision_unique = {}
                for c in vision_assisted_changes:
                    key = f"{c['old_text']} → {c['new_text']}"
                    if key not in vision_unique:
                        vision_unique[key] = c['vision_result']

                for change, result in vision_unique.items():
                    f.write(f"  [{change}]\n")
                    f.write(f"    → 识别: {result}\n")
                f.write(f"\n  帧图片已保存至: output/log/pic/\n")

        else:
            f.write("✅ 未发现需要矫正的错误\n")

    rprint(f"[dim]📝 矫正日志已保存: {log_path}[/dim]")
