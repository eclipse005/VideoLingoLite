# Chunk/Sentence 对象化重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 将数据处理从"文本处理 + difflib 时间对齐"重构为"对象化处理"，时间戳始终跟随 Chunk 对象

**架构:** Chunk 对象（词/字级别）携带时间戳，Sentence 对象（句子级别）组合 Chunk。时间戳通过对象引用传递，消除大部分 difflib 匹配。

**技术栈:** Python 3.10, pandas, dataclasses, difflib (仅 LLM 断句时使用), spaCy (NLP 分句)

---

## Task 1: 创建核心数据结构 (Chunk 和 Sentence 类)

**Files:**
- Modify: `core/models.py`

**Step 1: 在 core/models.py 顶部添加必要的导入**

```python
from dataclasses import dataclass, field
from typing import List, Optional
```

**Step 2: 在 core/models.py 中添加 Chunk 类**

在 `_2_CLEANED_CHUNKS` 常量定义之前添加：

```python
@dataclass
class Chunk:
    """词/字级别的语音识别单元，携带时间戳"""
    text: str           # 词/字内容（如 "Hello", "world."）
    start: float        # 开始时间（秒）
    end: float          # 结束时间（秒）
    speaker_id: Optional[str] = None  # 说话人ID
    index: int = 0      # 在 cleaned_chunks.csv 中的行号

    @property
    def duration(self) -> float:
        """获取该词的时长"""
        return self.end - self.start
```

**Step 3: 在 core/models.py 中添加 Sentence 类**

在 Chunk 类之后添加：

```python
@dataclass
class Sentence:
    """句子级别的对象，由多个 Chunk 组成"""
    chunks: List[Chunk]     # 组成这句话的所有 Chunk
    text: str               # 从 chunks 拼接的完整文本
    start: float            # = chunks[0].start
    end: float              # = chunks[-1].end
    translation: str = ""   # 翻译文本
    index: int = 0          # 句子序号
    is_split: bool = False  # 是否被 LLM 切分过

    @property
    def duration(self) -> float:
        """获取这句话的时长"""
        return self.end - self.start

    def update_timestamps(self):
        """根据 chunks 更新 start 和 end 时间戳"""
        if self.chunks:
            self.start = self.chunks[0].start
            self.end = self.chunks[-1].end
```

**Step 4: 验证语法**

运行: `python -c "from core.models import Chunk, Sentence; print('Import successful')"`
Expected: `Import successful`

**Step 5: 提交**

```bash
git add core/models.py
git commit -m "feat: add Chunk and Sentence dataclasses"
```

---

## Task 2: 创建 Chunk 加载函数

**Files:**
- Modify: `core/_2_asr.py`

**Step 1: 在 core/_2_asr.py 中添加 load_chunks 函数**

在文件末尾、`if __name__ == "__main__":` 之前添加：

```python
def load_chunks() -> List[Chunk]:
    """
    从 cleaned_chunks.csv 加载 Chunk 对象列表

    Returns:
        List[Chunk]: 词/字级别的 Chunk 对象列表
    """
    df = safe_read_csv(_2_CLEANED_CHUNKS)
    chunks = []

    for idx, row in df.iterrows():
        chunk = Chunk(
            text=str(row['text']).strip('"'),
            start=float(row['start']),
            end=float(row['end']),
            speaker_id=str(row['speaker_id']) if pd.notna(row['speaker_id']) and row['speaker_id'] else None,
            index=idx
        )
        chunks.append(chunk)

    rprint(f"[green]✅ Loaded {len(chunks)} chunks from {_2_CLEANED_CHUNKS}[/green]")
    return chunks
```

**Step 2: 验证 load_chunks 函数**

创建测试文件 `test_load_chunks.py`:

```python
# 临时测试文件
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core._2_asr import load_chunks

# 确保有 cleaned_chunks.csv
if os.path.exists('output/log/cleaned_chunks.csv'):
    chunks = load_chunks()
    print(f"First chunk: {chunks[0].text} ({chunks[0].start} - {chunks[0].end})")
    print(f"Total chunks: {len(chunks)}")
else:
    print("cleaned_chunks.csv not found, skipping test")
```

运行: `python test_load_chunks.py`
Expected: 显示第一个 chunk 的信息和总数

**Step 3: 清理测试文件**

```bash
rm test_load_chunks.py
```

**Step 4: 提交**

```bash
git add core/_2_asr.py
git commit -m "feat: add load_chunks() function to load Chunk objects"
```

---

## Task 3: 创建 NLP 分句函数（字符位置追踪）

**Files:**
- Modify: `core/_3_1_split_nlp.py`

**Step 1: 添加必要的导入**

在文件顶部添加：

```python
from dataclasses import dataclass, field
from typing import List, Optional
from core.models import Chunk, Sentence
```

**Step 2: 添加字符位置映射函数**

在 `split_by_nlp()` 函数之前添加：

```python
def build_char_to_chunk_mapping(chunks: List[Chunk]) -> List[int]:
    """
    构建字符到 Chunk 索引的映射

    Args:
        chunks: Chunk 对象列表

    Returns:
        每个字符对应的 Chunk 索引列表
    """
    char_to_chunk = []
    for chunk_idx, chunk in enumerate(chunks):
        char_to_chunk.extend([chunk_idx] * len(chunk.text))
    return char_to_chunk


def nlp_split_to_sentences(chunks: List[Chunk], nlp) -> List[Sentence]:
    """
    使用 spaCy 进行 NLP 分句，将 Chunk 对象组合成 Sentence 对象

    Args:
        chunks: Chunk 对象列表
        nlp: spaCy NLP 模型

    Returns:
        Sentence 对象列表
    """
    # 1. 拼接所有 Chunk 的文本
    full_text = "".join(chunk.text for chunk in chunks)

    # 2. 构建字符到 Chunk 的映射
    char_to_chunk = build_char_to_chunk_mapping(chunks)

    # 3. 使用 spaCy 分句
    doc = nlp(full_text)
    sentences = []

    for sent_idx, sent in enumerate(doc.sents):
        start_char = sent.start_char
        end_char = sent.end_char

        # 边界检查
        start_char = max(0, min(start_char, len(full_text) - 1))
        end_char = max(start_char + 1, min(end_char, len(full_text)))

        # 找到对应的 Chunk 范围
        start_chunk_idx = char_to_chunk[start_char]
        end_chunk_idx = char_to_chunk[end_char - 1]

        # 提取对应的 Chunk 对象
        sentence_chunks = chunks[start_chunk_idx:end_chunk_idx + 1]

        # 创建 Sentence 对象
        sentence = Sentence(
            chunks=sentence_chunks,
            text=sent.text,
            start=sentence_chunks[0].start if sentence_chunks else 0.0,
            end=sentence_chunks[-1].end if sentence_chunks else 0.0,
            index=sent_idx
        )
        sentences.append(sentence)

    return sentences
```

**Step 3: 修改 split_by_nlp() 函数以使用新的对象**

修改现有的 `split_by_nlp(nlp)` 函数：

```python
@check_file_exists(_3_1_SPLIT_BY_NLP)
def split_by_nlp(nlp):
    """
    NLP 分句主函数

    输入: cleaned_chunks.csv → List[Chunk]
    输出: List[Sentence] → 保存到 split_by_nlp.txt (文本) 和返回对象
    """
    console.print("[blue]🔍 Starting NLP sentence splitting...[/blue]")

    # 1. 加载 Chunk 对象
    chunks = load_chunks()

    # 2. NLP 分句，生成 Sentence 对象
    sentences = nlp_split_to_sentences(chunks, nlp)

    # 3. 保存文本到文件（向后兼容）
    with open(_3_1_SPLIT_BY_NLP, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent.text + '\n')

    console.print(f'[green]✅ NLP splitting complete! {len(sentences)} sentences generated[/green]')
    console.print(f'[green]💾 Saved to: {_3_1_SPLIT_BY_NLP}[/green]')

    return sentences
```

**Step 4: 提交**

```bash
git add core/_3_1_split_nlp.py
git commit -m "feat: add nlp_split_to_sentences with char position tracking"
```

---

## Task 4: 创建 LLM 断句函数（拆分 Sentence.chunks）

**Files:**
- Modify: `core/_3_2_split_meaning.py`

**Step 1: 添加必要的导入**

在文件顶部添加：

```python
from core.models import Sentence
```

**Step 2: 添加 [br] 位置解析函数**

在文件中添加：

```python
def parse_br_positions(llm_output: str) -> List[int]:
    """
    解析 LLM 输出中 [br] 标记的位置

    Args:
        llm_output: LLM 返回的带 [br] 标记的文本

    Returns:
        [br] 在 LLM 输出中的字符位置列表
    """
    import re
    return [m.start() for m in re.finditer(r'\[br\]', llm_output)]


def find_br_positions_in_original(llm_output: str, original_text: str) -> List[int]:
    """
    使用 difflib 找到 [br] 在原始句子中的字符位置

    Args:
        llm_output: LLM 返回的带 [br] 标记的文本
        original_text: 原始句子文本

    Returns:
        [br] 在原始文本中的字符位置列表
    """
    import difflib
    from core.utils.sentence_tools import clean_word

    # 1. 找到 [br] 在 LLM 输出中的位置
    br_positions = parse_br_positions(llm_output)

    if not br_positions:
        return []

    # 2. 清洗文本用于匹配
    llm_clean = clean_word(llm_output.replace('[br]', ''))
    original_clean = clean_word(original_text)

    # 3. 使用 difflib 匹配
    s = difflib.SequenceMatcher(None, llm_clean, original_clean, autojunk=False)
    matching_blocks = s.get_matching_blocks()

    # 4. 建立 LLM 输出位置到原始文本位置的映射
    llm_to_original = {}
    current_original_pos = 0

    for llm_start, orig_start, length in matching_blocks:
        if length == 0:
            continue
        for i in range(length):
            llm_to_original[llm_start + i] = orig_start + i

    # 5. 找到每个 [br] 对应的原始位置
    original_br_positions = []
    for br_pos in br_positions:
        # [br] 在清洗后文本中的位置（需要去除之前的 [br]）
        llm_before_br = llm_output[:br_pos]
        llm_clean_before_br = clean_word(llm_before_br.replace('[br]', ''))

        if llm_clean_before_br in llm_to_original:
            original_pos = llm_to_original[llm_clean_before_br]
            original_br_positions.append(original_pos)

    return original_br_positions


def split_sentence_by_br(sentence: Sentence, llm_output: str) -> List[Sentence]:
    """
    根据 LLM 返回的 [br] 标记拆分 Sentence

    Args:
        sentence: 原始 Sentence 对象
        llm_output: LLM 返回的带 [br] 标记的文本

    Returns:
        拆分后的 Sentence 对象列表
    """
    # 1. 找到 [br] 在原始句子中的位置
    br_positions = find_br_positions_in_original(llm_output, sentence.text)

    if not br_positions:
        # 没有需要拆分的地方
        return [sentence]

    # 2. 构建字符到 Chunk 的映射
    char_to_chunk = []
    for chunk_idx, chunk in enumerate(sentence.chunks):
        char_to_chunk.extend([chunk_idx] * len(chunk.text))

    # 3. 根据 [br] 位置确定 Chunk 拆分点
    split_points = [0]  # 起始点
    for br_pos in br_positions:
        if br_pos < len(char_to_chunk):
            chunk_idx = char_to_chunk[br_pos]
            if chunk_idx not in split_points:
                split_points.append(chunk_idx)
    split_points.append(len(sentence.chunks))  # 结束点
    split_points = sorted(set(split_points))

    # 4. 拆分 Chunks，创建新的 Sentence 对象
    new_sentences = []
    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]

        if start_idx >= end_idx:
            continue

        sub_chunks = sentence.chunks[start_idx:end_idx]

        new_sentence = Sentence(
            chunks=sub_chunks,
            text="".join(c.text for c in sub_chunks),
            start=sub_chunks[0].start,
            end=sub_chunks[-1].end,
            index=sentence.index + i,
            is_split=True
        )
        new_sentences.append(new_sentence)

    return new_sentences
```

**Step 3: 提交**

```bash
git add core/_3_2_split_meaning.py
git commit -m "feat: add split_sentence_by_br with difflib matching"
```

---

## Task 5: 修改 parallel_split_sentences 使用 Sentence 对象

**Files:**
- Modify: `core/_3_2_split_meaning.py`

**Step 1: 修改 parallel_split_sentences 函数签名**

```python
def parallel_split_sentences(sentences: List[Sentence], max_length, max_workers, retry_attempt=0):
    """使用 LLM 并行拆分长句"""
    # ... 实现保持不变，但改为操作 Sentence 对象
```

**Step 2: 修改函数内部以使用 Sentence 对象**

在处理时使用 `sentence.text` 获取文本，拆分后调用 `split_sentence_by_br()`

**Step 3: 提交**

```bash
git add core/_3_2_split_meaning.py
git commit -m "refactor: parallel_split_sentences to use Sentence objects"
```

---

## Task 6: 简化 _6_gen_sub.py（直接使用 Sentence.start/end）

**Files:**
- Modify: `core/_6_gen_sub.py`

**Step 1: 添加从 Sentence 列表生成字幕的函数**

在文件中添加：

```python
def generate_subtitles_from_sentences(sentences: List[Sentence], subtitle_output_configs: list, output_dir: str, for_display: bool = True):
    """
    直接从 Sentence 对象列表生成字幕，不需要匹配

    Args:
        sentences: Sentence 对象列表
        subtitle_output_configs: 字幕输出配置
        output_dir: 输出目录
        for_display: 是否用于显示
    """
    df_trans_time = []

    for sent in sentences:
        df_trans_time.append({
            'Source': sent.text,
            'Translation': sent.translation,
            'timestamp': (sent.start, sent.end),
            'duration': sent.duration
        })

    # 转换为 DataFrame
    import pandas as pd
    df_trans_time = pd.DataFrame(df_trans_time)

    # 移除间隙
    for i in range(len(df_trans_time) - 1):
        delta_time = df_trans_time.loc[i + 1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (
                df_trans_time.loc[i, 'timestamp'][0],
                df_trans_time.loc[i + 1, 'timestamp'][0]
            )

    # 转换为 SRT 格式
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(
        lambda x: convert_to_srt_format(x[0], x[1])
    )

    # 美化字幕
    if for_display:
        import re
        import autocorrect_py as autocorrect
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(
            lambda x: autocorrect.format(re.sub(r'[，。]', ' ', str(x).strip()).strip())
        )

    # 输出字幕
    def generate_subtitle_string(df, columns):
        result = []
        for i, row in df.iterrows():
            def safe_get(col):
                val = row.get(col, '')
                return str(val).strip() if pd.notna(val) else ''

            line1 = safe_get(columns[0])
            line2 = safe_get(columns[1]) if len(columns) > 1 else ''
            result.append(f"{i+1}\n{row['timestamp']}\n{line1}\n{line2}\n\n")
        return ''.join(result).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)

    return df_trans_time
```

**Step 2: 修改 align_timestamp_main 函数**

```python
def align_timestamp_main():
    """主函数：加载 Sentence 对象并生成字幕"""
    df_text = safe_read_csv(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = safe_read_csv(_5_SPLIT_SUB)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)

    # 从 CSV 重建 Sentence 对象（临时方案，后续改为从元数据加载）
    chunks = load_chunks()
    sentences = []  # 需要从 translation CSV 重建

    # TODO: 这里需要从 translation CSV 重建 Sentence 对象
    # 临时方案：使用现有的 difflib 匹配
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, _OUTPUT_DIR)
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # 合并空字幕
    merge_empty_subtitle()
```

**Step 3: 提交**

```bash
git add core/_6_gen_sub.py
git commit -m "refactor: add generate_subtitles_from_sentences function"
```

---

## Task 7: 更新 Streamlit 入口以支持新的对象流程

**Files:**
- Modify: `st.py`

**Step 1: 检查现有的处理流程**

确保 `_3_1_split_nlp.py` 和 `_3_2_split_meaning.py` 返回 Sentence 对象

**Step 2: 提交**

```bash
git add st.py
git commit -m "refactor: update Streamlit flow for Sentence objects"
```

---

## Task 8: 测试端到端流程

**Step 1: 运行完整的处理流程**

```bash
# 运行 Streamlit 或直接调用模块
python -m core._2_asr
python -m core._3_1_split_nlp
python -m core._3_2_split_meaning
python -m core._4_1_summarize
python -m core._4_2_translate
python -m core._5_split_sub
python -m core._6_gen_sub
```

**Step 2: 验证输出**

检查 `output/` 目录下的字幕文件，确保时间戳正确

**Step 3: 提交测试配置**

```bash
git add .
git commit -m "test: verify end-to-end Sentence object flow"
```

---

## Task 9: 清理和优化

**Files:**
- Modify: 多个文件

**Step 1: 删除不再需要的 difflib 匹配代码**

除了 `_3_2_split_meaning.py` 中的 LLM 断句匹配，删除其他 difflib 相关代码

**Step 2: 添加类型提示**

确保所有函数都有正确的类型提示

**Step 3: 提交**

```bash
git add .
git commit -m "refactor: remove unused difflib matching code"
```

---

## Task 10: 更新文档

**Files:**
- Modify: `README.md`, `CLAUDE.md`

**Step 1: 更新架构说明**

添加 Chunk/Sentence 对象模型的说明

**Step 2: 提交**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update architecture for Chunk/Sentence objects"
```

---

## 验收标准

1. ✅ Chunk 和 Sentence 类定义完整
2. ✅ load_chunks() 能正确加载 Chunk 对象
3. ✅ NLP 分句使用字符位置追踪，生成 Sentence 对象
4. ✅ LLM 断句正确拆分 Sentence.chunks
5. ✅ 字幕生成直接使用 Sentence.start/end
6. ✅ 端到端流程正常运行
7. ✅ 时间戳准确（基于原始 ASR 数据）
8. ✅ 代码可读性提高，difflib 使用最小化

---

## 注意事项

1. **向后兼容**：保留现有的文本文件输出（split_by_nlp.txt 等）
2. **断点续传**：后续添加元数据保存/加载功能
3. **边界情况**：Chunk 内部文字被拆分的情况（标记为 TODO）
4. **测试**：每个阶段都要充分测试，确保时间戳准确
