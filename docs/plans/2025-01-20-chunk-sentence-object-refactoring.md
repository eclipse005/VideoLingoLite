# Chunk/Sentence 对象化重构设计

## 概述

将 VideoLingoLite 的数据处理从"文本处理 + difflib 时间对齐"模式重构为"对象化处理"模式。核心思想：**Chunk 对象携带时间戳，Sentence 对象组合 Chunk，时间戳始终跟随对象引用**。

## 核心目标

1. **消除大部分 difflib 匹配** - 仅在 LLM 断句时保留一次 difflib
2. **时间戳 100% 准确** - 基于原始 ASR 数据，不依赖字符串匹配
3. **代码可读性** - `sentence.start/end` 直接取值，不需要复杂匹配逻辑

## 数据结构

### Chunk 类

```python
@dataclass
class Chunk:
    text: str           # 词/字内容（如 "Hello", "world."）
    start: float        # 开始时间（秒）
    end: float          # 结束时间（秒）
    speaker_id: str | None = None
    index: int = 0      # 在 cleaned_chunks.csv 中的行号
```

### Sentence 类

```python
@dataclass
class Sentence:
    chunks: List[Chunk]     # 组成这句话的所有 Chunk
    text: str               # 从 chunks 拼接的完整文本
    start: float            # = chunks[0].start
    end: float              # = chunks[-1].end
    translation: str = ""   # 翻译文本
    index: int = 0          # 句子序号
    is_split: bool = False  # 是否被 LLM 切分过

    @property
    def duration(self) -> float:
        return self.end - self.start
```

## 处理流程

```
cleaned_chunks.csv
    ↓
1. load_chunks() → List[Chunk]
    ↓
2. NLP 分句 → List[Sentence]  # 字符位置追踪，Chunk 组合成 Sentence
    ↓
3. LLM 切分长句 → List[Sentence]  # 拆分 Sentence.chunks，重建 Sentence（需要 difflib）
    ↓
4. 总结 + 术语提取
    ↓
5. 翻译 → Sentence.translation 填充
    ↓
6. 断句 → 进一步拆分 Sentence.chunks
    ↓
7. 对齐 → LLM 将译文按原文 [br] 位置对齐拆分
    ↓
8. 生成字幕 → 直接用 Sentence.start/end
```

## 关键设计点

### 1. NLP 分句 - 字符位置追踪

不需要 difflib，使用字符位置映射：

```python
# 构建字符到 chunk 的映射
full_text = "".join(chunk.text for chunk in chunks)
char_to_chunk = []
for chunk_idx, chunk in enumerate(chunks):
    char_to_chunk.extend([chunk_idx] * len(chunk.text))

# NLP 分句后，根据字符位置找到 chunk 范围
for sent in nlp(full_text).sents:
    start_chunk_idx = char_to_chunk[sent.start_char]
    end_chunk_idx = char_to_chunk[sent.end_char - 1]
    sentence_chunks = chunks[start_chunk_idx:end_chunk_idx + 1]
    sentences.append(Sentence(chunks=sentence_chunks, ...))
```

### 2. LLM 断句 - difflib 匹配

LLM 返回带 `[br]` 标记的文本，需要匹配回原始 Chunks：

```python
def split_by_br_tags(sentence: Sentence, llm_output: str) -> List[Sentence]:
    # 1. 找到 [br] 在 LLM 输出中的位置
    br_positions = [m.start() for m in re.finditer(r'\[br\]', llm_output)]

    # 2. 用 difflib 匹配，找到 [br] 对应原始句子的位置
    s = difflib.SequenceMatcher(None, sentence.text, llm_output, autojunk=False)
    # ... 匹配逻辑 ...

    # 3. 根据匹配结果，找到对应的 Chunk 索引
    # 4. 拆分 sentence.chunks，创建新的 Sentence 对象
```

### 3. 翻译和对齐

- **翻译**：填充 `Sentence.translation`，不影响时间戳
- **对齐**：LLM 根据原文的 `[br]` 位置，把译文也对应拆分
- **时间戳**：译文共用原文时间（`sentence.start/end`）

### 4. 字幕生成

大幅简化，不需要匹配：

```python
def generate_subtitles(sentences: List[Sentence]):
    for sent in sentences:
        start_time = sent.start
        end_time = sent.end
        text = sent.text
        translation = sent.translation
        # 直接输出 SRT 格式
```

## 文件保存/加载策略

使用元数据保存法支持断点续传：

```python
# 保存 Sentence 元数据
def save_sentence_metadata(sentences: List[Sentence], filepath: str):
    data = [{
        'index': sent.index,
        'start_chunk_idx': sent.chunks[0].index,
        'end_chunk_idx': sent.chunks[-1].index,
        'text': sent.text,
        'is_split': sent.is_split
    } for sent in sentences]
    pd.DataFrame(data).to_csv(filepath, index=False)

# 加载并重建 Sentence 对象
def load_sentence_metadata(chunks: List[Chunk], filepath: str) -> List[Sentence]:
    df = safe_read_csv(filepath)
    sentences = []
    for _, row in df.iterrows():
        sentence_chunks = chunks[row['start_chunk_idx']:row['end_chunk_idx']+1]
        sentences.append(Sentence(
            chunks=sentence_chunks,
            text=row['text'],
            start=sentence_chunks[0].start,
            end=sentence_chunks[-1].end,
            index=row['index'],
            is_split=row['is_split']
        ))
    return sentences
```

**优势**：Chunk 对象始终从 `cleaned_chunks.csv` 加载，确保一致性。

## 实施步骤

### Phase 1: 基础设施
- `core/models.py` - 添加 `Chunk`, `Sentence` 类
- `core/_2_asr.py` - 添加 `load_chunks()` 函数

### Phase 2: NLP 分句改造
- `core/_3_1_split_nlp.py` - 字符位置追踪，创建 Sentence 对象

### Phase 3: LLM 断句改造
- `core/_3_2_split_meaning.py` - 拆分 `Sentence.chunks`，重建对象

### Phase 4: 翻译和断句
- `core/_4_2_translate.py` - 填充 `Sentence.translation`
- `core/_5_split_sub.py` - 拆分 `Sentence.chunks`

### Phase 5: 字幕生成
- `core/_6_gen_sub.py` - 大幅简化，直接用 `Sentence.start/end`

### Phase 6: 清理
- 删除不再需要的 difflib 匹配代码（除 LLM 断句外）

## 已知 TODO

1. **Chunk 边界切分问题**
   - 场景：单个 Chunk 的内部文字（如 "了。"）被拆分到不同句子
   - 处理：预防性检查，如果分句边界落在 Chunk 内部，调整到 Chunk 边界
   - 状态：后续出现时处理

2. **LLM 修改文字**
   - 场景：LLM 断句时可能修改标点或措辞
   - 处理：在 `_3_2_split_meaning.py` 中保留 difflib 匹配
   - 状态：设计已包含

## 优势总结

| 方面 | 现有方案 | 重构后 |
|------|----------|--------|
| 时间戳准确性 | 依赖 difflib 匹配 | 100% 基于 ASR 原始数据 |
| 代码复杂度 | 多处 difflib 匹配逻辑 | 仅一处（LLM 断句） |
| 可维护性 | 字符串匹配分散 | 对象引用清晰 |
| 调试难度 | 匹配失败难以定位 | 对象状态可追踪 |
| 断点续传 | 文本文件 | 元数据 + Chunk 索引 |

## 风险点

1. **字符位置追踪** - 需要充分测试多语言和边界情况
2. **断点续传逻辑** - 需要验证元数据加载的正确性
3. **向后兼容** - 现有中间文件格式可能需要调整
