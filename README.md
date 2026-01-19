# VideoLingoLite

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **VideoLingo** 的轻量化版本，专注于音视频转写与翻译

**VideoLingoLite** 是 [VideoLingo](https://github.com/Huanshere/VideoLingo) 的精简版本，保留了核心的转写翻译能力，去除冗余功能，让字幕制作更加轻量高效。

---

## 核心功能

### 语音转文字
- **双引擎支持**：云端 Gemini API（多语言）+ 本地 Parakeet（25 种欧洲语言）
- **人声分离**：嘈杂环境下自动分离人声，大幅提升转录准确率
- **词级时间戳**：精确到每个词的时间定位

### 智能翻译
- **两阶段翻译**：先直译，再意译，确保译文自然流畅
- **上下文感知**：结合前后文和术语表进行翻译
- **自定义术语**：支持导入专业术语表，确保术语翻译准确
- **自动摘要**：提取视频主题和关键术语

### 多语言支持
- **CJK 优化**：针对中日韩语言的特殊分词和对齐处理
- **25+ 欧洲语言**：本地 Parakeet 引擎支持
- **云端多语言**：Gemini API 支持全球主流语言

### 字幕生成
- **4 种格式**：源语言字幕、翻译字幕、双语字幕（两种顺序）
- **智能对齐**：字幕长度自动适配，可读性更强
- **自动优化**：合并重复字幕、清理冗余内容

### 批量处理
- **Excel 配置**：通过表格配置多个视频任务
- **自动化队列**：一键处理整个文件夹的视频
- **错误恢复**：失败任务自动隔离，方便重新处理

---

## 快速开始

### 安装依赖

```bash
# 1. 安装 uv 包管理器
pip install uv

# 2. 克隆项目
git clone https://github.com/eclipse005/VideoLingoLite.git
cd VideoLingoLite

# 3. 安装项目依赖
uv sync
```

### 启动应用

**Windows 用户**：双击 `OneKeyStart.bat`

**跨平台方式**：
```bash
uv run python -m streamlit run st.py
```

### 配置 API

首次运行需要在侧边栏配置：
- **API 地址**：你的 LLM 服务地址
- **API 密钥**：对应的 API Key
- **选择模型**：用于翻译的模型

---

## 使用场景

| 场景 | 说明 |
|------|------|
| **学习笔记** | 将课程视频转成文字字幕，方便复习 |
| **内容创作** | 为短视频快速制作双语字幕 |
| **会议记录** | 自动生成会议纪要和翻译 |
| **影视翻译** | 批量处理视频素材 |
| **无障碍服务** | 为听障人士生成字幕 |

---

## 批量处理

需要处理多个视频？使用批处理模式：

1. 将视频文件放入 `batch/input/` 目录
2. 编辑 `batch/tasks_setting.xlsx` 配置每个任务的语言设置
3. 运行 `batch/OneKeyBatch.bat`

| 配置项 | 说明 |
|--------|------|
| Video File | 视频文件名或 YouTube 链接 |
| Source Language | 源语言（如 en, zh） |
| Target Language | 目标语言（如 English, 简体中文） |
| Dubbing | 是否启用配音 |

---

## 自定义术语

编辑 `custom_terms.xlsx` 添加专业术语，提升翻译准确率：

| 源语言术语 | 目标语言翻译 | 解释说明 |
|------------|-------------|----------|
| Machine Learning | 机器学习 | AI 领域术语 |
| Transformer | Transformer | 深度学习架构 |

---

## 系统要求

### 基础模式（云端 Gemini）
- Python 3.10+
- 网络连接

### 本地模式（Parakeet）
- Python 3.10+
- NVIDIA GPU（8GB+ 显存）
- CUDA 支持

---

## 配置选项

在 `config.yaml` 中可调整：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `asr.runtime` | `gemini` | ASR 引擎：gemini（云端）/ parakeet（本地） |
| `asr.language` | `en` | 源语言 ISO 639-1 代码（如 en, zh, ja） |
| `target_language` | `English` | 目标语言（自然语言描述） |
| `max_workers` | `8` | 并发处理数（本地 LLM 建议设为 1） |
| `pause_split_threshold` | `1` | 停顿切分阈值（秒），0 或 null 表示禁用 |
| `vocal_separation.enabled` | `false` | 是否启用人声分离 |
| `spacy_model_map` | - | spaCy 模型映射（多语言支持） |

---

## 输出文件

处理完成后，`output/` 目录包含：

- `src.srt` - 源语言字幕
- `trans.srt` - 翻译字幕
- `src_trans.srt` - 双语字幕（源+译）
- `trans_src.srt` - 双语字幕（译+源）

---

## 特色技术

- **两阶段分句架构**：NLP 语言学规则（快速）+ LLM 语义切分（精准），兼顾速度与质量
  - Stage 1 (spaCy)：标点、逗号、连接词、停顿间隙、根词等多维度规则切分
  - Stage 2 (LLM)：仅对超长句进行语义分析，大幅降低 API 成本
- **difflib 对齐算法**：100% 保留原始时间戳，字幕定位精准
- **多语言自适应分词**：CJK 语言与空格分隔语言分别优化
- **3 轮迭代切分**：确保所有长字幕符合长度限制

---

## 开源协议

本项目采用 [Apache 2.0](LICENSE) 协议开源

基于 [VideoLingo](https://github.com/Huanshere/VideoLingo) 改造

---

## 联系我们

- 项目主页：[https://github.com/eclipse005/VideoLingoLite](https://github.com/eclipse005/VideoLingoLite)
- 原版项目：[https://github.com/Huanshere/VideoLingo](https://github.com/Huanshere/VideoLingo)
- 问题反馈：提交 Issue 或 Pull Request

---

<div align="center">

**让字幕制作从此轻松高效**

</div>
