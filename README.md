# VideoLingoLite

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **VideoLingo** 的轻量化版本，专注于音视频转写与翻译

**VideoLingoLite** 是 [VideoLingo](https://github.com/Huanshere/VideoLingo) 的精简版本，保留了核心的转写翻译能力，去除冗余功能，让字幕制作更加轻量高效。

---

## 核心功能

### 语音转文字
- **Qwen3-ASR 引擎**：本地 GPU 加速，支持中英日韩等 9 种语言
- **分离架构设计**：转录与对齐分离加载，降低显存占用（峰值显存更小）
- **人声分离**：嘈杂环境下自动分离人声，大幅提升转录准确率
- **词级时间戳**：精确到每个词的时间定位
- **只转录模式**：跳过翻译，仅生成原文字幕

### ASR 术语矫正
- **自动矫正**：根据配置的术语列表，自动修正 ASR 识别错误（L L M → LLM）
- **适用场景**：视频中的专业术语、人名、地名等容易被语音识别错误时

### 智能翻译
- **两阶段翻译**：先直译，再意译，确保译文自然流畅
- **上下文感知**：结合前后文和术语表进行翻译
- **自定义术语**：支持导入专业术语表，确保术语翻译准确
- **自动摘要**：提取视频主题和关键术语

### 断点续跑
- **智能缓存**：自动缓存处理进度，支持从任意阶段继续
- **对象序列化**：完整的 Sentence 对象状态保存，断点恢复无损失
- **灵活重跑**：清除缓存即可重新执行特定阶段

### 多语言支持
- **9 种语言**：中文、英语、日语、韩语、西班牙语、法语、德语、意大利语、俄语
- **CJK 优化**：针对中日韩语言的特殊分词和对齐处理
- **自适应分词**：空格分隔语言与 CJK 语言分别优化
- **智能连接符**：根据语言类型自动添加空格（如英语）或不添加（如中文）

### 字幕生成
- **4 种格式**：源语言字幕、翻译字幕、双语字幕（两种顺序）
- **智能对齐**：字幕长度自动适配，可读性更强
- **自动优化**：合并重复字幕、清理冗余内容
- **混排美化**：中英/日英混排自动添加空格，提升可读性

### 批量处理
- **CSV 配置**：通过表格配置多个视频任务
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

**Windows 用户**：双击运行 `OneKeyStart.bat`

启动后自动打开浏览器访问 Web UI

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
2. 编辑 `batch/tasks_setting.csv` 配置每个任务的语言设置
3. 运行 `batch/OneKeyBatch.bat`

| 配置项 | 说明 |
|--------|------|
| Video File | 视频文件名或 YouTube 链接 |
| Source Language | 源语言（如 en, zh） |
| Target Language | 目标语言（如 English, 简体中文） |
| Dubbing | 是否启用配音 |

---

## 自定义术语

编辑 `custom_terms.csv` 添加专业术语，提升翻译准确率：

| 源语言术语 | 目标语言翻译 | 解释说明 |
|------------|-------------|----------|
| Machine Learning | 机器学习 | AI 领域术语 |
| Transformer | Transformer | 深度学习架构 |

---

## 系统要求

### 最低配置
- Python 3.10+
- NVIDIA GPU（4GB+ 显存）
- CUDA 12.x 支持

### 推荐配置
- **4GB+ 显存**：使用 Qwen3-ASR-0.6B 模型（速度更快）
- **8GB+ 显存**：使用 Qwen3-ASR-1.7B 模型（准确率更高）

---

## 输出文件

处理完成后，`output/` 目录包含：

- `src.srt` - 源语言字幕
- `trans.srt` - 翻译字幕
- `src_trans.srt` - 双语字幕（源+译）
- `trans_src.srt` - 双语字幕（译+源）

---

## 特色技术

- **Chunk/Sentence 对象架构**：类型安全的数据流设计，完整保留时间戳和状态
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
