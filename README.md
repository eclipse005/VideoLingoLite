# VideoLingoLite

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**VideoLingoLite** 是 "VideoLingo" 的轻量化版本，专注于音视频转写与翻译，集成了云端（Gemini API）和本地（NVIDIA Parakeet）ASR 能力。

## 项目简介

VideoLingoLite 是一个高效的音视频自动化处理工具：
- **双 ASR 后端**：云端 Gemini API（支持多语言）+ 本地 Parakeet（25 种欧洲语言，需 NVIDIA GPU）
- **LLM 驱动**：使用大语言模型进行语义分句、总结和多步翻译
- **批量处理**：支持批量任务自动化处理
- **精简设计**：去除配音、本地 ASR（非 GPU）、视频处理等冗余功能

## 处理流程

VideoLingoLite 采用 6 步处理流水线：

```
1. 语音识别转录 (ASR)
   ├─ Gemini API (云端，多语言)
   └─ Parakeet (本地，NVIDIA GPU)

2. LLM 语义分句
   └─ 使用 difflib 对齐保留原始时间戳

3. 长句智能切分
   └─ CJK 语言支持，避免单词内切分

4. 内容总结与术语提取
   └─ 生成摘要和领域术语表

5. 多步翻译
   └─ 直译 → 意译两阶段处理

6. 字幕生成
   └─ 长度限制对齐，生成 SRT 格式字幕
```

## 主要功能

- **混合 ASR**：云端 Gemini + 本地 Parakeet 双后端
- **语义分句**：LLM 驱动的句子边界检测
- **多步翻译**：直译 + 意译，确保翻译质量
- **自定义术语**：支持 `custom_terms.xlsx` 优化专业术语
- **批量处理**：通过 `batch/` 目录进行任务自动化
- **多语言支持**：CJK 语言（中日韩）特殊处理

## 目录结构

```
.
├── .cursorrules            # 编辑器规则
├── .gitignore              # Git 忽略规则
├── .streamlit/             # Streamlit 配置
├── LICENSE                 # Apache 2.0 许可证
├── OneKeyStart.bat         # 一键启动脚本 (Windows)
├── README.md               # 项目说明文档
├── batch/                  # 批量处理目录
│   ├── input/              # 输入文件放置处
│   ├── output/             # 处理结果输出处
│   └── tasks_setting.xlsx  # 批量任务配置
├── config.yaml             # 主配置文件
├── core/                   # 核心处理模块
│   ├── _1_ytdlp.py         # 视频下载
│   ├── _2_asr.py           # ASR 转录
│   ├── _3_llm_sentence_split.py  # LLM 语义分句
│   ├── _3_2_split_meaning.py      # 长句切分
│   ├── _4_1_summarize.py   # 总结与术语提取
│   ├── _4_2_translate.py   # 多步翻译
│   ├── _5_split_sub.py     # 字幕长度对齐
│   ├── _6_gen_sub.py       # 最终字幕生成
│   ├── asr_backend/        # ASR 后端实现
│   │   ├── gemini.py       # Gemini API
│   │   └── parakeet_local.py  # Parakeet 本地模型
│   ├── prompts.py          # 多语言提示词模板
│   └── utils/              # 工具函数
├── custom_terms.xlsx       # 自定义术语表模板
├── pyproject.toml          # 项目依赖 (uv)
├── st.py                   # Streamlit 主入口
└── ui/                     # UI 组件
```

## 快速开始

### 1. 环境准备

```bash
# 安装 uv 包管理器
pip install uv

# 克隆项目
git clone https://github.com/eclipse005/VideoLingoLite.git
cd VideoLingoLite

# 安装依赖
uv sync
```

### 2. 配置设置

编辑 `config.yaml`：
- **ASR 后端**：`asr.runtime: gemini` (云端) 或 `parakeet` (本地 GPU)
- **API 配置**：设置 Gemini API 密钥
- **语言设置**：配置源语言和目标语言
- **并发控制**：`max_workers` 控制 LLM 并发数

### 3. 运行项目

**交互式模式**：
```bash
# Windows
OneKeyStart.bat

# 跨平台
uv run python -m streamlit run st.py
```

**批处理模式**：
1. 将视频文件放入 `batch/input/`
2. 编辑 `batch/tasks_setting.xlsx` 配置任务
3. 运行 `batch/OneKeyBatch.bat`

### 4. 自定义术语

编辑 `custom_terms.xlsx` 添加领域术语，优化识别和翻译准确率。

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `asr.runtime` | ASR 后端 (gemini/parakeet) | gemini |
| `asr.language` | 源语言 | zh |
| `target_language` | 目标语言描述 | English |
| `max_workers` | LLM 并发数 | 10 |
| `subtitle.max_length` | 单行字数限制 | 75 |
| `max_split_length` | 长句切分阈值 | 40 |

## 系统要求

**Gemini 模式**（云端）：
- Python 3.10+
- 网络连接

**Parakeet 模式**（本地）：
- Python 3.10+
- NVIDIA GPU (CUDA 支持)
- 8GB+ 显存

## 贡献与许可证

- 本项目遵循 [Apache 2.0](LICENSE) 协议
- 欢迎提交 PR、反馈问题

## 联系

详见[项目主页](https://github.com/eclipse005/VideoLingoLite)获取最新信息。
