# VideoLingoLite

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**VideoLingoLite** 是“VideoLingo”的轻量化版本，集成了云端与本地（WhisperX）ASR（自动语音识别）和翻译功能，为需要音视频转写、自动翻译的用户提供了灵活高效的解决方案。

## 项目简介

VideoLingoLite 专注于音视频内容的自动转写与翻译：
- 支持音频、视频文件的批量自动转写为文本。
- 支持云端及本地（WhisperX）ASR（自动语音识别）与翻译能力。
- 提供术语自定义表（`custom_terms.xlsx`）以优化行业/领域专用词识别和翻译结果。
- 精简化设计，部署和使用简单。

> 该项目针对有转写和翻译需求的科研、教学、媒体等场景，去除了原VideoLingo平台的多余组件，仅保留核心处理能力。

## 主要功能

- **混合ASR支持**：支持云端及高性能的本地（WhisperX）语音转写。
- **自动翻译**：支持多语言翻译，提升音视频内容可达性。
- **批量处理**：可通过 `batch` 目录批量处理文件。
- **可扩展性**：自定义术语、灵活配置。

## 目录结构

```
.
├── .claude/                # Claude相关配置
├── .cursorrules            # 编辑相关规则
├── .gitignore              # Git忽略规则
├── .streamlit/             # Streamlit配置（可能用于界面）
├── LICENSE                 # 开源许可证
├── OneKeyStart.bat         # 一键启动脚本（Windows）
├── README.md               # 项目说明文档
├── batch/                  # 批量处理脚本或输入输出目录
├── config.yaml             # 主要配置文件
├── core/                   # 核心功能模块
├── custom_terms.xlsx       # 自定义术语表
├── pyproject.toml          # 项目配置与依赖管理 (uv)
├── st.py                   # 主要入口脚本
├── uv.lock                 # uv锁文件
└── translations/           # 翻译相关文件夹
```

## 快速开始

1. **环境准备**

   1. 安装 uv 包管理器（如果尚未安装）：
      ```bash
      # Windows 安装方式
      # 从 GitHub releases 下载: https://github.com/astral-sh/uv/releases
      # 或使用 pip 安装
      pip install uv

      # 更多安装方式请参考官方文档: https://docs.astral.sh/uv/
      ```

   2. 下载项目到本地：
      ```bash
      git clone https://github.com/eclipse005/VideoLingoLite.git
      cd VideoLingoLite
      ```

   3. 项目目录运行 uv sync 安装依赖：
      ```bash
      uv sync
      ```

2. **运行项目**

   - Windows用户可直接双击 `OneKeyStart.bat`。
   - 或运行主脚本：
     ```bash
     python st.py
     ```

3. **自定义配置**

   - 编辑 `config.yaml` 设置云端API、语言选项等参数。
   - 修改或添加 `custom_terms.xlsx` 以适配专业术语。

## 贡献与许可证

- 本项目遵循 [Apache 2.0](LICENSE) 协议。
- 欢迎提交PR、反馈问题。

## 联系

详见[项目主页](https://github.com/eclipse005/VideoLingoLite)获取最新信息与交流方式。