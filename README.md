<div align="center">

<img src="/docs/logo.png" alt="VideoLingoLite Logo" height="140">

# Connect the World, Frame by Frame (Lite Version)

<a href="https://trendshift.io/repositories/12200" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12200" alt="Huanshere%2FVideoLingoLite | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[**English**](/README.md)｜[**简体中文**](/translations/README.zh.md)｜[**繁體中文**](/translations/README.zh-TW.md)｜[**日本語**](/translations/README.ja.md)｜[**Español**](/translations/README.es.md)｜[**Русский**](/translations/README.ru.md)｜[**Français**](/translations/README.fr.md)

</div>

## 🌟 Overview ([Try VL Now!](https://videolingo.io))

VideoLingoLite is a streamlined video translation and localization tool focused on generating high-quality subtitles. This lite version removes heavy features like video subtitle burning and local Whisper ASR to provide a more lightweight and efficient solution. It eliminates stiff machine translations and multi-line subtitles, enabling global knowledge sharing across language barriers.

**Note: VideoLingoLite is a streamlined version of the original [VideoLingo](https://github.com/Huanshere/VideoLingo) project.**

Key features:
- 🎥 YouTube video download via yt-dlp

- **🎙️ Word-level and Low-illusion subtitle recognition with cloud ASR (Gemini)**

- **📝 NLP and AI-powered subtitle segmentation**

- **📚 Custom + AI-generated terminology for coherent translation**

- **🔄 3-step Translate-Reflect-Adaptation for cinematic quality**

- **✅ Netflix-standard, Single-line subtitles Only**

- 🚀 One-click startup and processing in Streamlit

- 🌍 Multi-language support in Streamlit UI

- 📝 Detailed logging with progress resumption

Difference from similar projects: **Single-line subtitles only, superior translation quality**

## 🎥 Demo

<table>
<tr>
<td width="33%">

### Dual Subtitles
---
https://github.com/user-attachments/assets/a5c3d8d1-2b29-4ba9-b0d0-25896829d951

</td>
<td width="33%">

### Cosy2 Voice Clone
---
https://github.com/user-attachments/assets/e065fe4c-3694-477f-b4d6-316917df7c0a

</td>
<td width="33%">

### GPT-SoVITS with my voice
---
https://github.com/user-attachments/assets/47d965b2-b4ab-4a0b-9d08-b49a7bf3508c

</td>
</tr>
</table>

### Language Support

**Input Language Support(more to come):**

🇺🇸 English 🤩 | 🇷🇺 Russian 😊 | 🇫🇷 French 🤩 | 🇩🇪 German 🤩 | 🇮🇹 Italian 🤩 | 🇪🇸 Spanish 🤩 | 🇯🇵 Japanese 😐 | 🇨🇳 Chinese* 😊

> *Chinese uses a separate punctuation-enhanced ASR model, for now...

**Translation supports all languages, while dubbing language depends on the chosen TTS method.**

## Installation

Meet any problem? Chat with our free online AI agent [**here**](https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh) to help you.

> **Note:** FFmpeg is required. Please install it via package managers:
> - Windows: ```choco install ffmpeg``` (via [Chocolatey](https://chocolatey.org/))
> - macOS: ```brew install ffmpeg``` (via [Homebrew](https://brew.sh/))
> - Linux: ```sudo apt install ffmpeg``` (Debian/Ubuntu)

1. Clone the repository

```bash
git clone https://github.com/Huanshere/VideoLingoLite.git
cd VideoLingoLite
```

2. Install dependencies(requires `python=3.10`)

```bash
conda create -n videolingo python=3.10.0 -y
conda activate videolingo
python install.py
```

3. Start the application

```bash
streamlit run st.py
```

### Docker
Alternatively, you can use Docker (requires NVIDIA Driver version >550), see [Docker docs](/docs/pages/docs/docker.en-US.md):

```bash
docker build -t videolingo .
docker run -d -p 8501:8501 --gpus all videolingo
```

## APIs
VideoLingoLite supports OpenAI-Like API format and various TTS interfaces:
- LLM: `claude-3-5-sonnet`, `gpt-4.1`, `deepseek-v3`, `gemini-2.0-flash`, ... (sorted by performance, be cautious with gemini-2.5-flash...)
- ASR: Use Gemini ASR service for transcription *Note: No local Whisper support in Lite version*
- TTS: `azure-tts`, `openai-tts`, `siliconflow-fishtts`, **`fish-tts`**, `GPT-SoVITS`, `edge-tts`, `*custom-tts`(You can modify your own TTS in custom_tts.py!)

> **Note:** VideoLingoLite works with various services - one API key for all services (LLM, ASR, TTS). No local processing required!

For detailed installation, API configuration, and batch mode instructions, please refer to the documentation: [English](/docs/pages/docs/start.en-US.md) | [中文](/docs/pages/docs/start.zh-CN.md)

## Differences from Full Version

VideoLingoLite is a streamlined version of the original VideoLingo project with the following changes:
- Removed dubbing/voice cloning functionality (TTS features)
- Removed local WhisperX speech recognition (uses only cloud ASR services)
- Removed video subtitle burning functionality (video processing)
- Removed GPU-accelerated video processing
- Simplified dependencies for faster installation
- Focuses on core translation and subtitle generation functionality
- Lightweight interface for basic translation needs

## Current Limitations

1. ASR transcription performance may be affected by video background noise. For videos with loud background music, please enable Voice Separation Enhancement.

2. Using weaker models can lead to errors during processes due to strict JSON format requirements for responses (tried my best to prompt llm😊). If this error occurs, please delete the `output` folder and retry with a different LLM, otherwise repeated execution will read the previous erroneous response causing the same error.

3. The dubbing feature may not be 100% perfect due to differences in speech rates and intonation between languages, as well as the impact of the translation step. However, this project has implemented extensive engineering processing for speech rates to ensure the best possible dubbing results.

4. **Multilingual video transcription recognition will only retain the main language**. This is because ASR systems typically use a specialized model for a single language when processing subtitles, and will delete unrecognized languages.

5. **For now, cannot dub multiple characters separately**, as ASR speaker distinction capability is not sufficiently reliable.

## 📄 License

This project is licensed under the Apache 2.0 License. Special thanks to the following open source projects for their contributions:

[yt-dlp](https://github.com/yt-dlp/yt-dlp), [json_repair](https://github.com/mangiucugna/json_repair), [BELLE](https://github.com/LianjiaTech/BELLE), [OpenAI](https://github.com/openai)

## 📬 Contact Me

- Submit [Issues](https://github.com/Huanshere/VideoLingoLite/issues) or [Pull Requests](https://github.com/Huanshere/VideoLingoLite/pulls) on GitHub
- DM me on Twitter: [@Huanshere](https://twitter.com/Huanshere)
- Email me at: team@videolingo.io

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Huanshere/VideoLingoLite&type=Timeline)](https://star-history.com/#Huanshere/VideoLingoLite&Timeline)

---

<p align="center">If you find VideoLingoLite helpful, please give me a ⭐️!</p>
