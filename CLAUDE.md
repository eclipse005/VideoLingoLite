# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VideoLingoLite** is a lightweight audio/video transcription and translation tool that combines cloud-based (Gemini) and local (Qwen3-ASR) ASR with LLM-powered processing.

**Core Value Proposition**: Streamlined version of VideoLingo, removing unnecessary components while keeping core transcription/translation capabilities.

## Common Commands

### Development Setup
```bash
# Install dependencies (requires uv package manager)
uv sync

# Activate environment (if using conda/virtual env)
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
```

### Running the Application
```bash
# Method 1: One-click start (Windows)
OneKeyStart.bat

# Method 2: Streamlit directly
uv run python -m streamlit run st.py
# or simply
python st.py

# Method 3: Batch processing mode
cd batch && OneKeyBatch.bat
```

### Testing Changes
```bash
# Run Streamlit with auto-reload
uv run python -m streamlit run st.py --server.runOnSave true
```

## Architecture Overview

### Processing Pipeline (Numbered Modules)

The core processing follows a strict sequential pipeline defined by numbered modules in `core/`:

1. **_1_ytdlp.py** - Video file discovery and download (YouTube support)
2. **_2_asr.py** - ASR transcription (dual backend: Gemini/Qwen3-ASR)
3. **_3_1_split_nlp.py** - Stage 1: NLP-based sentence splitting (spaCy)
4. **_3_2_split_meaning.py** - Stage 2: LLM-powered semantic sentence splitting
5. **_4_1_summarize.py** - Content summarization and terminology extraction
6. **_4_2_translate.py** - Two-stage translation (literal → free translation)
7. **_5_split_sub.py** - Subtitle splitting based on length constraints
8. **_6_gen_sub.py** - Final subtitle generation (SRT format)

### Key Design Patterns

**Chunk/Sentence Object Architecture** (Core Data Model):
- Located in `core/utils/models.py` - defines `Chunk` and `Sentence` dataclasses
- **Chunk**: Word/character-level ASR unit with timestamps (text, start, end, speaker_id, index)
- **Sentence**: Sentence-level object composed of multiple Chunks (chunks, text, start, end, translation, is_split)
- Flow: ASR outputs Chunks → NLP splitting creates Sentences → LLM splitting preserves timestamps via Chunk references
- Benefits: Type safety, clear data ownership, easier debugging, proper timestamp preservation through object references

**difflib Alignment Algorithm** (Core Innovation):
- Located in `core/_3_2_split_meaning.py` and `core/_6_gen_sub.py`
- Problem: LLM may modify words, breaking timestamp alignment
- Solution: LLM only finds semantic boundaries → difflib matches back to original words → 100% preserves original timestamps
- Uses `difflib.SequenceMatcher` for fuzzy matching between LLM output and original ASR
- Applied at Chunk level when splitting Sentences in `_3_2_split_meaning.py`

**Multi-Language Adaptation**:
- CJK languages (zh, ja): Regex-based word segmentation, character-level splitting
- Space-separated languages (en, es, fr, de, it, ru): Standard word-level splitting
- Character weight system for subtitle length limits (CJK: 1.75x, Korean: 1.5x, English: 1x)

**Dual ASR Backend Architecture**:
- `core/asr_backend/gemini.py` - Cloud-based, RESTful API, audio chunking with offset correction
- `core/asr_backend/qwen3_asr.py` - Local GPU-only ASR (9 languages: zh/en/ja/ko/es/fr/de/it/ru)

### Module Dependencies

```
st.py (entry point)
  ├─> core/st_utils/          # Streamlit UI components
  ├─> core/utils/             # Shared utilities
  │    ├─> ask_gpt.py        # LLM API client with caching/retry
  │    ├─> config_utils.py   # Thread-safe config management
  │    ├─> models.py         # File path + Chunk/Sentence dataclass definitions
  │    └─> sentence_tools.py # Text cleaning and length calculation utilities
  ├─> core/_1_ytdlp.py through core/_6_gen_sub.py  # Processing pipeline
  └─> core/prompts.py        # Multi-language prompt templates
```

**Data Flow**:
```
_2_asr.py (transcribe)
  └─> List[Chunk] (word-level with timestamps)

_3_1_split_nlp.py (split_by_nlp)
  └─> List[Sentence] (sentence-level, Chunk references preserved)

_3_2_split_meaning.py (split_sentences_by_meaning)
  └─> List[Sentence] (split long sentences, Chunk references preserved)

_4_2_translate.py (translate_all)
  └─> Sentences with .translation field populated

_6_gen_sub.py (align_timestamp_main)
  └─> Final SRT files with timestamps from Chunk references
```

### Configuration System

**config.yaml Structure**:
- Multi-API support: `api`, `api_split`, `api_summary`, `api_translate`, `api_reflection`
- Nested key access supported (e.g., `asr.language`, `subtitle.max_length`)
- Thread-safe read/write via `core/utils/config_utils.py`
- Advanced settings (marked with `*`) are CLI-only, not exposed in Streamlit UI

**Key Configuration Groups**:
- `asr.runtime`: 'gemini' (cloud) or 'qwen' (local GPU)
- `target_language`: Natural language description for prompts
- `max_workers`: LLM concurrent requests (set to 1 for local LLMs)
- `subtitle.max_length`: Per-line character limit (default: 75)
- `language_split_with_space/without_space`: Defines word segmentation behavior

### Batch Processing System

**batch/tasks_setting.xlsx** Structure:
| Column | Description |
|--------|-------------|
| Video File | Filename (without `input/` prefix) or YouTube URL |
| Source Language | ISO 639-1 code or empty for default |
| Target Language | Natural language description or empty |
| Dubbing | 1 to enable, empty/0 to disable |
| Status | Auto-updated during processing |

**Input/Output**:
- Place video files in `batch/input/` folder
- Failed tasks moved to `batch/output/ERROR/`

**Error Recovery**:
- To retry failed task: Move folder from `batch/output/ERROR/` to root → rename to `output` → reprocess via Streamlit
- Dynamic `config.yaml` modification per task (language settings)

### Important Constraints

**Pandas Version Lock**:
- `pandas==2.2.3` for dependency stability

**PyTorch Installation**:
- Platform-specific: CUDA 12.8 for Windows/Linux x86_64, CPU for macOS/other
- Managed via `[[tool.uv.index]]` sections in `pyproject.toml`

**Python Version**: Locked to `==3.10.*`

### Code Style Guidelines (from .cursorrules)

- Use `# ------------` style for large block comments
- Avoid complex inline comments
- No type definitions in function variables
- Use English for comments and print statements
- Note: .cursorrules contains Chinese comments explaining these rules

### Output File Structure

After processing, `output/` directory contains:
- `src.srt` - Source language subtitles
- `trans.srt` - Translated subtitles
- `src_trans.srt` - Bilingual (source + translation)
- `trans_src.srt` - Bilingual (translation + source)

### Custom Terminology System

**custom_terms.xlsx**: User-defined terms for improving recognition and translation accuracy
- Processed during `_4_1_summarize.py` stage
- Integrated into translation prompts in `_4_2_translate.py`
- Output terminology available at `output/log/terminology.json`

### LLM Response Validation

**valid_def Pattern**: Multiple validation functions throughout codebase:
- JSON schema validation for structured outputs
- Retry mechanism (up to 3-5 attempts)
- Fallback to `json-repair` library for malformed JSON
- Critical in `_3_2_split_meaning.py` (sentence boundary detection) and `_5_split_sub.py` (alignment validation)

### Testing

**test_sentence_objects.py**: Unit tests for Chunk/Sentence object model
- Tests Chunk creation and duration calculation
- Tests Sentence creation with Chunk references
- Tests timestamp preservation through update_timestamps()
- Tests Sentence splitting logic with is_split flag
- Run with: `python test_sentence_objects.py`

### Recent Architecture Changes

- **Added (2025-01)**: Chunk/Sentence object architecture for type-safe data flow
  - `core/utils/models.py`: Chunk and Sentence dataclasses
  - `core/_3_1_split_nlp.py`: Returns List[Sentence] with Chunk references
  - `core/_3_2_split_meaning.py`: Splits Sentences while preserving Chunk timestamps
  - Benefits: Improved timestamp preservation, better debugging, clearer data ownership
- **Removed**: Pause-based sentence splitting (reverted to LLM-only semantic splitting)
- **Removed**: i18n/internationalization (switched to Chinese-only UI)
- **Enhanced**: Multi-language support for CJK languages in sentence splitting
- **Enhanced**: Qwen3-ASR local attention for long audio
