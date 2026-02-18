# ------------
# Qwen3-ASR Backend (Separated Architecture)
# Stage 1: ASR Transcription (text only)
# Stage 2: Forced Alignment (timestamps)
# ------------

import os
import sys
import logging
import warnings
import gc
import unicodedata
from typing import Optional, List, Tuple

# ------------
# Suppress warnings (MUST be before library imports)
# ------------

# Suppress transformers warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

for logger_name in ['transformers', 'qwen_asr']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

# Now safe to import libraries
import torch
import librosa
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from core.utils import load_key, except_handler

# Model cache directory
MODEL_DIR = load_key("model_dir")
ABS_MODEL_DIR = os.path.abspath(MODEL_DIR)


def _is_word_char(char: str) -> bool:
    """
    判断字符是否是"词字符"（使用 Unicode 标准分类）

    基于 unicodedata.category():
    - L* (Letter): 字母 - 保留
    - N* (Number): 数字 - 保留
    - P* (Punctuation): 标点 - 跳过
    - S* (Symbol): 符号 - 跳过
    - Z* (Separator): 分隔符 - 跳过（包括空格）
    - C* (Control): 控制字符 - 跳过
    """
    if char.isspace():
        return False

    category = unicodedata.category(char)

    # 保留字母(L*)和数字(N*)
    # 这自动支持：中文、日文、韩文、英文、阿拉伯文等所有语言
    if category.startswith('L') or category.startswith('N'):
        return True

    return False


def _align_text_to_timestamps(full_text: str, word_timestamps: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    将完整文本的所有字符（包括标点等）对齐到单词时间戳

    使用 difflib 进行模糊匹配，处理格式不一致问题（如 "seventyone" vs "seventy-one"）

    Args:
        full_text: 完整文本，包含所有字符
        word_timestamps: [(word, start_time, end_time), ...]

    Returns:
        [(text_with_following_chars, start_time, end_time), ...]
    """
    if not word_timestamps or not full_text:
        return []

    # Build cleaned version of full_text for matching (remove spaces but keep char positions)
    # We need to maintain position mapping
    full_text_chars = list(full_text)
    clean_to_original_map = []  # Maps clean position -> original position
    original_to_clean_map = []  # Maps original position -> clean position

    for i, char in enumerate(full_text):
        if _is_word_char(char):
            original_to_clean_map.append(len(clean_to_original_map))
            clean_to_original_map.append(i)
        else:
            original_to_clean_map.append(-1)  # Non-word chars map to -1

    clean_text = ''.join([full_text[i] for i in clean_to_original_map])

    # Clean each word from word_timestamps
    cleaned_words = []
    for word, start, end in word_timestamps:
        cleaned = ''.join([c for c in word if _is_word_char(c)])
        cleaned_words.append(cleaned)

    # Build result using word-level alignment
    result = []
    word_idx = 0
    clean_pos = 0  # Position in clean_text

    while word_idx < len(word_timestamps) and clean_pos < len(clean_text):
        current_clean_word = cleaned_words[word_idx]

        # Try to find current_clean_word in clean_text starting at clean_pos
        if clean_text[clean_pos:clean_pos + len(current_clean_word)].lower() == current_clean_word.lower():
            # Found match - calculate original position
            match_start_clean = clean_pos
            match_end_clean = clean_pos + len(current_clean_word)

            # Map back to original position
            original_start = clean_to_original_map[match_start_clean]
            original_end = clean_to_original_map[match_end_clean - 1] + 1

            # Collect following non-word characters (punctuation, skip spaces)
            following_original_end = original_end
            # First skip any spaces after the word
            while following_original_end < len(full_text) and full_text[following_original_end].isspace():
                following_original_end += 1
            # Then collect punctuation
            while following_original_end < len(full_text):
                char = full_text[following_original_end]
                if _is_word_char(char):
                    break
                following_original_end += 1

            # Get the text segment (original word + following punctuation, no trailing spaces)
            text_segment = full_text[original_start:following_original_end].rstrip()

            original_word, start, end = word_timestamps[word_idx]
            result.append((text_segment, start, end))

            # Move past this word
            clean_pos = match_end_clean
            word_idx += 1
        else:
            # Try to skip ahead in clean_text to find a match
            # This handles cases where words appear in different order or with extra text
            clean_pos += 1

            # Safety: if we've gone too far without finding a match, advance word_idx
            if clean_pos >= len(clean_text):
                word_idx += 1
                clean_pos = 0  # Reset and try to find next word

    # Fallback: if alignment failed significantly, use original approach
    if len(result) < len(word_timestamps) * 0.5:  # Less than 50% matched
        import os
        os.makedirs('output/log', exist_ok=True)
        with open('output/log/debug_alignment_error.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n=== Alignment Partially Failed ===\n")
            f.write(f"Total word_timestamps: {len(word_timestamps)}\n")
            f.write(f"Matched words: {len(result)}\n")
            f.write(f"Using fallback method...\n")

        # Fallback: just use original words without punctuation attachment
        return [(w[0], w[1], w[2]) for w in word_timestamps]

    return result


def _convert_aligner_to_standard_asr_format(aligner_result, full_text: str, start_offset=0):
    """
    Convert Aligner output to standard ASR format with punctuation alignment

    Args:
        aligner_result: Qwen3ForcedAligner result (list of timestamps)
        full_text: Full text with punctuation
        start_offset: Time offset for this audio segment

    Returns:
        dict: Standard ASR format with segments
    """
    if not aligner_result:
        return {'segments': [], 'language': 'unknown'}

    # Build word timestamps from aligner result
    word_timestamps = [(ts.text, ts.start_time, ts.end_time) for ts in aligner_result]
    aligned_timestamps = _align_text_to_timestamps(full_text, word_timestamps)

    aligned_words = []
    for text, start, end in aligned_timestamps:
        aligned_words.append({
            'word': text,
            'start': round(start + start_offset, 2),
            'end': round(end + start_offset, 2)
        })

    if not aligned_words:
        return {'segments': [], 'language': 'unknown'}

    segment = {
        'start': round(aligned_words[0]['start'], 2),
        'end': round(aligned_words[-1]['end'], 2),
        'text': full_text,
        'words': aligned_words
    }

    return {'segments': [segment], 'language': 'unknown'}


def _download_model_if_needed(model_path, repo_id):
    """Download model from modelscope if not exists"""
    if os.path.exists(model_path):
        return

    rprint(f"[yellow]📥 Model not found locally. Downloading {repo_id}...[/yellow]")
    rprint(f"[dim]This may take a while on first run...[/dim]")

    try:
        from modelscope import snapshot_download
        os.makedirs(ABS_MODEL_DIR, exist_ok=True)
        snapshot_download(
            repo_id,
            local_dir=model_path,
            revision="master",
        )
        rprint(f"[green]✅ Downloaded {repo_id}[/green]")
    except ImportError:
        raise ImportError(
            "modelscope not installed. Please run: uv sync"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download {repo_id}: {e}")


def _load_qwen_asr_model():
    """Load Qwen3-ASR model ONLY (without forced_aligner) for transcription"""
    model_name = load_key("asr.model", default="Qwen3-ASR-1.7B")
    asr_model_path = os.path.join(ABS_MODEL_DIR, model_name)

    # Auto-download if missing
    _download_model_if_needed(asr_model_path, f"Qwen/{model_name}")

    rprint(f"[cyan]📥 Loading ASR model {model_name}...[/cyan]")

    try:
        from qwen_asr import Qwen3ASRModel

        # Detect device
        if torch.cuda.is_available():
            device = "cuda:0"
            rprint(f"[bold green]🚀 GPU Detected | Using CUDA acceleration[/bold green]")
        else:
            device = "cpu"
            rprint(f"[bold yellow]⚠️ No GPU found | Using CPU (will be slow)[/bold yellow]")

        # Initialize ASR model ONLY (no forced_aligner)
        asr_model = Qwen3ASRModel.from_pretrained(
            asr_model_path,
            dtype=torch.bfloat16,
            device_map=device,
            max_new_tokens=4096,  # Support long audio
            max_inference_batch_size=4,
        )

        rprint(f"[green]✅ ASR model loaded (no aligner)[/green]")
        return asr_model

    except ImportError:
        raise ImportError(
            "qwen-asr package not found. Please run: uv sync"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen3-ASR model: {e}")


def _load_aligner_model():
    """Load Qwen3-ForcedAligner model ONLY for alignment"""
    aligner_model_path = os.path.join(ABS_MODEL_DIR, "Qwen3-ForcedAligner-0.6B")

    # Auto-download if missing
    _download_model_if_needed(aligner_model_path, "Qwen/Qwen3-ForcedAligner-0.6B")

    rprint(f"[cyan]📥 Loading Aligner model...[/cyan]")

    try:
        from qwen_asr import Qwen3ForcedAligner

        # Detect device
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        # Initialize Aligner model ONLY
        aligner_model = Qwen3ForcedAligner.from_pretrained(
            aligner_model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )

        rprint(f"[green]✅ Aligner model loaded[/green]")
        return aligner_model

    except ImportError:
        raise ImportError(
            "qwen-asr package not found. Please run: uv sync"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Aligner model: {e}")


def _get_language():
    """Get language setting from config"""
    asr_language = load_key("asr.language")
    lang_map = {
        'zh': 'Chinese', 'en': 'English', 'ja': 'Japanese', 'ko': 'Korean',
        'yue': 'Cantonese', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'ru': 'Russian', 'ar': 'Arabic', 'pt': 'Portuguese',
        'th': 'Thai', 'vi': 'Vietnamese', 'id': 'Indonesian', 'tr': 'Turkish',
        'hi': 'Hindi', 'ms': 'Malay', 'nl': 'Dutch', 'sv': 'Swedish',
        'da': 'Danish', 'fi': 'Finnish', 'pl': 'Polish', 'cs': 'Czech',
        'fil': 'Filipino', 'fa': 'Persian', 'el': 'Greek', 'hu': 'Hungarian',
        'mk': 'Macedonian', 'ro': 'Romanian',
    }
    return lang_map.get(asr_language, None)


# ------------
# Stage 1: Transcription (text only)
# ------------

@except_handler("Qwen3-ASR transcription error:")
def transcribe_text_only(model, vocal_audio_file, start, end, language=None):
    """
    Transcribe a single audio segment, return text only (no timestamps)

    Args:
        model: Pre-loaded Qwen3-ASR model
        vocal_audio_file: Audio file path
        start: Start time offset (seconds)
        end: End time offset (seconds)
        language: Language for ASR (optional)

    Returns:
        str: Transcribed text
    """
    # Load audio segment
    audio_length = end - start
    audio, sr = librosa.load(vocal_audio_file, sr=16000, offset=start, duration=audio_length, mono=True)

    # Load language setting if not provided
    if language is None:
        language = _get_language()

    # Transcribe (NO timestamps)
    results = model.transcribe(
        audio=(audio, sr),
        language=language,
        return_time_stamps=False,  # Text only
    )

    if not results or len(results) == 0:
        raise Exception("Qwen3-ASR returned empty result")

    text = results[0].text

    # Clean up audio data to free GPU memory
    del audio, results

    return text


@except_handler("Qwen3-ASR batch transcription error:")
def transcribe_batch_for_text(vocal_audio_file, segments, progress=None):
    """
    Stage 1: Batch transcription (text only), then unload model

    Args:
        vocal_audio_file: Audio file path
        segments: List of (start, end) tuples in seconds
        progress: Rich Progress object (optional)

    Returns:
        List[str]: Transcribed text for each segment
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = _load_qwen_asr_model()
    language = _get_language()

    try:
        texts = []
        total = len(segments)

        for i, (start, end) in enumerate(segments, 1):
            text = transcribe_text_only(model, vocal_audio_file, start, end, language)
            texts.append(text)

            # Update progress (0% - 45% for transcription stage)
            if progress:
                percent = int((i / total) * 45)
                task_id = list(progress.tasks.keys())[0]  # Get task_id from tasks dict
                progress.update(task_id, completed=percent, description=f"[cyan]正在转录音频... ({i}/{total})")

            # Clean up after each segment to prevent memory leak
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return texts
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rprint("[dim]♻️ ASR model unloaded[/dim]")


# ------------
# Stage 2: Alignment (timestamps)
# ------------

@except_handler("Qwen3-Aligner alignment error:")
def align_with_text(model, vocal_audio_file, start, end, text, language=None):
    """
    Align audio segment with text to get timestamps

    Args:
        model: Pre-loaded Qwen3ForcedAligner model
        vocal_audio_file: Audio file path
        start: Start time offset (seconds)
        end: End time offset (seconds)
        text: Transcribed text
        language: Language for alignment (optional)

    Returns:
        dict: Standard ASR format with segments
    """
    # Load audio segment
    audio_length = end - start
    audio, sr = librosa.load(vocal_audio_file, sr=16000, offset=start, duration=audio_length, mono=True)

    # Load language setting if not provided
    if language is None:
        language = _get_language()

    # Clean text: remove empty lines, merge to single line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = " ".join(lines)

    # Align
    aligner_results = model.align(
        audio=(audio, sr),
        text=clean_text,
        language=language,
    )

    if not aligner_results or len(aligner_results) == 0:
        raise Exception("Qwen3-Aligner returned empty result")

    # Convert to standard format
    result = _convert_aligner_to_standard_asr_format(aligner_results[0], clean_text, start_offset=start)

    # Clean up audio data to free GPU memory
    del audio, aligner_results

    return result


@except_handler("Qwen3-Aligner batch alignment error:")
def align_batch_with_text(vocal_audio_file, segments, texts, progress=None):
    """
    Stage 2: Batch alignment, then unload model

    Args:
        vocal_audio_file: Audio file path
        segments: List of (start, end) tuples in seconds
        texts: List of transcribed texts (must match segments length)
        progress: Rich Progress object (optional)

    Returns:
        List[dict]: Standard ASR format for each segment
    """
    if len(segments) != len(texts):
        raise ValueError(f"Segments count ({len(segments)}) != texts count ({len(texts)})")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = _load_aligner_model()
    language = _get_language()

    try:
        results = []
        total = len(segments)

        for i, ((start, end), text) in enumerate(zip(segments, texts), 1):
            result = align_with_text(model, vocal_audio_file, start, end, text, language)
            results.append(result)

            # Update progress (45% - 95% for alignment stage)
            if progress:
                percent = 45 + int((i / total) * 50)
                task_id = list(progress.tasks.keys())[0]  # Get task_id from tasks dict
                progress.update(task_id, completed=percent, description=f"[cyan]正在对齐音频... ({i}/{total})")

            # Clean up after each segment to prevent memory leak
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rprint("[dim]♻️ Aligner model unloaded[/dim]")


# ------------
# Main: Combined two-stage pipeline
# ------------

@except_handler("Qwen3-ASR batch processing error:")
def transcribe_batch(vocal_audio_file, segments):
    """
    Batch transcription with two-stage separated architecture

    Stage 1: ASR transcription (text only)  →  0% - 45%
    Stage 2: Forced alignment (timestamps)  → 45% - 95%
    Cleanup:                              → 95% - 100%

    Args:
        vocal_audio_file: Audio file path
        segments: List of (start, end) tuples in seconds

    Returns:
        List[dict]: Standard ASR format for each segment
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]准备中...", total=100)

        # Stage 1: Transcription (text only)
        rprint(f"[blue]📝 Stage 1: ASR 转录 ({len(segments)} segments)[/blue]")
        texts = transcribe_batch_for_text(vocal_audio_file, segments, progress)
        progress.update(task, completed=45, description="[cyan]转录完成，正在加载对齐模型...")

        # Stage 2: Alignment (timestamps)
        rprint(f"[blue]🔗 Stage 2: 强制对齐 ({len(segments)} segments)[/blue]")
        results = align_batch_with_text(vocal_audio_file, segments, texts, progress)

        progress.update(task, completed=100, description="[green]✅ 完成")

    return results


# ------------
# Legacy: Single segment transcription (for compatibility)
# ------------

@except_handler("Qwen3-ASR processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    """
    Qwen3-ASR transcription main function (single segment, two-stage)

    Args:
        raw_audio_file: Original audio file path
        vocal_audio_file: Vocal-separated audio (or original if not separated)
        start: Start time offset (seconds)
        end: End time offset (seconds)

    Returns:
        dict: Standard ASR format with segments
    """
    # Single segment = batch with one item
    results = transcribe_batch(vocal_audio_file, [(start, end)])
    return results[0]


if __name__ == "__main__":
    import json
    result = transcribe_audio("output/audio/raw.wav", "output/audio/raw.wav", 0, 30)
    print(json.dumps(result, indent=2, ensure_ascii=False))
