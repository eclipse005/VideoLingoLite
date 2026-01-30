# ------------
# Qwen3-ASR Backend
# Local ASR with 52 language support + 22 Chinese dialects
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


def _convert_qwen_to_standard_asr_format(result, start_offset=0, align_punctuation=True):
    """
    Convert Qwen3-ASR output to standard ASR format

    Args:
        result: Qwen3-ASR result object with .language, .text, .time_stamps
        start_offset: Time offset for this audio segment
        align_punctuation: Whether to attach punctuation to word timestamps

    Returns:
        dict: Standard ASR format with segments
    """
    if not result or not result.time_stamps:
        return {'segments': [], 'language': 'unknown'}

    # Extract language
    language = result.language if hasattr(result, 'language') else 'unknown'

    # Extract full text
    full_text = result.text if hasattr(result, 'text') else ''

    if align_punctuation:
        # Apply character alignment (attach punctuation to words)
        word_timestamps = [(ts.text, ts.start_time, ts.end_time) for ts in result.time_stamps]
        aligned_timestamps = _align_text_to_timestamps(full_text, word_timestamps)

        # Build word list from aligned timestamps
        words = []
        for text, start, end in aligned_timestamps:
            words.append({
                'word': text,
                'start': round(start + start_offset, 2),
                'end': round(end + start_offset, 2)
            })
    else:
        # Build word list from original timestamps (no punctuation alignment)
        words = []
        for ts in result.time_stamps:
            words.append({
                'word': ts.text,
                'start': round(ts.start_time + start_offset, 2),
                'end': round(ts.end_time + start_offset, 2)
            })

    segment = {
        'start': round(words[0]['start'], 2) if words else round(start_offset, 2),
        'end': round(words[-1]['end'], 2) if words else round(start_offset, 2),
        'text': full_text,
        'words': words
    }

    return {'segments': [segment], 'language': language}


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


def _load_qwen_model():
    """Load Qwen3-ASR model from local cache (auto-download if missing)"""
    model_name = load_key("asr.model", default="Qwen3-ASR-1.7B")
    asr_model_path = os.path.join(ABS_MODEL_DIR, model_name)
    aligner_model_path = os.path.join(ABS_MODEL_DIR, "Qwen3-ForcedAligner-0.6B")

    # Auto-download if missing
    _download_model_if_needed(asr_model_path, f"Qwen/{model_name}")
    _download_model_if_needed(aligner_model_path, "Qwen/Qwen3-ForcedAligner-0.6B")

    rprint(f"[cyan]📥 Loading {model_name} from local cache...[/cyan]")

    try:
        from qwen_asr import Qwen3ASRModel

        # Detect device
        if torch.cuda.is_available():
            device = "cuda:0"
            rprint(f"[bold green]🚀 GPU Detected | Using CUDA acceleration[/bold green]")
        else:
            device = "cpu"
            rprint(f"[bold yellow]⚠️ No GPU found | Using CPU (will be slow)[/bold yellow]")

        # Initialize model
        asr_model = Qwen3ASRModel.from_pretrained(
            asr_model_path,
            dtype=torch.bfloat16,
            device_map=device,
            forced_aligner=aligner_model_path,
            forced_aligner_kwargs=dict(
                dtype=torch.bfloat16,
                device_map=device,
            ),
            max_new_tokens=4096,  # Support long audio
            max_inference_batch_size=4,
        )

        rprint(f"[green]✅ Qwen3-ASR-{model_name} loaded successfully[/green]")
        return asr_model

    except ImportError:
        raise ImportError(
            "qwen-asr package not found. Please run: uv sync"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen3-ASR model: {e}")


@except_handler("Qwen3-ASR processing error:")
def transcribe_with_model(model, vocal_audio_file, start, end, language=None, return_raw=False):
    """
    Transcribe a single audio segment using pre-loaded model

    Args:
        model: Pre-loaded Qwen3-ASR model
        vocal_audio_file: Audio file path
        start: Start time offset (seconds)
        end: End time offset (seconds)
        language: Language for ASR (optional, will load from config if None)
        return_raw: If True, return both aligned and raw results

    Returns:
        dict: Standard ASR format with segments (with punctuation alignment)
        or tuple: (aligned_result, raw_result) if return_raw=True
    """
    # Load audio segment
    audio_length = end - start
    audio, sr = librosa.load(vocal_audio_file, sr=16000, offset=start, duration=audio_length, mono=True)

    # Load language setting if not provided
    if language is None:
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
        language = lang_map.get(asr_language, None)

    # Transcribe
    results = model.transcribe(
        audio=(audio, sr),
        language=language,
        return_time_stamps=True,
    )

    if not results or len(results) == 0:
        raise Exception("Qwen3-ASR returned empty result")

    # Convert to standard format (with punctuation alignment for processing)
    aligned_result = _convert_qwen_to_standard_asr_format(results[0], start_offset=start, align_punctuation=True)

    # Also generate raw version (without punctuation alignment) for debugging
    raw_result = _convert_qwen_to_standard_asr_format(results[0], start_offset=start, align_punctuation=False)

    # Clean up audio data to free GPU memory
    del audio, results

    if return_raw:
        return aligned_result, raw_result
    return aligned_result


@except_handler("Qwen3-ASR processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    """
    Qwen3-ASR transcription main function (single segment, loads model each time)

    Args:
        raw_audio_file: Original audio file path
        vocal_audio_file: Vocal-separated audio (or original if not separated)
        start: Start time offset (seconds)
        end: End time offset (seconds)

    Returns:
        dict: Standard ASR format with segments
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = _load_qwen_model()

    try:
        return transcribe_with_model(model, vocal_audio_file, start, end)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rprint("[dim]♻️ Resources released[/dim]")


@except_handler("Qwen3-ASR batch processing error:")
def transcribe_batch(vocal_audio_file, segments):
    """
    Batch transcription with single model load (efficient for multi-segment audio)

    Args:
        vocal_audio_file: Audio file path
        segments: List of (start, end) tuples in seconds

    Returns:
        List[tuple]: List of (aligned_result, raw_result) for each segment
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = _load_qwen_model()

    # Load language once
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
    language = lang_map.get(asr_language, None)

    try:
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]正在转录音频...", total=len(segments))
            for i, (start, end) in enumerate(segments, 1):
                aligned, raw = transcribe_with_model(model, vocal_audio_file, start, end, language, return_raw=True)
                results.append((aligned, raw))
                progress.update(task, advance=1)

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
        rprint("[dim]♻️ Resources released[/dim]")


if __name__ == "__main__":
    import json
    result = transcribe_audio("output/audio/raw.wav", "output/audio/raw.wav", 0, 30)
    print(json.dumps(result, indent=2, ensure_ascii=False))
