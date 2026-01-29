# ------------
# Qwen3-ASR Backend
# Local ASR with 52 language support + 22 Chinese dialects
# ------------

import os
import sys
import logging
import warnings
import gc
import io
from typing import Optional

# ------------
# Suppress warnings (MUST be before library imports)
# ------------

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

for logger_name in ['transformers', 'qwen_asr']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    logger.propagate = False

# Now safe to import libraries
import torch
import librosa
import soundfile as sf
from rich import print as rprint
from core.utils import load_key, except_handler

# Model cache directory
MODEL_DIR = load_key("model_dir")
ABS_MODEL_DIR = os.path.abspath(MODEL_DIR)


def _convert_qwen_to_standard_asr_format(result, start_offset=0):
    """
    Convert Qwen3-ASR output to standard ASR format

    Args:
        result: Qwen3-ASR result object with .language, .text, .time_stamps
        start_offset: Time offset for this audio segment

    Returns:
        dict: Standard ASR format with segments
    """
    if not result or not result.time_stamps:
        return {'segments': [], 'language': 'unknown'}

    # Extract language
    language = result.language if hasattr(result, 'language') else 'unknown'

    # Build word list from timestamps
    words = []
    for ts in result.time_stamps:
        words.append({
            'word': ts.text,
            'start': round(ts.start_time + start_offset, 2),
            'end': round(ts.end_time + start_offset, 2)
        })

    # Combine all words into a single segment
    segment_text = ' '.join([w['word'] for w in words])

    segment = {
        'start': round(words[0]['start'], 2) if words else round(start_offset, 2),
        'end': round(words[-1]['end'], 2) if words else round(start_offset, 2),
        'text': segment_text,
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
            max_inference_batch_size=16,
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
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    """
    Qwen3-ASR transcription main function

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

    # Load model
    model = _load_qwen_model()

    # Load audio segment
    audio_length = end - start
    rprint(f"[cyan]🎵 Loading audio segment ({audio_length:.1f}s)...[/cyan]")

    audio, sr = librosa.load(vocal_audio_file, sr=16000, offset=start, duration=audio_length, mono=True)

    # Save to temp buffer
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, sr, format='WAV', subtype='PCM_16')
    audio_buffer.seek(0)

    try:
        rprint(f"[bold green]Transcribing with Qwen3-ASR...[/bold green]")

        # Get language setting
        asr_language = load_key("asr.language")

        # Map ISO code to natural language name for Qwen3-ASR
        lang_map = {
            'zh': 'Chinese',
            'en': 'English',
            'ja': 'Japanese',
            'ko': 'Korean',
            'yue': 'Cantonese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'ru': 'Russian',
            'ar': 'Arabic',
            'pt': 'Portuguese',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'tr': 'Turkish',
            'hi': 'Hindi',
            'ms': 'Malay',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'cs': 'Czech',
            'fil': 'Filipino',
            'fa': 'Persian',
            'el': 'Greek',
            'hu': 'Hungarian',
            'mk': 'Macedonian',
            'ro': 'Romanian',
        }

        language = lang_map.get(asr_language, None)  # None for auto-detect

        # Transcribe
        results = model.transcribe(
            audio=audio_buffer.getvalue(),
            language=language,
            return_time_stamps=True,
        )

        if not results or len(results) == 0:
            raise Exception("Qwen3-ASR returned empty result")

        # Convert to standard format
        result = _convert_qwen_to_standard_asr_format(results[0], start_offset=start)

        rprint(f"[green]✅ Transcription completed: {result['language']}[/green]")
        return result

    finally:
        # Clean up
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rprint("[dim]♻️ Resources released[/dim]")


if __name__ == "__main__":
    import json
    result = transcribe_audio("output/audio/raw.wav", "output/audio/raw.wav", 0, 30)
    print(json.dumps(result, indent=2, ensure_ascii=False))
