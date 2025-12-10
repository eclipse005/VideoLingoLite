from core.utils import *
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core._1_ytdlp import find_video_files
from core.utils.models import *

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. video to audio
    video_file = find_video_files()
    convert_video_to_audio(video_file)

    # 2. Use original audio directly (no vocal separation)
    vocal_audio = _RAW_AUDIO_FILE

    # 3. Extract audio
    segments = split_audio(_RAW_AUDIO_FILE)

    # 4. Select ASR backend based on config
    asr_runtime = load_key("asr.runtime")
    if asr_runtime == "whisperX":
        from core.asr_backend.whisperX_local import transcribe_audio as ts
        rprint("[cyan]🎤 Transcribing audio with WhisperX...[/cyan]")
    elif asr_runtime == "gemini":
        from core.asr_backend.gemini import transcribe_audio_gemini as ts
        rprint("[cyan]🎤 Transcribing audio with Gemini API...[/cyan]")
    else:
        raise ValueError(f"Unsupported ASR runtime: {asr_runtime}")

    # 5. Transcribe audio by clips
    all_results = []
    for start, end in segments:
        result = ts(_RAW_AUDIO_FILE, vocal_audio, start, end)
        all_results.append(result)

    # 6. Combine results
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])

    # 7. Process df
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()