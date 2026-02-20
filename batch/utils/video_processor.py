import os
from core.utils.onekeycleanup import cleanup
from core.utils import load_key
import shutil
from functools import partial
from rich.panel import Panel
from rich.console import Console
from core import *

console = Console()

INPUT_DIR = 'batch/input'
OUTPUT_DIR = 'output'
SAVE_DIR = 'batch/output'
ERROR_OUTPUT_DIR = 'batch/output/ERROR'
YTB_RESOLUTION_KEY = "ytb_resolution"

def process_video(file, is_retry=False):
    if not is_retry:
        prepare_output_folder(OUTPUT_DIR)

    text_steps = [
        ("🎥 Processing input file", partial(process_input_file, file)),
        ("🎙️ Transcribing with ASR", partial(_2_asr.transcribe)),
        ("✂️ Splitting sentences", split_sentences),
        ("📝 Summarizing and translating", summarize_and_translate),
        ("⚡ Processing and aligning subtitles", process_and_align_subtitles),
    ]
    
    current_step = ""
    for step_name, step_func in text_steps:
        current_step = step_name
        for attempt in range(3):
            try:
                console.print(Panel(
                    f"[bold green]{step_name}[/]",
                    subtitle=f"Attempt {attempt + 1}/3" if attempt > 0 else None,
                    border_style="blue"
                ))
                result = step_func()
                if result is not None:
                    globals().update(result)
                break
            except Exception as e:
                if attempt == 2:
                    error_panel = Panel(
                        f"[bold red]Error in step '{current_step}':[/]\n{str(e)}",
                        border_style="red"
                    )
                    console.print(error_panel)
                    cleanup(ERROR_OUTPUT_DIR)
                    return False, current_step, str(e)
                console.print(Panel(
                    f"[yellow]Attempt {attempt + 1} failed. Retrying...[/]",
                    border_style="yellow"
                ))
    
    console.print(Panel("[bold green]All steps completed successfully! 🎉[/]", border_style="green"))
    cleanup(SAVE_DIR)
    return True, "", ""

def prepare_output_folder(output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

def process_input_file(file):
    if file.startswith('http'):
        # YouTube 下载 - 使用最佳质量
        import uuid
        file_id = f"file_{uuid.uuid4().hex[:8]}"
        _1_ytdlp.download_video_ytdlp(file, output_dir=OUTPUT_DIR, file_id=file_id)
        video_file = _1_ytdlp.find_video_files(OUTPUT_DIR)
    else:
        input_file = os.path.join('batch', 'input', file)
        output_file = os.path.join(OUTPUT_DIR, file)
        shutil.copy(input_file, output_file)
        video_file = output_file
    return {'video_file': video_file}

def split_sentences():
    sentences = _3_1_split_by_punctuation.split_by_punctuation()
    sentences = _3_2_hotword.correct_terms_in_sentences(sentences)
    sentences = _3_3_split_meaning.split_sentences_by_meaning(sentences)
    return {'sentences': sentences}

def summarize_and_translate():
    sentences = globals().get('sentences')
    _4_1_summarize.get_summary(sentences)
    sentences = _4_2_translate.translate_all(sentences)
    return {'sentences': sentences}

def process_and_align_subtitles():
    sentences = globals().get('sentences')
    sentences = _5_split_sub.split_for_sub_main(sentences)
    _6_gen_sub.align_timestamp_main(sentences)
    return {'sentences': sentences}
