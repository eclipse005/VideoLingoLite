import os
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from core.utils import safe_read_csv

# Constants
SETTINGS_FILE = 'batch/tasks_setting.csv'
INPUT_FOLDER = os.path.join('batch', 'input')

console = Console()

def check_settings():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    df = safe_read_csv(SETTINGS_FILE)
    input_files = set(os.listdir(INPUT_FOLDER))
    csv_files = set(df['Video File'].tolist())
    files_not_in_csv = input_files - csv_files

    all_passed = True
    local_video_tasks = 0

    if files_not_in_csv:
        console.print(Panel(
            "\n".join([f"- {file}" for file in files_not_in_csv]),
            title="[bold red]Warning: Files in input folder not mentioned in CSV file",
            expand=False
        ))
        all_passed = False

    for index, row in df.iterrows():
        video_file = row['Video File']
        source_language = row['Source Language']

        if os.path.isfile(os.path.join(INPUT_FOLDER, video_file)):
            local_video_tasks += 1
        else:
            console.print(Panel(f"Invalid video file 「{video_file}」", title=f"[bold red]Error in row {index + 2}", expand=False))
            all_passed = False

    if all_passed:
        console.print(Panel(f"✅ All settings passed the check!\nDetected {local_video_tasks} local video tasks.", title="[bold green]Success", expand=False))

    return all_passed


if __name__ == "__main__":  
    check_settings()