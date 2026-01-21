import os
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

INPUT_FOLDER = os.path.join('batch', 'input')
CSV_FILE = 'batch/tasks_setting.csv'

# Video file extensions to scan
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mp3', '.wav', '.m4a', '.flac', '.aac'}


def scan_input_folder():
    """Scan input folder for video/audio files, sorted by filename"""
    if not os.path.exists(INPUT_FOLDER):
        console.print(Panel(f"[bold red]Input folder not found: {INPUT_FOLDER}[/]", title="Error", expand=False))
        return []

    files = []
    for filename in os.listdir(INPUT_FOLDER):
        ext = os.path.splitext(filename)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            files.append(filename)

    return sorted(files, key=str.lower)


def get_user_input(default_source_lang='en', default_target_lang='简体中文', default_dubbing='0'):
    """Get user input with default values"""
    console.print(Panel(
        f"[bold cyan]Default values:[/]\n"
        f"Source Language: [yellow]{default_source_lang}[/]\n"
        f"Target Language: [yellow]{default_target_lang}[/]\n"
        f"Dubbing: [yellow]{default_dubbing}[/]\n\n"
        f"[dim]Press Enter to use default value[/]",
        title="Configuration",
        expand=False
    ))

    source_lang = console.input(f"\n[bold cyan]Source Language[/] [dim][default: {default_source_lang}][/]: ").strip()
    if not source_lang:
        source_lang = default_source_lang

    target_lang = console.input(f"[bold cyan]Target Language[/] [dim][default: {default_target_lang}][/]: ").strip()
    if not target_lang:
        target_lang = default_target_lang

    dubbing = console.input(f"[bold cyan]Dubbing[/] [dim][default: {default_dubbing}][/]: ").strip()
    if not dubbing:
        dubbing = default_dubbing

    return source_lang, target_lang, dubbing


def create_csv(files, source_lang, target_lang, dubbing):
    """Create tasks_setting.csv with scanned files"""
    data = {
        'Video File': files,
        'Source Language': [source_lang] * len(files),
        'Target Language': [target_lang] * len(files),
        'Dubbing': [dubbing] * len(files),
        'Status': [''] * len(files)
    }

    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    return df


def preview_table(df):
    """Display preview table before saving"""
    table = Table(title="[bold green]Tasks Preview[/]", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Video File", style="yellow")
    table.add_column("Source Language", style="green")
    table.add_column("Target Language", style="green")
    table.add_column("Dubbing", style="green")

    for idx, row in df.iterrows():
        table.add_row(
            str(idx + 1),
            str(row['Video File']),
            str(row['Source Language']),
            str(row['Target Language']),
            str(row['Dubbing'])
        )

    console.print(table)


def main():
    console.print(Panel("[bold cyan]VideoLingoLite - Tasks Generator[/]", title="Welcome", expand=False))

    # Scan input folder
    files = scan_input_folder()

    if not files:
        console.print(Panel("[bold yellow]No video/audio files found in input folder[/]", title="Warning", expand=False))
        return

    console.print(Panel(f"[bold green]Found {len(files)} file(s)[/]", title="Scan Result", expand=False))

    # Get user input
    source_lang, target_lang, dubbing = get_user_input()

    # Create DataFrame
    df = create_csv(files, source_lang, target_lang, dubbing)

    # Preview
    preview_table(df)

    # Save
    df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    console.print(Panel(f"[bold green]Successfully generated {CSV_FILE}[/]\n[bold]{len(files)}[/] task(s) added", title="Done", expand=False))


if __name__ == "__main__":
    main()
