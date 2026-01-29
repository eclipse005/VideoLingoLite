# ------------
# Download Qwen3-ASR Models
# Uses modelscope to download models to _model_cache
# ------------

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.config_utils import load_key
from rich import print as rprint

MODEL_DIR = load_key("model_dir", default="./_model_cache")
ABS_MODEL_DIR = os.path.abspath(MODEL_DIR)


def download_models():
    """Download Qwen3-ASR models using modelscope"""

    models = [
        ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR-1.7B"),
        ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR-0.6B"),
        ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner-0.6B"),
    ]

    rprint(f"[cyan]📥 Model cache directory: {ABS_MODEL_DIR}[/cyan]")
    os.makedirs(ABS_MODEL_DIR, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ImportError:
        rprint("[red]❌ modelscope not installed. Run: uv sync[/red]")
        sys.exit(1)

    for repo_id, local_dir_name in models:
        local_dir = os.path.join(ABS_MODEL_DIR, local_dir_name)

        # Check if already downloaded
        if os.path.exists(local_dir) and os.listdir(local_dir):
            rprint(f"[yellow]⏭️ Skipping {local_dir_name} (already exists)[/yellow]")
            continue

        rprint(f"[cyan]📥 Downloading {repo_id}...[/cyan]")

        try:
            snapshot_download(
                repo_id,
                local_dir=local_dir,
                revision="master",
            )
            rprint(f"[green]✅ Downloaded {local_dir_name}[/green]")
        except Exception as e:
            rprint(f"[red]❌ Failed to download {repo_id}: {e}[/red]")

    rprint("[green]✨ All downloads completed![/green]")


if __name__ == "__main__":
    download_models()
