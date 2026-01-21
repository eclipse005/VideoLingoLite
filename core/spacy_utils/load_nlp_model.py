import os
import spacy
from spacy.cli import download
from core.utils import rprint, load_key, except_handler

SPACY_MODEL_MAP = load_key("spacy_model_map")

def get_spacy_model(language: str):
    model = SPACY_MODEL_MAP.get(language.lower(), "en_core_web_md")
    if language not in SPACY_MODEL_MAP:
        rprint(f"[yellow]Spacy 模型不支持 '{language}'，使用 en_core_web_md 作为备用模型...[/yellow]")
    return model

@except_handler("NLP Spacy 模型加载失败")
def init_nlp():
    # Use asr.language from VideoLingoLite
    language = load_key("asr.language")
    model = get_spacy_model(language)

    # Set model download path to _model_cache
    model_dir = load_key("model_dir")
    os.environ["NLTK_DATA"] = model_dir

    rprint(f"[blue]⏳ 正在加载 NLP Spacy 模型: <{model}> ...[/blue]")
    try:
        nlp = spacy.load(model)
    except:
        rprint(f"[yellow]正在下载 {model} 模型...[/yellow]")
        download(model)
        nlp = spacy.load(model)
    rprint("[green]✅ NLP Spacy 模型加载成功！[/green]")
    return nlp

# --------------------
# define the intermediate files
# --------------------
SPLIT_BY_MARK_FILE = "output/log/split_by_mark.txt"
SPLIT_BY_COMMA_FILE = "output/log/split_by_comma.txt"
SPLIT_BY_CONNECTOR_FILE = "output/log/split_by_connector.txt"
SPLIT_BY_NLP_FILE = "output/log/split_by_nlp.txt"
SPLIT_BY_PAUSE_FILE = "output/log/split_by_pause.txt"
