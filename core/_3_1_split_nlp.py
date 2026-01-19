"""
NLP-based Sentence Segmentation Module (Stage 1)

This module uses spaCy to perform rule-based sentence splitting:
1. Split by punctuation marks (spaCy sentence boundaries)
2. Split by commas (with linguistic analysis)
3. Split by connectors (that, which, because, but, and, etc.)
4. Split long sentences by root (dynamic programming)

Output: split_by_nlp.txt (Stage 1 result)
"""

from core.spacy_utils import *
from core.utils.models import _3_1_SPLIT_BY_NLP
from core.utils import check_file_exists, rprint

@check_file_exists(_3_1_SPLIT_BY_NLP)
def split_by_spacy():
    rprint("[blue]🔍 Starting NLP-based sentence segmentation (Stage 1)[/blue]")

    nlp = init_nlp()

    split_by_mark(nlp)
    split_by_comma_main(nlp)
    split_sentences_main(nlp)
    split_long_by_root_main(nlp)
    split_by_pause()
    rprint(f"[green]✅ NLP sentence segmentation completed: {_3_1_SPLIT_BY_NLP}[/green]")
    return

if __name__ == '__main__':
    split_by_spacy()
