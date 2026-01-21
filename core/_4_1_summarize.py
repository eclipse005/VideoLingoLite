import json
import re
import unicodedata
import time
from core.prompts import get_summary_prompt
import pandas as pd
from core.utils import load_key, rprint, safe_read_csv, ask_gpt, format_duration
from core.utils.models import _3_2_SPLIT_BY_MEANING, _4_1_TERMINOLOGY
from core.utils.sentence_tools import clean_word

CUSTOM_TERMS_PATH = 'custom_terms.csv'

def combine_chunks():
    """Combine the text chunks identified by ASR into a single long text"""
    with open(_3_2_SPLIT_BY_MEANING, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    cleaned_sentences = [line.strip() for line in sentences]
    combined_text = ' '.join(cleaned_sentences)
    return combined_text[:load_key('summary_length')]  #! Return only the first x characters

def clean_text_for_comparison(text):
    """Clean text by removing spaces and punctuation for pure text comparison

    Uses unicodedata normalization, consistent with clean_word().
    """
    # Use clean_word() for standardization (already uses unicodedata)
    # clean_word() removes all punctuation and symbols, keeping only letters and numbers
    return clean_word(text)

def expand_keywords(src):
    """Expand term keywords, supporting both / and () splitting

    Examples:
        'FLIP/Fluid' -> ['FLIP', 'Fluid']
        'FLIP (Fluid)' -> ['FLIP', 'Fluid']
        'Fluid (FLIP)' -> ['Fluid', 'FLIP']
        'FLIP' -> ['FLIP']
    """
    keywords = []
    parts = src.split('/')

    for part in parts:
        part = part.strip()
        match = re.match(r'^(.+?)\s*\(([^)]+)\)$', part)
        if match:
            outer = match.group(1).strip()
            inner = match.group(2).strip()
            if outer:
                keywords.append(outer)
            if inner:
                keywords.append(inner)
        else:
            if part:
                keywords.append(part)

    return keywords

def search_things_to_note_in_prompt(sentence):
    """Search for terms to note in the given sentence using cleaned text comparison"""
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        things_to_note = json.load(file)
    
    # Clean the sentence for comparison
    cleaned_sentence = clean_text_for_comparison(sentence)
    
    things_to_note_list = []
    for term in things_to_note['terms']:
        # Expand keywords from term src (supports / and () splitting)
        keywords = expand_keywords(term['src'])
        for kw in keywords:
            cleaned_kw = clean_text_for_comparison(kw.strip())
            if cleaned_kw and cleaned_kw in cleaned_sentence:
                things_to_note_list.append(term['src'])
                break
    
    if things_to_note_list:
        prompt = '\n'.join(
            f'{i+1}. "{term["src"]}": "{term["tgt"]}",'
            f' meaning: {term["note"]}'
            for i, term in enumerate(things_to_note['terms'])
            if term['src'] in things_to_note_list
        )
        return prompt
    else:
        return None

def get_summary():
    rprint("📝 正在总结和提取术语...")

    start_time = time.time()

    src_content = combine_chunks()
    custom_terms = safe_read_csv(CUSTOM_TERMS_PATH)
    custom_terms_json = {
        "terms":
            [
                {
                    "src": str(row.iloc[0]),
                    "tgt": str(row.iloc[1]),
                    "note": str(row.iloc[2])
                }
                for _, row in custom_terms.iterrows()
            ]
    }
    if len(custom_terms) > 0:
        rprint(f"📖 已加载自定义术语：{len(custom_terms)} 条")
        rprint("📝 术语内容：", json.dumps(custom_terms_json, indent=2, ensure_ascii=False))
    summary_prompt = get_summary_prompt(src_content, custom_terms_json)

    def valid_summary(response_data):
        required_keys = {'src', 'tgt', 'note'}
        if 'terms' not in response_data:
            return {"status": "error", "message": "Invalid response format"}
        for term in response_data['terms']:
            if not all(key in term for key in required_keys):
                return {"status": "error", "message": "Invalid response format"}
        return {"status": "success", "message": "Summary completed"}

    summary = ask_gpt(summary_prompt, resp_type='json', valid_def=valid_summary, log_title='summary')
    summary['terms'].extend(custom_terms_json['terms'])

    with open(_4_1_TERMINOLOGY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    elapsed = time.time() - start_time
    rprint(f'💾 总结已保存到 → `{_4_1_TERMINOLOGY}`')
    rprint(f'[dim]⏱️ 总结和术语提取耗时: {format_duration(elapsed)}[/dim]')

if __name__ == '__main__':
    get_summary()