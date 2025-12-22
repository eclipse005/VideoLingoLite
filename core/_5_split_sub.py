import pandas as pd
from typing import List, Tuple
import concurrent.futures
import math

from core._3_llm_sentence_split import split_sentence
from core.prompts import get_align_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    
    def valid_align(response_data):
        # 1. 检查response是字典
        if not isinstance(response_data, dict):
            return {"status": "error", "message": "Response must be a dictionary"}

        # 2. 检查analysis字段存在
        if 'analysis' not in response_data:
            return {"status": "error", "message": "Missing required field: 'analysis'"}

        # 3. 检查align字段存在
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required field: 'align'"}

        # 4. 检查align是列表
        if not isinstance(response_data['align'], list):
            return {"status": "error", "message": "Field 'align' must be a list"}

        # 5. 检查align长度
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Field 'align' must contain at least 2 items"}

        # 6. 检查align数组内每个元素的字段结构
        for i, item in enumerate(response_data['align']):
            # 每个元素必须是字典
            if not isinstance(item, dict):
                return {"status": "error", "message": f"align[{i}] must be a dictionary"}

            # 必须包含src_part_{i+1}和target_part_{i+1}字段
            if f'src_part_{i+1}' not in item:
                return {"status": "error", "message": f"Missing required field: 'src_part_{i+1}' in align[{i}]"}

            if f'target_part_{i+1}' not in item:
                return {"status": "error", "message": f"Missing required field: 'target_part_{i+1}' in align[{i}]"}

        return {"status": "success", "message": "Align validation completed"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']
    src_parts = src_part.split('[br]')
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]

    
    asr_language = load_key("asr.language")
    joiner = get_joiner(asr_language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    @except_handler("Error in split_align_subs")
    def process(i):
        # Calculate the number of parts dynamically based on sentence length
        # Support both English (word-based) and Chinese (character-based)
        asr_language = load_key("asr.language")
        is_cjk = asr_language.lower() in ['zh', 'chinese', 'ja', 'japanese', 'ko', 'korean']

        if is_cjk:
            # For CJK languages: use character count with weight
            text_length = calc_len(src_lines[i])
        else:
            # For English and other space-delimited languages: use word count
            text_length = len(src_lines[i].split())

        max_split_length = load_key("max_split_length")
        num_parts = max(2, math.ceil(text_length / max_split_length))

        split_src = split_sentence(src_lines[i], num_parts=num_parts).strip()

        # 只有当 LLM 认为需要拆分时才调用 align_subs
        if '[br]' in split_src:
            src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
        else:
            # LLM 认为不需要拆分，直接使用原始句子
            src_parts = [src_lines[i]]
            tr_parts = [tr_lines[i]]
            tr_remerged = tr_lines[i]
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        executor.map(process, to_split)
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    return src_lines, tr_lines, remerged_tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans)
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))
    
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()
