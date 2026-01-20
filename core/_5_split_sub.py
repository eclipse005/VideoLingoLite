import pandas as pd
from typing import List, Tuple
import concurrent.futures
import math

from core.prompts import get_align_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], List[str]]:
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
    src_parts = [part.strip() for part in src_part.split('[br]')]
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]

    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)

    # Return tr_parts directly instead of merging, so remerged can be flattened correctly
    return src_parts, tr_parts, tr_parts

def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    # Get source and target language ISO codes
    asr_language = load_key("asr.language")
    target_lang_desc = load_key("target_language")
    from core.utils.models import TARGET_LANG_MAP
    target_language = TARGET_LANG_MAP.get(target_lang_desc, 'en')

    # Get soft limits for source and target languages
    origin_soft_limit = get_language_length_limit(asr_language, 'origin')
    translate_soft_limit = get_language_length_limit(target_language, 'translate')

    remerged_tr_lines = tr_lines.copy()

    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        # Check if source or translation exceeds hard limit
        src_exceeds = check_length_exceeds(src, origin_soft_limit, asr_language)
        tr_exceeds = check_length_exceeds(tr, translate_soft_limit, target_language)
        if src_exceeds or tr_exceeds:
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
        text_length = get_effective_length(src_lines[i], asr_language)
        num_parts = max(2, math.ceil(text_length / origin_soft_limit))

        split_src = split_sentence(src_lines[i], num_parts=num_parts).strip()

        # 只有当 LLM 认为需要拆分时才调用 align_subs
        if '[br]' in split_src:
            src_parts, tr_parts, tr_remerged_parts = align_subs(src_lines[i], tr_lines[i], split_src)
        else:
            # LLM 认为不需要拆分，直接使用原始句子
            src_parts = [src_lines[i]]
            tr_parts = [tr_lines[i]]
            tr_remerged_parts = [tr_lines[i]]
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged_parts

    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        executor.map(process, to_split)

    # Flatten `src_lines`, `tr_lines`, and `remerged_tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    remerged_tr_lines = [item for sublist in remerged_tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]

    return src_lines, tr_lines, remerged_tr_lines

def split_for_sub_main(sentences=None):
    """
    字幕切分主函数，处理 Sentence 对象

    Args:
        sentences: Sentence 对象列表（如果为 None，从 CSV 加载）

    Returns:
        List[Sentence]: 切分后的 Sentence 对象列表
    """
    console.print("[bold green]🚀 Start splitting subtitles...[/]")

    # 📊 显示接收到的 Sentence 对象信息
    if sentences:
        console.print(f'[cyan]📊 Received {len(sentences)} Sentence objects from Stage 3[/cyan]')
        has_translation = sum(1 for s in sentences if s.translation)
        console.print(f'[dim]Sentences with translation: {has_translation}/{len(sentences)}[/dim]')
    else:
        console.print('[yellow]⚠️ No Sentence objects received, loading from CSV...[/yellow]')

    # 如果没有传入 Sentence 对象，从 CSV 加载（向后兼容）
    if sentences is None:
        from core._2_asr import load_chunks

        df = safe_read_csv(_4_2_TRANSLATION).fillna('')
        src = df['Source'].tolist()
        trans = df['Translation'].tolist()

        # 重建 Sentence 对象
        chunks = load_chunks()
        sentences = []
        char_pos = 0
        chunk_idx = 0

        for src_text, trans_text in zip(src, trans):
            sentence_chunks = []
            text_length = len(src_text)

            while chunk_idx < len(chunks) and char_pos < text_length:
                chunk = chunks[chunk_idx]
                sentence_chunks.append(chunk)
                char_pos += len(chunk.text)
                chunk_idx += 1

            sentence = Sentence(
                chunks=sentence_chunks,
                text=src_text,
                translation=trans_text,
                start=sentence_chunks[0].start if sentence_chunks else 0.0,
                end=sentence_chunks[-1].end if sentence_chunks else 0.0,
                index=len(sentences),
                is_split=False
            )
            sentences.append(sentence)
            char_pos = 0

    # Get source and target language ISO codes
    asr_language = load_key("asr.language")
    target_lang_desc = load_key("target_language")
    from core.utils.models import TARGET_LANG_MAP
    target_language = TARGET_LANG_MAP.get(target_lang_desc, 'en')

    # Get soft limits for source and target languages
    origin_soft_limit = get_language_length_limit(asr_language, 'origin')
    translate_soft_limit = get_language_length_limit(target_language, 'translate')

    # 多轮切割
    for attempt in range(3):
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))

        # 找出需要切分的句子
        to_split = []
        for i, sent in enumerate(sentences):
            src_exceeds = check_length_exceeds(sent.text, origin_soft_limit, asr_language)
            tr_exceeds = check_length_exceeds(sent.translation, translate_soft_limit, target_language)
            if src_exceeds or tr_exceeds:
                to_split.append(i)

        if not to_split:
            console.print("[green]✅ All subtitles are within length limits![/green]")
            break

        # 处理需要切分的句子，构建新列表
        new_sentences = []
        for i, sent in enumerate(sentences):
            if i not in to_split:
                # 不需要拆分，直接添加
                new_sentences.append(sent)
                continue

            # 需要拆分
            # 计算需要拆分成几份
            text_length = get_effective_length(sent.text, asr_language)
            num_parts = max(2, math.ceil(text_length / origin_soft_limit))

            # 使用 LLM 拆分原文
            split_src = split_sentence(sent.text, num_parts=num_parts).strip()

            if '[br]' in split_src:
                # 需要拆分：使用 difflib 匹配找到 [br] 位置，拆分 chunks
                from core._3_2_split_meaning import find_br_positions_in_original
                from core.utils.sentence_tools import clean_word

                br_positions = find_br_positions_in_original(split_src, sent.text)

                if br_positions:
                    # 构建字符到 Chunk 的映射（使用清洗后的文本）
                    char_to_chunk = []
                    for chunk_idx, chunk in enumerate(sent.chunks):
                        cleaned_chunk_text = clean_word(chunk.text)
                        char_to_chunk.extend([chunk_idx] * len(cleaned_chunk_text))

                    # 确定 Chunk 拆分点
                    split_points = [0]
                    for br_pos in br_positions:
                        if br_pos < len(char_to_chunk):
                            chunk_idx = char_to_chunk[br_pos]
                            if chunk_idx not in split_points:
                                split_points.append(chunk_idx)
                    split_points.append(len(sent.chunks))
                    split_points = sorted(set(split_points))

                    # 拆分 Chunks，创建新的 Sentence 对象
                    for j in range(len(split_points) - 1):
                        start_idx = split_points[j]
                        end_idx = split_points[j + 1]

                        if start_idx >= end_idx:
                            continue

                        sub_chunks = sent.chunks[start_idx:end_idx]
                        sub_text = "".join(c.text for c in sub_chunks)

                        new_sentence = Sentence(
                            chunks=sub_chunks,
                            text=sub_text,
                            translation="",  # 译文需要后续对齐
                            start=sub_chunks[0].start,
                            end=sub_chunks[-1].end,
                            index=sent.index + j,
                            is_split=True
                        )
                        new_sentences.append(new_sentence)
                else:
                    # 没有找到拆分点，保持原样
                    new_sentences.append(sent)
            else:
                # LLM 认为不需要拆分
                new_sentences.append(sent)

        # 更新句子列表
        sentences = new_sentences

    # 保存结果到 CSV（向后兼容）
    split_src = [sent.text for sent in sentences]
    split_trans = [sent.translation for sent in sentences]
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_csv(_5_SPLIT_SUB, index=False, encoding='utf-8-sig')
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_csv(_5_REMERGED, index=False, encoding='utf-8-sig')

    console.print("[bold green]✅ Subtitle splitting completed![/bold green]")
    console.print(f'[cyan]📊 Returning {len(sentences)} Sentence objects to Stage 5[/cyan]')
    return sentences

if __name__ == '__main__':
    split_for_sub_main()
