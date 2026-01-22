import pandas as pd
from typing import List
import math
import concurrent.futures

from core.utils import *
from core.utils.models import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def align_translation_with_source(src_text: str, tr_text: str, split_src: str) -> List[str]:
    """
    对齐译文与原文拆分（使用 LLM）

    Args:
        src_text: 原文
        tr_text: 译文
        split_src: 带有 [br] 标记的拆分后原文

    Returns:
        对齐拆分后的译文列表
    """
    from core.prompts import get_align_prompt

    align_prompt = get_align_prompt(src_text, tr_text, split_src)

    def valid_align(response_data):
        if not isinstance(response_data, dict):
            return {"status": "error", "message": "Response must be a dictionary"}
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required field: 'align'"}
        if not isinstance(response_data['align'], list):
            return {"status": "error", "message": "Field 'align' must be a list"}
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Field 'align' must contain at least 2 items"}
        for i, item in enumerate(response_data['align']):
            if not isinstance(item, dict):
                return {"status": "error", "message": f"align[{i}] must be a dictionary"}
            if f'target_part_{i+1}' not in item:
                return {"status": "error", "message": f"Missing required field: 'target_part_{i+1}' in align[{i}]"}
        return {"status": "success", "message": "Align validation completed"}

    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']

    # 提取对齐后的译文
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    return tr_parts


def process_single_sentence_split(sent: Sentence, num_parts: int, index: int, asr_language: str) -> List[Sentence]:
    """
    处理单个句子的拆分对齐（用于并发调用）

    Args:
        sent: 要拆分的 Sentence 对象
        num_parts: 需要拆分成几份
        index: 句子索引
        asr_language: ASR 语言

    Returns:
        拆分对齐后的 Sentence 对象列表
    """
    from core._3_2_split_meaning import find_br_positions_in_original
    from core.utils.sentence_tools import clean_word

    # 使用 LLM 拆分原文
    split_src = split_sentence(sent.text, num_parts=num_parts, index=index).strip()

    if '[br]' not in split_src:
        # LLM 认为不需要拆分对齐
        return [sent]

    # 调用 LLM 对齐译文
    tr_parts = align_translation_with_source(sent.text, sent.translation, split_src)

    # 需要拆分对齐：使用 difflib 匹配找到 [br] 位置，拆分 chunks
    br_positions = find_br_positions_in_original(split_src, sent.text)

    if not br_positions:
        # 没有找到拆分点，保持原样
        return [sent]

    # 构建字符到 Chunk 的映射
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

    # 获取语言连接符
    joiner = get_joiner(asr_language)

    # 拆分 Chunks，创建新的 Sentence 对象
    new_sentences = []
    for j in range(min(len(split_points) - 1, len(tr_parts))):
        start_idx = split_points[j]
        end_idx = split_points[j + 1]

        if start_idx >= end_idx:
            continue

        sub_chunks = sent.chunks[start_idx:end_idx]
        sub_text = joiner.join(c.text for c in sub_chunks)

        # 使用对齐后的译文
        sub_translation = tr_parts[j] if j < len(tr_parts) else ""

        new_sentence = Sentence(
            chunks=sub_chunks,
            text=sub_text,
            translation=sub_translation,
            start=sub_chunks[0].start,
            end=sub_chunks[-1].end,
            index=sent.index + j,
            is_split=True
        )
        new_sentences.append(new_sentence)

    return new_sentences


@timer("拆分对齐")
def split_for_sub_main(sentences: List[Sentence]) -> List[Sentence]:
    """
    字幕拆分对齐主函数，处理 Sentence 对象

    Args:
        sentences: Sentence 对象列表

    Returns:
        List[Sentence]: 切分对齐后的 Sentence 对象列表
    """
    console.print(f"[cyan]🔍 开始拆分对齐，共 {len(sentences)} 个句子[/cyan]")

    # Get source and target language ISO codes
    asr_language = load_key("asr.language")
    target_lang_desc = load_key("target_language")
    target_language = TARGET_LANG_MAP.get(target_lang_desc, 'en')

    # Get soft limits for source and target languages
    origin_soft_limit = get_language_length_limit(asr_language, 'origin')
    translate_soft_limit = get_language_length_limit(target_language, 'translate')

    # 多轮拆分对齐
    for attempt in range(3):
        console.print(f"[dim]检查长度限制 (第 {attempt + 1} 轮)...[/dim]")

        # 找出需要拆分对齐的句子
        to_split = []
        for i, sent in enumerate(sentences):
            src_exceeds = check_length_exceeds(sent.text, origin_soft_limit, asr_language)
            tr_exceeds = check_length_exceeds(sent.translation, translate_soft_limit, target_language)
            if src_exceeds or tr_exceeds:
                to_split.append(i)

            # 每100个句子显示一次进度
            if (i + 1) % 100 == 0:
                console.print(f"[dim]已检查 {i + 1}/{len(sentences)} 个句子...[/dim]")

        console.print(f"[dim]长度检查完成，发现 {len(to_split)} 个需要拆分的句子[/dim]")

        if not to_split:
            if attempt > 0:
                console.print('[green]✅ 所有句子已符合长度限制[/green]')
            break

        if to_split:
            console.print(f'[yellow]⚠️ 发现 {len(to_split)} 个需要拆分的句子[/yellow]')

        # 处理需要拆分对齐的句子，构建新列表
        new_sentences = [None] * len(sentences)
        total_to_split = len(to_split)

        # 先填充不需要拆分对齐的句子
        for i, sent in enumerate(sentences):
            if i not in to_split:
                new_sentences[i] = [sent]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            transient=False,
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]第 {attempt + 1} 轮: 拆分对齐 {total_to_split} 个句子...", total=total_to_split)

            with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
                # 提交所有需要拆分的任务
                futures = {}
                for i in to_split:
                    sent = sentences[i]
                    # 计算需要拆分成几份
                    text_length = get_effective_length(sent.text, asr_language)
                    num_parts = max(2, math.ceil(text_length / origin_soft_limit))

                    # 提交 LLM 拆分对齐任务
                    future = executor.submit(
                        process_single_sentence_split,
                        sent,
                        num_parts,
                        i,
                        asr_language
                    )
                    futures[future] = i

                # 按完成顺序收集拆分对齐结果
                for future in concurrent.futures.as_completed(futures.keys()):
                    index = futures[future]
                    split_result_list = future.result()
                    new_sentences[index] = split_result_list
                    progress.update(task, advance=1)

        # 展平拆分对齐后的句子列表
        sentences = [s for sublist in new_sentences for s in sublist]

    # 保存结果到 CSV（向后兼容）
    split_src = [sent.text for sent in sentences]
    split_trans = [sent.translation for sent in sentences]
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_csv(_5_SPLIT_SUB, index=False, encoding='utf-8-sig')
    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_csv(_5_REMERGED, index=False, encoding='utf-8-sig')

    console.print("[bold green]✅ 字幕拆分对齐完成！[/bold green]")
    return sentences


if __name__ == '__main__':
    # 测试需要从外部提供 sentences
    print("This module requires Sentence objects as input.")
