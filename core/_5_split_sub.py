import pandas as pd
from typing import List
import concurrent.futures
import math

from core.utils import *
from core.utils.models import *
from rich.panel import Panel
from rich.console import Console

console = Console()


def split_for_sub_main(sentences: List[Sentence]) -> List[Sentence]:
    """
    字幕切分主函数，处理 Sentence 对象

    Args:
        sentences: Sentence 对象列表

    Returns:
        List[Sentence]: 切分后的 Sentence 对象列表
    """
    console.print("[bold green]🚀 Start splitting subtitles...[/]")

    # 📊 显示接收到的 Sentence 对象信息
    console.print(f'[cyan]📊 Received {len(sentences)} Sentence objects from Stage 3[/cyan]')
    has_translation = sum(1 for s in sentences if s.translation)
    console.print(f'[dim]Sentences with translation: {has_translation}/{len(sentences)}[/dim]')

    # Get source and target language ISO codes
    asr_language = load_key("asr.language")
    target_lang_desc = load_key("target_language")
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
                    # 注意：这里不需要添加空格，因为 find_br_positions_in_original 返回的位置
                    # 是基于清洗后的文本（clean_word 去除了空格），所以 char_to_chunk 也应该不包含空格
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
                    for j in range(len(split_points) - 1):
                        start_idx = split_points[j]
                        end_idx = split_points[j + 1]

                        if start_idx >= end_idx:
                            continue

                        sub_chunks = sent.chunks[start_idx:end_idx]
                        sub_text = joiner.join(c.text for c in sub_chunks)  # 使用 joiner 分隔

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
    # 测试需要从外部提供 sentences
    print("This module requires Sentence objects as input.")
