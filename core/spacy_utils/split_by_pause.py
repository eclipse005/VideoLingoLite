import os
import pandas as pd
import difflib
import unicodedata
from core.utils.config_utils import load_key, get_joiner
from core.spacy_utils.load_nlp_model import SPLIT_BY_NLP_FILE, SPLIT_BY_PAUSE_FILE
from rich import print as rprint
from core.utils.models import _2_CLEANED_CHUNKS


def is_latin_text(text):
    """检测文本是否是拉丁字母或数字（包括拉丁字符集）"""
    if not text:
        return False

    for char in text:
        code = ord(char)
        # Basic Latin (0x0020-0x007F): 空格, a-z, A-Z, 0-9, 常见标点
        # Latin-1 Supplement (0x0080-0x00FF): è, é, ê, ë, à, â, ä, 等
        # Latin Extended-A (0x0100-0x017F): Ā, ā, Ă, ă, Ą, ą, 等
        # Latin Extended-B (0x0180-0x024F): ƀ, Ƀ, ɂ, 等
        if not (0x0020 <= code <= 0x024F):
            return False
    return True


def smart_join(chunks):
    """智能连接 chunks：如果相邻两个都是拉丁字母，用空格连接"""
    if not chunks:
        return ""

    result = str(chunks[0]).strip('"')
    for i in range(1, len(chunks)):
        prev = result
        curr = str(chunks[i]).strip('"')

        # 检测前一个文本的结尾和当前文本的开头是否都是拉丁字母
        if prev and curr:
            prev_last = prev[-1] if prev else ""
            curr_first = curr[0] if curr else ""

            if is_latin_text(prev_last) and is_latin_text(curr_first):
                result += " " + curr
            else:
                result += curr
        else:
            result += curr

    return result


def split_by_pause():
    # 输入输出路径 - 使用统一的常量
    input_nlp_path = SPLIT_BY_NLP_FILE
    chunks_path = _2_CLEANED_CHUNKS
    # 直接覆盖 split_by_nlp.txt，这样后续的 split_by_meaning.py 能读取到修复后的内容
    output_pause_path = SPLIT_BY_NLP_FILE

    # Get pause threshold from config (0 or null means disabled)
    pause_threshold = load_key("pause_split_threshold")
    if pause_threshold is None or pause_threshold == 0:
        rprint("[yellow]⏭️  Pause-based splitting is disabled (pause_split_threshold=0 or null)[/yellow]")
        return

    # Get language and joiner from config
    language = load_key("asr.language")
    joiner = get_joiner(language)
    rprint(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")
    rprint(f"[blue]🔍 Pause threshold: {pause_threshold}s[/blue]")

    print(f"🚀 开始执行物理停顿切分...")

    if not os.path.exists(input_nlp_path) or not os.path.exists(chunks_path):
        print(f"❌ 错误：找不到文件。请确保以下路径存在：\n- {input_nlp_path}\n- {chunks_path}")
        return

    # ========== 加载数据文件 ==========
    chunks_df = pd.read_excel(chunks_path)
    with open(input_nlp_path, 'r', encoding='utf-8') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    # ========== 边界修复：逐字符对比法 ==========
    # 既然 nlp 拼接 = chunks 拼接，就两边同时遍历，发现不一致就修复
    print("🔧 开始修复 nlp 句子边界...")

    # 预处理：建立 (字符, chunk索引) 的映射
    chunk_char_map = []  # [(char, chunk_idx), ...]
    for idx, row in chunks_df.iterrows():
        chunk_text = str(row['text']).strip('"')
        if joiner == " ":
            cleaned = "".join(chunk_text.split()).lower()
        else:
            cleaned = chunk_text.lower()
        for char in cleaned:
            chunk_char_map.append((char, idx))

    # 提取纯字符列表
    chunk_chars = [char for char, _ in chunk_char_map]

    # 逐字符对比，发现不一致就修复
    fixed_lines = list(raw_lines)  # 复制一份
    i = 0  # nlp 句子索引
    j = 0  # chunk 字符索引
    changes = 0

    while i < len(fixed_lines) and j < len(chunk_chars):
        line = fixed_lines[i]
        if joiner == " ":
            line_chars = list("".join(line.split()).lower())
        else:
            line_chars = list(line.lower())

        # 逐字符对比
        k = 0
        while k < len(line_chars) and j < len(chunk_chars):
            if line_chars[k] == chunk_chars[j]:
                k += 1
                j += 1
            else:
                # 不匹配！这不应该发生（因为 nlp 拼接 = chunks 拼接）
                # 跳过不匹配的字符
                print(f"  警告: line[{i}] position {k} '{line_chars[k]}' != chunk[{j}] '{chunk_chars[j]}'")
                j += 1

        # 检查：line 是否在 chunk 中间结束？
        # 如果 j < len(chunk_chars)，检查当前 chunk 是否已完成
        if k >= len(line_chars) and j < len(chunk_chars):
            current_chunk_idx = chunk_char_map[j][1]

            # 找到当前 chunk 的起始位置
            chunk_start_idx = j
            while chunk_start_idx > 0 and chunk_char_map[chunk_start_idx - 1][1] == current_chunk_idx:
                chunk_start_idx -= 1

            # 找到当前 chunk 的结束位置
            chunk_end_idx = j
            while chunk_end_idx < len(chunk_char_map) and chunk_char_map[chunk_end_idx][1] == current_chunk_idx:
                chunk_end_idx += 1

            # 如果当前 chunk 未完成（j 不在 chunk 起始位置），说明 line 在 chunk 中间结束
            # j > chunk_start_idx 表示 j 在 chunk 中间，spaCy 在 chunk 内部进行了切分
            if j > chunk_start_idx:
                # 当前 chunk 的一部分在当前 line，另一部分在下一个 line
                # 需要把 chunk 剩余部分移到当前 line
                if i + 1 < len(fixed_lines):
                    next_line = fixed_lines[i + 1]
                    if joiner == " ":
                        next_line_chars = list("".join(next_line.split()).lower())
                    else:
                        next_line_chars = list(next_line.lower())

                    # chunk 中剩余的字符数量
                    chunk_remaining = chunk_end_idx - j
                    # 当前 line 已经结束，需要从下一个 line 取 chunk_remaining 个字符
                    if len(next_line_chars) >= chunk_remaining:
                        # 从下一个 line 取字符
                        chars_to_move = ''.join(next_line_chars[:chunk_remaining])
                        # 加到当前 line
                        fixed_lines[i] = line + chars_to_move
                        # 从下一个 line 去掉这些字符
                        fixed_lines[i + 1] = next_line[chunk_remaining:] if len(next_line) > chunk_remaining else ''
                        print(f"  边界修复: line[{i}] 加 '{chars_to_move}'（来自 chunk {current_chunk_idx} 的剩余部分）")
                        changes += 1
                        # 更新 j 位置
                        j = chunk_end_idx
                    else:
                        print(f"  警告: line[{i+1}] 字符不足，无法修复 chunk {current_chunk_idx} 的分割")
                        j = chunk_end_idx
                else:
                    print(f"  警告: line[{i}] 是最后一行，无法修复 chunk {current_chunk_idx} 的分割")
                    j = chunk_end_idx

        # 移动到下一行
        i += 1

    if changes > 0:
        print(f"✅ 共修复 {changes} 处边界错误")
    else:
        print("ℹ️  未发现需要修复的边界")

    raw_lines = fixed_lines
    # =======================================================================

    print(f"📊 成功加载 {len(raw_lines)} 行文本和 {len(chunks_df)} 个词块时间戳。")

    # 3. 预处理：建立 (清洗后的字符 -> chunk索引) 的映射
    # 这样即使 spaCy 修改了文本，我们依然能通过模糊匹配找到对应的 chunks
    chunk_char_map = []  # [(char, chunk_idx), ...]
    for idx, row in chunks_df.iterrows():
        chunk_text = str(row['text']).strip('"')
        # 根据 joiner 决定是否保留空格
        if joiner == " ":
            cleaned = "".join(chunk_text.split()).lower()
        else:
            cleaned = chunk_text.lower()
        # 为每个字符建立映射
        for char in cleaned:
            chunk_char_map.append({'char': char, 'chunk_idx': idx})

    # 提取纯字符列表用于 difflib 匹配
    all_chars = [item['char'] for item in chunk_char_map]

    # 定义句末标点（用于边界修复）
    sentence_end_punctuations = ['。', '！', '？', '.', '!', '?']

    # 记录当前在字符列表中的位置
    current_char_pos = 0
    final_sentences = []

    # 4. 对每个句子进行匹配和切分
    for sentence in raw_lines:
        # 根据 joiner 决定是否去掉空格
        if joiner == " ":
            sentence_chars = list("".join(sentence.split()).lower())
        else:
            sentence_chars = list(sentence.lower())

        # 如果句子为空，跳过
        if not sentence_chars:
            continue

        # 在剩余字符列表里进行模糊匹配
        remaining_chars = all_chars[current_char_pos:]

        s = difflib.SequenceMatcher(None, remaining_chars, sentence_chars, autojunk=False)
        matching_blocks = s.get_matching_blocks()

        # 寻找匹配块：必须匹配句子的开头 (b_start == 0)
        match_start_rel_idx = -1
        match_length = 0

        for a_start, b_start, length in matching_blocks:
            if b_start == 0:
                match_start_rel_idx = a_start
                match_length = length
                break

        if match_start_rel_idx == -1:
            print(f"\n⚠️ Warning: No close match found for sentence: {sentence}")
            # 兜底策略：保留原句，不切分
            final_sentences.append(sentence)
            continue

        # 计算匹配开始在字符列表中的绝对位置
        absolute_start_char_idx = current_char_pos + match_start_rel_idx

        # 计算匹配结束位置（处理非连续匹配）
        absolute_end_char_idx = absolute_start_char_idx + match_length - 1

        if match_length < len(sentence_chars):
            found_chars_count = match_length
            for a_start, b_start, length in matching_blocks:
                if a_start <= match_start_rel_idx:
                    continue
                # 如果这一块接着上一块的句子内容
                if b_start == found_chars_count:
                    additional_len = min(length, len(sentence_chars) - found_chars_count)
                    absolute_end_char_idx = current_char_pos + a_start + additional_len - 1
                    found_chars_count += additional_len
                    if found_chars_count >= len(sentence_chars):
                        break

        # 越界保护
        max_char_idx = len(chunk_char_map) - 1
        absolute_end_char_idx = min(absolute_end_char_idx, max_char_idx)
        absolute_start_char_idx = min(absolute_start_char_idx, max_char_idx)
        if absolute_end_char_idx < absolute_start_char_idx:
            absolute_end_char_idx = absolute_start_char_idx

        # 5. 根据字符位置获取对应的 chunk 范围
        start_chunk_idx = chunk_char_map[absolute_start_char_idx]['chunk_idx']
        end_chunk_idx = chunk_char_map[absolute_end_char_idx]['chunk_idx']

        # 获取这些 chunks 的数据
        sentence_chunks = chunks_df.iloc[start_chunk_idx:end_chunk_idx + 1]

        # 如果句子为空或只有 1 个 chunk，无法计算间隔，直接保留
        if len(sentence_chunks) < 2:
            final_sentences.append(sentence)
            # 更新位置到匹配结束后的下一个字符
            current_char_pos = absolute_end_char_idx + 1
            if current_char_pos >= len(all_chars):
                current_char_pos = len(all_chars) - 1
            continue

        # 6. 检查物理间隙 (Pause)，必要时进一步切分
        temp_group = []
        last_chunk = sentence_chunks.iloc[0]
        temp_group.append(str(last_chunk['text']).strip('"'))

        for i in range(1, len(sentence_chunks)):
            current_chunk = sentence_chunks.iloc[i]

            # 计算间隙：当前词 Start - 上个词 End
            gap = current_chunk['start'] - last_chunk['end']

            if gap > pause_threshold:
                # 间隙过大，强制断开成新行
                final_sentences.append(smart_join(temp_group))
                print(f"✂️  检测到停顿 {gap:.2f}s，已在词块 '{last_chunk['text']}' 后断句")
                temp_group = [str(current_chunk['text']).strip('"')]
            else:
                temp_group.append(str(current_chunk['text']).strip('"'))

            last_chunk = current_chunk

        if temp_group:
            final_sentences.append(smart_join(temp_group))

        # 7. 更新位置到匹配结束后的下一个字符
        current_char_pos = absolute_end_char_idx + 1
        if current_char_pos >= len(all_chars):
            current_char_pos = len(all_chars) - 1

    # 8. 写入文件
    with open(output_pause_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(final_sentences))

    print(f"\n✨ 处理完成！")
    print(f"📝 原始行数: {len(raw_lines)} -> 切分后行数: {len(final_sentences)}")
    print(f"💾 已保存到文件：{output_pause_path}")


if __name__ == '__main__':
    split_by_pause()
