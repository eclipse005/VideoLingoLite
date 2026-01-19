import os
import pandas as pd
from core.utils.config_utils import load_key, get_joiner
from rich import print as rprint

def split_by_pause():
    # 输入输出路径
    input_nlp_path = r"output\log\split_by_nlp.txt"
    chunks_path = r"output\log\cleaned_chunks.xlsx"

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

    # 1. 读取 NLP 结果
    with open(input_nlp_path, 'r', encoding='utf-8') as f:
        # 重点：strip() 去掉换行，然后 strip('"') 去掉你提到的每一行前后的引号
        raw_lines = [line.strip().strip('"') for line in f.readlines() if line.strip()]
    
    # 2. 读取时间戳 Excel
    chunks_df = pd.read_excel(chunks_path)
    
    print(f"📊 成功加载 {len(raw_lines)} 行文本和 {len(chunks_df)} 个词块时间戳。")

    final_sentences = []
    chunk_idx = 0
    total_chunks = len(chunks_df)

    # 3. 匹配与检测逻辑
    for sentence in raw_lines:
        # 为了比对，去掉句子中所有空格并转小写
        target_text = "".join(sentence.split()).lower()
        
        current_sentence_chunks = []
        collected_text = ""
        
        # 在 chunks 列表中滑动，找到属于这一行文本的所有词块
        while chunk_idx < total_chunks and len(collected_text) < len(target_text):
            # 同样去掉 chunk 文本中的引号进行比对
            chunk_raw = str(chunks_df.iloc[chunk_idx]['text']).strip('"')
            clean_chunk = "".join(chunk_raw.split()).lower()
            
            collected_text += clean_chunk
            current_sentence_chunks.append(chunks_df.iloc[chunk_idx])
            chunk_idx += 1
            
        # 如果只对应 1 个词块，无法计算间隔，直接保留
        if len(current_sentence_chunks) < 2:
            final_sentences.append(sentence)
            continue

        # 4. 核心：检查物理间隙 (Pause)
        temp_group = []
        last_chunk = current_sentence_chunks[0]
        temp_group.append(str(last_chunk['text']).strip('"'))

        for i in range(1, len(current_sentence_chunks)):
            current_chunk = current_sentence_chunks[i]
            
            # 计算间隙：当前词 Start - 上个词 End
            gap = current_chunk['start'] - last_chunk['end']
            
            if gap > pause_threshold:
                # 间隙过大，强制断开成新行
                final_sentences.append(joiner.join(temp_group))
                print(f"✂️  检测到停顿 {gap:.2f}s，已在词块 '{last_chunk['text']}' 后断句")
                temp_group = [str(current_chunk['text']).strip('"')]
            else:
                temp_group.append(str(current_chunk['text']).strip('"'))
            
            last_chunk = current_chunk
            
        if temp_group:
            final_sentences.append(joiner.join(temp_group))

    # 5. 写回原文件
    with open(input_nlp_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(final_sentences))

    print(f"\n✨ 处理完成！")
    print(f"📝 原始行数: {len(raw_lines)} -> 切分后行数: {len(final_sentences)}")
    print(f"💾 已更新文件：{input_nlp_path}")

if __name__ == '__main__':
    split_by_pause()