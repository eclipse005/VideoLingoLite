#!/usr/bin/env python3
"""
测试[br]位置映射算法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core._3_llm_sentence_split import map_br_to_original_sentence

def test_br_mapping():
    # 测试用例1：LLM添加了"Whether"
    original = "or ICT is teaching you or Romeo is teaching you, none of this stuff is going to be 100% right."
    llm_result = "Whether ICT is teaching you or Romeo is teaching you,[br] none of this stuff is going to be 100% right."

    mapped = map_br_to_original_sentence(original, llm_result)
    print(f"原始: {original}")
    print(f"LLM:  {llm_result}")
    print(f"映射: {mapped}")
    print(f"预期: or ICT is teaching you or Romeo is teaching you,[br] none of this stuff is going to be 100% right.")
    print("-" * 80)

    # 测试用例2：LLM没有修改内容
    original2 = "This is a simple test sentence with a natural break point here."
    llm_result2 = "This is a simple test sentence[br] with a natural break point here."

    mapped2 = map_br_to_original_sentence(original2, llm_result2)
    print(f"原始: {original2}")
    print(f"LLM:  {llm_result2}")
    print(f"映射: {mapped2}")
    print("-" * 80)

    # 测试用例3：LLM删除了填充词
    original3 = "Uh, this is a test sentence with filler words, uh, that should be processed correctly."
    llm_result3 = "This is a test sentence with filler words[br] that should be processed correctly."

    mapped3 = map_br_to_original_sentence(original3, llm_result3)
    print(f"原始: {original3}")
    print(f"LLM:  {llm_result3}")
    print(f"映射: {mapped3}")
    print("-" * 80)

    # 测试用例4：无[br]的情况
    original4 = "This sentence has no br tag."
    llm_result4 = "This sentence has no br tag."

    mapped4 = map_br_to_original_sentence(original4, llm_result4)
    print(f"原始: {original4}")
    print(f"LLM:  {llm_result4}")
    print(f"映射: {mapped4}")
    print("-" * 80)

if __name__ == "__main__":
    test_br_mapping()