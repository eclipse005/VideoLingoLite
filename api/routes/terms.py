"""
术语管理接口
处理术语列表的读取和保存
操作 custom_terms.csv 文件
"""

from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
import os

from api.models.schemas import Term, TermsList

router = APIRouter()

# 术语文件路径
TERMS_FILE = "custom_terms.csv"


def load_terms_from_csv() -> List[Term]:
    """从 custom_terms.csv 加载术语"""
    if not os.path.exists(TERMS_FILE):
        # 文件不存在，创建空文件（使用英文列名）
        df = pd.DataFrame(columns=["src", "tgt", "note"])
        df.to_csv(TERMS_FILE, index=False, encoding="utf-8-sig")
        return []

    try:
        # 使用 safe_read_csv 读取
        from core.utils.config_utils import safe_read_csv
        df = safe_read_csv(TERMS_FILE)

        terms = []
        for _, row in df.iterrows():
            terms.append(Term(
                original=str(row.iloc[0]) if pd.notna(row.iloc[0]) else "",
                translation=str(row.iloc[1]) if pd.notna(row.iloc[1]) else "",
                notes=str(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else ""
            ))
        return terms
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"术语文件读取失败: {str(e)}")


def save_terms_to_csv(terms: List[Term]):
    """保存术语到 custom_terms.csv"""
    try:
        # 转换为 DataFrame（使用英文列名）
        data = {
            "src": [term.original for term in terms],
            "tgt": [term.translation for term in terms],
            "note": [term.notes for term in terms]
        }
        df = pd.DataFrame(data)

        # 保存为 CSV（使用 utf-8-sig 编码，Excel 能正确打开）
        df.to_csv(TERMS_FILE, index=False, encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"术语文件保存失败: {str(e)}")


@router.get("/terms", response_model=TermsList)
async def get_terms():
    """
    获取术语列表

    从 custom_terms.csv 读取所有术语
    """
    terms = load_terms_from_csv()
    return TermsList(terms=terms, count=len(terms))


@router.put("/terms", response_model=TermsList)
async def save_terms(terms_data: TermsList):
    """
    保存术语列表

    将术语列表保存到 custom_terms.csv
    """
    save_terms_to_csv(terms_data.terms)
    count = terms_data.count if terms_data.count is not None else len(terms_data.terms)
    return TermsList(terms=terms_data.terms, count=count)


@router.patch("/terms/{index}", response_model=Term)
async def update_term(index: int, term: Term):
    """
    更新单个术语

    根据索引更新指定位置的术语
    """
    # 加载现有术语
    terms = load_terms_from_csv()

    # 检查索引是否有效
    if index < 0 or index >= len(terms):
        raise HTTPException(status_code=404, detail=f"术语索引 {index} 不存在")

    # 更新术语
    terms[index] = term

    # 保存
    save_terms_to_csv(terms)

    return term


@router.post("/terms", response_model=Term)
async def add_term(term: Term):
    """
    添加单个术语

    添加一个新的术语到 custom_terms.csv
    """
    # 加载现有术语
    terms = load_terms_from_csv()

    # 添加新术语
    terms.append(term)

    # 保存
    save_terms_to_csv(terms)

    return term


@router.delete("/terms/{index}")
async def delete_term(index: int):
    """
    删除指定术语

    根据索引删除术语
    """
    # 加载现有术语
    terms = load_terms_from_csv()

    if index < 0 or index >= len(terms):
        raise HTTPException(status_code=404, detail="术语索引不存在")

    # 删除术语
    deleted_term = terms.pop(index)

    # 保存
    save_terms_to_csv(terms)

    return {"success": True, "message": "术语已删除", "deleted": deleted_term}
