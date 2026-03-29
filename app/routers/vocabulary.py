import sqlite3
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field

DB_PATH = Path(__file__).parent.parent.parent / "jclid.db"


class VocabularyItem(BaseModel):
    """語彙アイテム"""

    notation: str = Field(..., description="表記")
    reading: str | None = Field(None, description="読み仮名（カタカナ）")
    pos_major: str | None = Field(None, description="品詞大分類（例: 名詞、動詞）")
    pos_middle: str | None = Field(None, description="品詞中分類（例: 普通名詞、固有名詞）")
    pos_minor: str | None = Field(None, description="品詞小分類（例: 一般、サ変可能）")
    pos_detail: str | None = Field(None, description="品詞細分類（例: 一般、名）")
    vdrj_student_level_raw: str | None = Field(None, description="留学生用語彙レベル（生の値: IS_01K、IS_07K 等）")
    vdrj_student_level: int | None = Field(None, description="留学生用語彙レベル（数値: IS_02K→2、IS_21K+→21、それ以外はNULL）")


class VocabularyListResponse(BaseModel):
    """語彙リストレスポンス"""

    items: list[VocabularyItem] = Field(..., description="語彙リスト（10件）")


router = APIRouter(
    prefix="/vocabulary",
    tags=["vocabulary"],
)


def _fetch_random_words(pos_major: str) -> list[VocabularyItem]:
    """指定した品詞大分類に一致する語彙をランダムに10件取得する。"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT notation, reading, pos_major, pos_middle, pos_minor, pos_detail, vdrj_student_level_raw, vdrj_student_level FROM vocabulary WHERE pos_major = ? ORDER BY RANDOM() LIMIT 10",
            (pos_major,),
        )
        return [
            VocabularyItem(
                notation=row["notation"],
                reading=row["reading"],
                pos_major=row["pos_major"],
                pos_middle=row["pos_middle"],
                pos_minor=row["pos_minor"],
                pos_detail=row["pos_detail"],
                vdrj_student_level_raw=row["vdrj_student_level_raw"],
                vdrj_student_level=row["vdrj_student_level"],
            )
            for row in cursor.fetchall()
        ]


@router.get(
    "/verbs",
    summary="ランダム動詞の取得",
    description="動詞をランダムに10件返します。",
    response_model=VocabularyListResponse,
)
async def get_random_verbs() -> VocabularyListResponse:
    """ランダムな動詞を10件返します。"""
    items = _fetch_random_words("動詞")
    return VocabularyListResponse(items=items)


@router.get(
    "/nouns",
    summary="ランダム名詞の取得",
    description="名詞をランダムに10件返します。",
    response_model=VocabularyListResponse,
)
async def get_random_nouns() -> VocabularyListResponse:
    """ランダムな名詞を10件返します。"""
    items = _fetch_random_words("名詞")
    return VocabularyListResponse(items=items)
