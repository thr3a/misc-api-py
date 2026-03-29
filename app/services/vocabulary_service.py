import sqlite3
from pathlib import Path

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

    items: list[VocabularyItem] = Field(..., description="語彙リスト")


def fetch_random_words(
    pos_major: str,
    count: int = 10,
    level_min: int | None = None,
    level_max: int | None = None,
    include_proper_noun: bool = False,
    db_path: Path = DB_PATH,
) -> list[VocabularyItem]:
    """指定した品詞大分類に一致する語彙をランダムに取得する。

    Args:
        pos_major: 品詞大分類（例: "動詞"、"名詞"）
        count: 取得件数（デフォルト: 10）
        level_min: 留学生用語彙レベルの最小値（指定しない場合は下限なし）
        level_max: 留学生用語彙レベルの最大値（指定しない場合は上限なし）
        include_proper_noun: IS_PN（固有名詞: 地名・人名・会社名等）を含めるか（デフォルト: False）
        db_path: SQLiteデータベースファイルのパス（テスト時に差し替え可能）

    Returns:
        VocabularyItemのリスト
    """
    conditions = ["pos_major = ?"]
    params: list[object] = [pos_major]

    if not include_proper_noun:
        conditions.append("(vdrj_student_level_raw IS NULL OR vdrj_student_level_raw != 'IS_PN')")

    if level_min is not None:
        conditions.append("vdrj_student_level >= ?")
        params.append(level_min)

    if level_max is not None:
        conditions.append("vdrj_student_level <= ?")
        params.append(level_max)

    where_clause = " AND ".join(conditions)
    sql = (
        "SELECT notation, reading, pos_major, pos_middle, pos_minor, pos_detail,"
        " vdrj_student_level_raw, vdrj_student_level"
        f" FROM vocabulary WHERE {where_clause} ORDER BY RANDOM() LIMIT ?"
    )
    params.append(count)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
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
