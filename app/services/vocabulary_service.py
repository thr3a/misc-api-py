## データベース: jclid.db

# 日本語語彙データベース（J-CLID: Japanese Corpus of Lexical Items Database）のSQLiteファイル。
# 総レコード数: 45,603件

# ### テーブル: vocabulary

# | カラム名 | 型 | 元CSVカラム名 | 説明 |
# |---|---|---|---|
# | id | INTEGER | 見通し番号 | レコードの通し番号 |
# | notation | TEXT | 表記 | 語彙の表記（漢字・かな等） |
# | alt_notation | TEXT | 他表記 | 別表記 |
# | reading | TEXT | 綴り | 読み仮名（カタカナ） |
# | notes | TEXT | 注 | 語義や補足説明 |
# | pos_major | TEXT | 品詞（大分類） | 品詞の大分類（例: 名詞、動詞、副詞） |
# | pos_middle | TEXT | 品詞（中分類） | 品詞の中分類（例: 普通名詞、固有名詞） |
# | pos_minor | TEXT | 品詞（小分類） | 品詞の小分類（例: 一般、サ変可能、地名） |
# | pos_detail | TEXT | 品詞（細分類） | 品詞の細分類（例: 一般、名）。値がない場合はNULL |
# | conjugation_type | TEXT | 活用型 | 活用の種類 |
# | jel_difficulty | TEXT | 日本語教育語彙表_語彙の難易度 | 日本語教育語彙表による難易度（例: 初級前半、中級後半、上級後半等） |
# | vdrj_jlpt_level | REAL | VDRJ_旧日本語能力試験出題基準レベル | 旧JLPTレベル（1〜4の数値） |
# | vdrj_student_level | INTEGER | VDRJ_留学生用語彙レベル（数値） | IS_XXK 形式から抽出した数値（例: 1, 7, 21）。IS_HE・IS_MI・IS_PN等は NULL |
# | vdrj_student_level_raw | TEXT | VDRJ_留学生用語彙レベル（生値） | 元の生テキスト（例: IS_01K、IS_07K、IS_HE、IS_PN等） |
# | jkvc_semantic | TEXT | JKVC_意味対応(日＿中) | 日中対照漢字語データベースにおける意味対応 |
# | textbook_first_grade | TEXT | 教科書コーパス語彙表_初出学年 | 学校教科書における初出学年（例: 小_前、中、高等） |
# | minna_first_volume | TEXT | みんなの日本語_初出冊 | 「みんなの日本語」テキストにおける初出冊数 |
# | speed_master_first_volume | TEXT | スピードマスター_初出冊 | 「スピードマスター」テキストにおける初出冊数 |
# | xinjingdian_first_volume | TEXT | 新経典日本語基礎教程_初出冊 | 「新経典日本語基礎教程」テキストにおける初出冊数 |
# | integrated_no | INTEGER | 統合No. | 語彙統合用の番号 |

# sqlite変換前のCSV

# ```csv
# 見通し番号,表記,他表記,綴り,注,品詞,活用型,日本語教育語彙表_語彙の難易度,日本語を読むための語彙データベースVDRJ_旧日本語能力試験出題基準レベル,日本語を読むための語彙データベースVDRJ_留学生用語彙レベル,日中対照漢字語データベースJKVC_意味対応(日＿中),教科書コーパス語彙表_初出学年,みんなの日本語_初出冊,スピードマスター_初出冊,新経典日本語基礎教程_初出冊,統合No.
# 1,亜,,ア,,,接頭辞,,上級後半,,1,IS_07K,＝,中,,,6621
# 2,あ,,ア,,気付き,感動詞,,,,初級1,N4,,,,134159
# 3,ああ,,アア,,感動詞（oh）,感動詞-一般,,初級後半,,4,IS_01K,,小_前,初級1,第一冊,337
# 4,ああ,,アア,,like that,副詞,,中級前半,,,,,小_前,,N4,,124329
# 5,あー,,アー,,,感動詞-フィラー,,,,,IS_HE,,中,,,58187
# 6,アー,,アー,,,名詞-普通名詞-一般,,,,,IS_MI,,中,,,58206
# 7,アーキタイプ,,アーキタイプ,,,名詞-普通名詞-一般,,,,,IS_21K+,,高,,,55507
# 8,アーク,,アーク,,arc,名詞-普通名詞-一般,,,,,IS_21K+,,高,,,27782
# 9,アーグラ,,アーグラ,,,名詞-固有名詞-地名-一般,,,,,IS_PN,,高,,,72363
# 10,アーケード,,アーケード,,,名詞-普通名詞-一般,,中級後半,,,IS_15K,,高,,,14262
# 11,アーサー,,アーサー,,,名詞-固有名詞-人名-一般,,,,,IS_PN,,小_後,,,59395
# ```

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
    pos_major: str | list[str],
    count: int = 10,
    level_min: int | None = None,
    level_max: int | None = None,
    include_proper_noun: bool = False,
    db_path: Path = DB_PATH,
) -> list[VocabularyItem]:
    """指定した品詞大分類に一致する語彙をランダムに取得する。

    Args:
        pos_major: 品詞大分類（例: "動詞"、"名詞"、または ["イ形容詞", "ナ形容詞"] のようなリスト）
        count: 取得件数（デフォルト: 10）
        level_min: 留学生用語彙レベルの最小値（指定しない場合は下限なし）
        level_max: 留学生用語彙レベルの最大値（指定しない場合は上限なし）
        include_proper_noun: IS_PN（固有名詞: 地名・人名・会社名等）を含めるか（デフォルト: False）
        db_path: SQLiteデータベースファイルのパス（テスト時に差し替え可能）

    Returns:
        VocabularyItemのリスト
    """
    pos_major_list = [pos_major] if isinstance(pos_major, str) else pos_major
    placeholders = ",".join("?" * len(pos_major_list))
    conditions = [f"pos_major IN ({placeholders})"]
    params: list[object] = list(pos_major_list)

    if not include_proper_noun:
        conditions.append("(vdrj_student_level_raw IS NULL OR vdrj_student_level_raw != 'IS_PN')")

    if level_min is not None:
        conditions.append("vdrj_student_level >= ?")
        params.append(level_min)

    if level_max is not None:
        conditions.append("vdrj_student_level <= ?")
        params.append(level_max)

    where_clause = " AND ".join(conditions)
    sql = f"SELECT notation, reading, pos_major, pos_middle, pos_minor, pos_detail, vdrj_student_level_raw, vdrj_student_level FROM vocabulary WHERE {where_clause} ORDER BY RANDOM() LIMIT ?"
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
