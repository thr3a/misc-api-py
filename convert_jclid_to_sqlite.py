import re
import sqlite3

import pandas as pd

XLSX_PATH = "J_CLID_C7_ver1.0.xlsx"
DB_PATH = "jclid.db"

COLUMN_MAP = {
    "見通し\n番号": "id",
    "表記": "notation",
    "他表記": "alt_notation",
    "綴り": "reading",
    "注": "notes",
    "品詞": "_part_of_speech_raw",
    "活用型": "conjugation_type",
    "日本語教育\n語彙表\n_語彙の難易度": "jel_difficulty",
    "日本語を読むための\n語彙データベースVDRJ\n_旧日本語能力試験\n出題基準レベル": "vdrj_jlpt_level",
    "日本語を読むための\n語彙データベースVDRJ\n_留学生用語彙レベル": "vdrj_student_level_raw",
    "日中対照漢字語\nデータベースJKVC\n_意味対応(日＿中)": "jkvc_semantic",
    "教科書\nコーパス\n語彙表\n_初出学年": "textbook_first_grade",
    "みんなの\n日本語\n_初出冊": "minna_first_volume",
    "スピード\nマスター\n_初出冊": "speed_master_first_volume",
    "新経典\n日本語\n基礎教程\n_初出冊": "xinjingdian_first_volume",
    "統合No.": "integrated_no",
}

df = pd.read_excel(XLSX_PATH, sheet_name=0)
df = df.rename(columns=COLUMN_MAP)

# 文字列カラムの空白を NaN に統一
str_cols = df.select_dtypes(include="object").columns
df[str_cols] = df[str_cols].replace(r"^\s*$", None, regex=True)

# 品詞を「-」で分割して大分類・中分類・小分類・細分類に展開
pos_split = df["_part_of_speech_raw"].str.split("-", expand=True).reindex(columns=[0, 1, 2, 3])
df["pos_major"] = pos_split[0]  # 品詞大分類（例: 名詞、動詞）
df["pos_middle"] = pos_split[1]  # 品詞中分類（例: 普通名詞、固有名詞）
df["pos_minor"] = pos_split[2]  # 品詞小分類（例: 一般、サ変可能）
df["pos_detail"] = pos_split[3]  # 品詞細分類（例: 名）

# 元の生データカラムは不要なので削除
df = df.drop(columns=["_part_of_speech_raw"])


def _parse_student_level(raw: str | None) -> int | None:
    """IS_XXK または IS_XXK+ 形式の値を整数に変換する。それ以外は NULL を返す。"""
    if not isinstance(raw, str):
        return None
    m = re.fullmatch(r"IS_(\d+)K\+?", raw)
    if m:
        return int(m.group(1))
    return None


# vdrj_student_level_raw の生の値を保持しつつ、加工済み整数値を vdrj_student_level に格納
df["vdrj_student_level"] = df["vdrj_student_level_raw"].apply(_parse_student_level).astype("Int64")

with sqlite3.connect(DB_PATH) as conn:
    df.to_sql("vocabulary", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]

print(f"完了: {count} 件を {DB_PATH} の vocabulary テーブルに保存しました")
