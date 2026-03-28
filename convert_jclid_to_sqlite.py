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
    "品詞": "part_of_speech",
    "活用型": "conjugation_type",
    "日本語教育\n語彙表\n_語彙の難易度": "jel_difficulty",
    "日本語を読むための\n語彙データベースVDRJ\n_旧日本語能力試験\n出題基準レベル": "vdrj_jlpt_level",
    "日本語を読むための\n語彙データベースVDRJ\n_留学生用語彙レベル": "vdrj_student_level",
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

with sqlite3.connect(DB_PATH) as conn:
    df.to_sql("vocabulary", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]

print(f"完了: {count} 件を {DB_PATH} の vocabulary テーブルに保存しました")
