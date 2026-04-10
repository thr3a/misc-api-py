Directory Structure:
```
.
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── routers
│   │   ├── __init__.py
│   │   ├── html2md.py
│   │   ├── items.py
│   │   ├── puzzle.py
│   │   ├── roleplay.py
│   │   └── vocabulary.py
│   └── services
│       ├── __init__.py
│       ├── calculator.py
│       ├── github_html2md.py
│       ├── puzzle_lib.py
│       ├── puzzle_service.py
│       ├── roleplay_repository.py
│       └── vocabulary_service.py
├── convert_jclid_to_sqlite.py
├── pyproject.toml
└── tests
    ├── __init__.py
    └── test_vocabulary_service.py

```

---
File: convert_jclid_to_sqlite.py
---
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

---
File: tests/test_vocabulary_service.py
---
"""app/services/vocabulary_service の単体テスト。

実際のjclid.dbには依存せず、インメモリSQLiteを使用する。
"""

import sqlite3
from pathlib import Path

import pytest

from app.services.vocabulary_service import VocabularyItem, fetch_random_words

# テスト用サンプルデータ（品詞ごとに複数件）
_SAMPLE_ROWS = [
    # (notation, reading, pos_major, pos_middle, pos_minor, pos_detail, vdrj_student_level_raw, vdrj_student_level)
    ("走る", "ハシル", "動詞", "一般", "一般", None, "IS_01K", 1),
    ("食べる", "タベル", "動詞", "一般", "一般", None, "IS_02K", 2),
    ("書く", "カク", "動詞", "一般", "一般", None, "IS_03K", 3),
    ("読む", "ヨム", "動詞", "一般", "一般", None, "IS_04K", 4),
    ("見る", "ミル", "動詞", "一般", "一般", None, "IS_05K", 5),
    ("山", "ヤマ", "名詞", "普通名詞", "一般", None, "IS_01K", 1),
    ("川", "カワ", "名詞", "普通名詞", "一般", None, "IS_02K", 2),
    ("東京", "トウキョウ", "名詞", "固有名詞", "地名", None, "IS_PN", None),
    ("大きい", "オオキイ", "イ形容詞", "一般", "一般", None, "IS_01K", 1),
    ("小さい", "チイサイ", "イ形容詞", "一般", "一般", None, "IS_02K", 2),
    ("静か", "シズカ", "ナ形容詞", "一般", "一般", None, "IS_01K", 1),
    ("便利", "ベンリ", "ナ形容詞", "一般", "一般", None, "IS_03K", 3),
]


@pytest.fixture()
def test_db(tmp_path: Path) -> Path:
    """テスト用インメモリ相当のSQLiteファイルを生成するフィクスチャ。"""
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE vocabulary (
                notation TEXT NOT NULL,
                reading TEXT,
                pos_major TEXT,
                pos_middle TEXT,
                pos_minor TEXT,
                pos_detail TEXT,
                vdrj_student_level_raw TEXT,
                vdrj_student_level INTEGER
            )
        """)
        conn.executemany(
            "INSERT INTO vocabulary VALUES (?,?,?,?,?,?,?,?)",
            _SAMPLE_ROWS,
        )
    return db_path


class TestFetchRandomWords:
    """fetch_random_words の単体テスト。"""

    def test_returns_only_verbs(self, test_db: Path) -> None:
        """pos_major="動詞" を指定すると動詞だけが返ること。"""
        results = fetch_random_words("動詞", db_path=test_db)
        assert len(results) > 0
        assert all(item.pos_major == "動詞" for item in results)

    def test_returns_only_nouns(self, test_db: Path) -> None:
        """pos_major="名詞" を指定すると名詞だけが返ること（IS_PN除外）。"""
        results = fetch_random_words("名詞", db_path=test_db)
        assert len(results) > 0
        assert all(item.pos_major == "名詞" for item in results)

    def test_returns_empty_list_for_unknown_pos(self, test_db: Path) -> None:
        """該当する品詞が存在しない場合は空リストを返すこと。"""
        results = fetch_random_words("形容詞", db_path=test_db)
        assert results == []

    def test_returns_list_of_vocabulary_items(self, test_db: Path) -> None:
        """返り値が list[VocabularyItem] であること。"""
        results = fetch_random_words("動詞", db_path=test_db)
        assert isinstance(results, list)
        assert all(isinstance(item, VocabularyItem) for item in results)

    def test_sets_notation_field(self, test_db: Path) -> None:
        """各アイテムの notation が空でないこと。"""
        results = fetch_random_words("動詞", db_path=test_db)
        assert all(item.notation for item in results)

    def test_excludes_proper_noun_by_default(self, test_db: Path) -> None:
        """デフォルトでは IS_PN（固有名詞）が除外されること。"""
        results = fetch_random_words("名詞", db_path=test_db)
        assert all(item.vdrj_student_level_raw != "IS_PN" for item in results if item.vdrj_student_level_raw is not None)

    def test_includes_proper_noun_when_specified(self, test_db: Path) -> None:
        """include_proper_noun=True のとき IS_PN が含まれること。"""
        results = fetch_random_words("名詞", include_proper_noun=True, db_path=test_db)
        assert any(item.vdrj_student_level_raw == "IS_PN" for item in results)

    def test_allows_none_for_vdrj_student_level(self, test_db: Path) -> None:
        """IS_PN など数値変換できない場合は vdrj_student_level が None になること。"""
        results = fetch_random_words("名詞", include_proper_noun=True, db_path=test_db)
        none_level_items = [item for item in results if item.vdrj_student_level is None]
        assert len(none_level_items) > 0

    def test_count_parameter(self, test_db: Path) -> None:
        """count=2 を指定すると最大2件しか返らないこと。"""
        results = fetch_random_words("動詞", count=2, db_path=test_db)
        assert len(results) <= 2

    def test_returns_at_most_ten_items_by_default(self, tmp_path: Path) -> None:
        """サンプルが10件超あってもデフォルトでは最大10件であること。"""
        db_path = tmp_path / "large.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE vocabulary (
                    notation TEXT NOT NULL,
                    reading TEXT,
                    pos_major TEXT,
                    pos_middle TEXT,
                    pos_minor TEXT,
                    pos_detail TEXT,
                    vdrj_student_level_raw TEXT,
                    vdrj_student_level INTEGER
                )
            """)
            # 15件の動詞を挿入
            conn.executemany(
                "INSERT INTO vocabulary VALUES (?,?,?,?,?,?,?,?)",
                [(f"動詞{i}", f"ドウシ{i}", "動詞", "一般", "一般", None, None, None) for i in range(15)],
            )
        results = fetch_random_words("動詞", db_path=db_path)
        assert len(results) == 10

    def test_level_min_filter(self, test_db: Path) -> None:
        """level_min=3 を指定すると vdrj_student_level >= 3 の語彙だけが返ること。"""
        results = fetch_random_words("動詞", level_min=3, db_path=test_db)
        assert len(results) > 0
        assert all(item.vdrj_student_level is not None and item.vdrj_student_level >= 3 for item in results)

    def test_level_max_filter(self, test_db: Path) -> None:
        """level_max=2 を指定すると vdrj_student_level <= 2 の語彙だけが返ること。"""
        results = fetch_random_words("動詞", level_max=2, db_path=test_db)
        assert len(results) > 0
        assert all(item.vdrj_student_level is not None and item.vdrj_student_level <= 2 for item in results)

    def test_level_range_filter(self, test_db: Path) -> None:
        """level_min=2, level_max=4 を指定すると範囲内の語彙だけが返ること。"""
        results = fetch_random_words("動詞", level_min=2, level_max=4, db_path=test_db)
        assert len(results) > 0
        assert all(
            item.vdrj_student_level is not None and 2 <= item.vdrj_student_level <= 4
            for item in results
        )

    def test_pos_major_as_list(self, test_db: Path) -> None:
        """pos_major にリストを渡すと複数品詞が返ること。"""
        results = fetch_random_words(["イ形容詞", "ナ形容詞"], db_path=test_db)
        assert len(results) > 0
        assert all(item.pos_major in {"イ形容詞", "ナ形容詞"} for item in results)


class TestFetchRandomAdjectives:
    """形容詞（イ形容詞・ナ形容詞）取得の単体テスト。"""

    def test_returns_only_adjectives(self, test_db: Path) -> None:
        """イ形容詞・ナ形容詞のみが返ること。"""
        results = fetch_random_words(["イ形容詞", "ナ形容詞"], db_path=test_db)
        assert len(results) > 0
        assert all(item.pos_major in {"イ形容詞", "ナ形容詞"} for item in results)

    def test_returns_both_i_and_na_adjectives(self, test_db: Path) -> None:
        """イ形容詞とナ形容詞の両方が含まれること。"""
        results = fetch_random_words(["イ形容詞", "ナ形容詞"], db_path=test_db)
        pos_set = {item.pos_major for item in results}
        assert "イ形容詞" in pos_set
        assert "ナ形容詞" in pos_set

    def test_does_not_include_verbs_or_nouns(self, test_db: Path) -> None:
        """動詞・名詞が含まれないこと。"""
        results = fetch_random_words(["イ形容詞", "ナ形容詞"], db_path=test_db)
        assert all(item.pos_major not in {"動詞", "名詞"} for item in results)

    def test_level_min_filter_for_adjectives(self, test_db: Path) -> None:
        """level_min=2 を指定すると vdrj_student_level >= 2 の形容詞だけが返ること。"""
        results = fetch_random_words(["イ形容詞", "ナ形容詞"], level_min=2, db_path=test_db)
        assert len(results) > 0
        assert all(item.vdrj_student_level is not None and item.vdrj_student_level >= 2 for item in results)

    def test_count_parameter_for_adjectives(self, test_db: Path) -> None:
        """count=1 を指定すると最大1件しか返らないこと。"""
        results = fetch_random_words(["イ形容詞", "ナ形容詞"], count=1, db_path=test_db)
        assert len(results) <= 1

---
File: tests/__init__.py
---

---
File: app/main.py
---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import html2md, items, puzzle, roleplay, vocabulary

app = FastAPI()

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.16.12:5173",
        "http://localhost:5173",
        "https://turai.work",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

app.include_router(items.router)
app.include_router(html2md.router)
app.include_router(roleplay.router)
app.include_router(vocabulary.router)
app.include_router(puzzle.router)


@app.get("/")
async def root() -> dict[str, str]:
    """トップページ。"""
    return {"message": "turai.work"}


@app.get("/health")
async def health() -> dict[str, str]:
    """ヘルスチェック用エンドポイント。"""
    return {"status": "ok"}

---
File: app/__init__.py
---

---
File: app/routers/roleplay.py
---
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.roleplay_repository import get_random_roleplay_record


class RoleplayTemplate(BaseModel):
    """ロールプレイ用プロンプトテンプレート。"""

    major_genre: str | None = Field(default=None, description="ジャンル（大分類）")
    minor_genre: str | None = Field(default=None, description="ジャンル（小分類）")
    world_setting: str | None = Field(default=None, description="舞台・世界観の設定")
    scene_setting: str | None = Field(default=None, description="対話シーンの設定")
    user_setting: str | None = Field(default=None, description="ユーザー側キャラクターの設定")
    assistant_setting: str | None = Field(default=None, description="アシスタント側キャラクターの設定")

    model_config = {
        "json_schema_extra": {
            "example": {
                "major_genre": "ファンタジー",
                "minor_genre": "学園もの",
                "world_setting": "魔法が日常的に使われる現代日本の学園",
                "scene_setting": "放課後の教室で、秘密の部活動について話している",
                "user_setting": "内気だが好奇心旺盛な一年生の魔法使い",
                "assistant_setting": "面倒見がよく少し意地悪な先輩魔法使い",
            }
        }
    }


router = APIRouter(
    prefix="/roleplay",
    tags=["roleplay"],
)


@router.get(
    "/random",
    summary="ランダムなロールプレイ設定の取得",
    description="起動時に読み込んだ hoge.parquet から 1 件ランダムにロールプレイ設定を返します。",
    response_model=RoleplayTemplate,
    response_description="ランダムに選ばれたロールプレイ設定",
)
async def get_random_roleplay() -> RoleplayTemplate:
    """ロールプレイテンプレートを 1 件ランダムに返します。"""
    record = get_random_roleplay_record()
    return RoleplayTemplate(**record)


---
File: app/routers/puzzle.py
---
"""Knights and Knaves パズル API ルーター."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.puzzle_service import generate_puzzle, solve_puzzle, to_text_list


class Statement(BaseModel):
    """パズルの発言モデル."""

    type: str = Field(..., description="発言タイプ（例: accusation, affirmation, sympathetic など）")
    speaker: str = Field(..., description="発言者の名前")
    target: str | None = Field(None, description="対象者の名前（タイプによっては不要）")
    target2: str | None = Field(None, description="2番目の対象者（pair_same / pair_diff / if_then / either_* のみ）")
    claimed_is_knight: bool | None = Field(None, description="主張する対象者の種類（disjoint / joint のみ）")
    n: int | None = Field(None, description="人数の閾値（at_least_knight / at_least_knave のみ）")
    condition_knight: bool | None = Field(None, description="条件部の種類（if_then のみ）")
    conclusion_knight: bool | None = Field(None, description="結論部の種類（if_then のみ）")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"type": "accusation", "speaker": "Aさん", "target": "Bさん"},
                {"type": "affirmation", "speaker": "Bさん", "target": "Aさん"},
            ]
        }
    }


class PuzzleGenerateRequest(BaseModel):
    """パズル生成リクエスト."""

    num_persons: int = Field(..., ge=2, le=100, description="人数（2〜100）")
    level: Literal["easy", "normal", "hard"] = Field(..., description="難易度（easy / normal / hard）")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"num_persons": 3, "level": "easy"},
                {"num_persons": 4, "level": "normal"},
            ]
        }
    }


class PuzzleResponse(BaseModel):
    """パズル生成レスポンス（solution と statements_text を含む）."""

    version: str = Field(..., description="パズルのバージョン")
    level: str = Field(..., description="難易度")
    num_persons: int = Field(..., description="人数")
    persons: list[str] = Field(..., description="登場人物のリスト")
    statements: list[Statement] = Field(..., description="発言のリスト")
    solution: dict[str, str] = Field(..., description='正解（{人名: "knight" | "knave"}）')
    statements_text: list[str] = Field(..., description="テキスト化した発言のリスト")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "version": "1.0",
                    "level": "easy",
                    "num_persons": 2,
                    "persons": ["Aさん", "Bさん"],
                    "statements": [
                        {"type": "accusation", "speaker": "Aさん", "target": "Bさん"},
                        {"type": "affirmation", "speaker": "Bさん", "target": "Aさん"},
                    ],
                    "solution": {"Aさん": "knight", "Bさん": "knave"},
                    "statements_text": [
                        "Aさんは言った: 「Bさんは悪党だ。」",
                        "Bさんは言った: 「Aさんは騎士だ。」",
                    ],
                }
            ]
        }
    }


class PuzzleSolveRequest(BaseModel):
    """パズル解答リクエスト."""

    persons: list[str] = Field(..., min_length=2, description="登場人物のリスト（2人以上）")
    statements: list[Statement] = Field(..., min_length=1, description="発言のリスト")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "persons": ["Aさん", "Bさん"],
                    "statements": [
                        {"type": "accusation", "speaker": "Aさん", "target": "Bさん"},
                        {"type": "affirmation", "speaker": "Bさん", "target": "Aさん"},
                    ],
                }
            ]
        }
    }


class PuzzleSolveResponse(BaseModel):
    """パズル解答レスポンス."""

    has_solution: bool = Field(..., description="解が存在するかどうか")
    solution: dict[str, str] | None = Field(None, description='解（{人名: "knight" | "knave"}）。解なしの場合は null')

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"has_solution": True, "solution": {"Aさん": "knight", "Bさん": "knave"}},
                {"has_solution": False, "solution": None},
            ]
        }
    }


class ErrorResponse(BaseModel):
    """エラーレスポンスの共通形式."""

    detail: str = Field(..., description="エラーメッセージの詳細")


router = APIRouter(
    prefix="/puzzle",
    tags=["puzzle"],
    responses={500: {"model": ErrorResponse, "description": "サーバー内部エラー"}},
)


@router.post(
    "/generate",
    summary="Knights and Knaves パズルの生成",
    description=(
        "指定した人数・難易度で Knights and Knaves パズルを生成します。\n\n"
        "- **easy**: accusation / affirmation / sympathetic / antithetic の4種類の発言のみ使用\n"
        "- **normal**: すべての発言タイプを使用（人数と同数の発言）\n"
        "- **hard**: すべての発言タイプを使用（人数 × 1.5 の発言）\n\n"
        "z3 ソルバーにより一意解を持つパズルのみ返します。"
    ),
    response_model=PuzzleResponse,
    response_description="生成されたパズル（solution と statements_text を含む）",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "リクエストパラメータが不正（人数が範囲外、難易度が不正など）",
            "content": {"application/json": {"example": {"detail": "num_persons は 2 以上が必要です"}}},
        },
        500: {
            "model": ErrorResponse,
            "description": "リトライ上限を超えてパズルを生成できなかった",
            "content": {"application/json": {"example": {"detail": "1000 回リトライしても一意なパズルを生成できませんでした"}}},
        },
    },
)
async def generate(req: PuzzleGenerateRequest) -> PuzzleResponse:
    """Knights and Knaves パズルを生成します."""
    try:
        puzzle = generate_puzzle(req.num_persons, req.level)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    statements = [Statement(**s) for s in puzzle["statements"]]
    return PuzzleResponse(
        version=puzzle["version"],
        level=puzzle["level"],
        num_persons=puzzle["num_persons"],
        persons=puzzle["persons"],
        statements=statements,
        solution=puzzle["solution"],
        statements_text=to_text_list(puzzle["statements"]),
    )


@router.post(
    "/solve",
    summary="Knights and Knaves パズルの解答",
    description=(
        "persons と statements を受け取り、z3 ソルバーで解を求めます。\n\n"
        "解が存在する場合は `has_solution: true` と `solution` を返します。\n"
        "制約が矛盾して解なしの場合は `has_solution: false`、`solution: null` を返します。"
    ),
    response_model=PuzzleSolveResponse,
    response_description="解答結果",
    responses={
        200: {
            "content": {
                "application/json": {
                    "examples": {
                        "解あり": {"value": {"has_solution": True, "solution": {"Aさん": "knight", "Bさん": "knave"}}},
                        "解なし": {"value": {"has_solution": False, "solution": None}},
                    }
                }
            }
        }
    },
)
async def solve(req: PuzzleSolveRequest) -> PuzzleSolveResponse:
    """Knights and Knaves パズルを解きます."""
    puzzle = {
        "persons": req.persons,
        "statements": [s.model_dump(exclude_none=True) for s in req.statements],
    }
    solution = solve_puzzle(puzzle)
    return PuzzleSolveResponse(has_solution=solution is not None, solution=solution)

---
File: app/routers/vocabulary.py
---
from fastapi import APIRouter, Query

from app.services.vocabulary_service import VocabularyListResponse, fetch_random_words

router = APIRouter(
    prefix="/vocabulary",
    tags=["vocabulary"],
)


@router.get(
    "/verbs",
    summary="ランダム動詞の取得",
    description="動詞をランダムに返します。件数・レベル範囲・固有名詞の包含可否を指定できます。",
    response_model=VocabularyListResponse,
)
async def get_random_verbs(
    count: int = Query(10, ge=1, le=500, description="取得件数（デフォルト: 10）"),
    level_min: int | None = Query(None, ge=1, le=21, description="留学生用語彙レベルの最小値"),
    level_max: int | None = Query(None, ge=1, le=21, description="留学生用語彙レベルの最大値"),
    include_proper_noun: bool = Query(False, description="固有名詞（IS_PN: 地名・人名・会社名等）を含めるか（デフォルト: false）"),
) -> VocabularyListResponse:
    """ランダムな動詞を返します。"""
    items = fetch_random_words(
        "動詞",
        count=count,
        level_min=level_min,
        level_max=level_max,
        include_proper_noun=include_proper_noun,
    )
    return VocabularyListResponse(items=items)


@router.get(
    "/nouns",
    summary="ランダム名詞の取得",
    description="名詞をランダムに返します。件数・レベル範囲・固有名詞の包含可否を指定できます。",
    response_model=VocabularyListResponse,
)
async def get_random_nouns(
    count: int = Query(10, ge=1, le=500, description="取得件数（デフォルト: 10）"),
    level_min: int | None = Query(None, ge=1, le=21, description="留学生用語彙レベルの最小値"),
    level_max: int | None = Query(None, ge=1, le=21, description="留学生用語彙レベルの最大値"),
    include_proper_noun: bool = Query(False, description="固有名詞（IS_PN: 地名・人名・会社名等）を含めるか（デフォルト: false）"),
) -> VocabularyListResponse:
    """ランダムな名詞を返します。"""
    items = fetch_random_words(
        "名詞",
        count=count,
        level_min=level_min,
        level_max=level_max,
        include_proper_noun=include_proper_noun,
    )
    return VocabularyListResponse(items=items)


@router.get(
    "/adjectives",
    summary="ランダム形容詞の取得",
    description="形容詞（イ形容詞・ナ形容詞）をランダムに返します。件数・レベル範囲を指定できます。",
    response_model=VocabularyListResponse,
)
async def get_random_adjectives(
    count: int = Query(10, ge=1, le=500, description="取得件数（デフォルト: 10）"),
    level_min: int | None = Query(None, ge=1, le=21, description="留学生用語彙レベルの最小値"),
    level_max: int | None = Query(None, ge=1, le=21, description="留学生用語彙レベルの最大値"),
) -> VocabularyListResponse:
    """ランダムな形容詞（イ形容詞・ナ形容詞）を返します。"""
    items = fetch_random_words(
        ["イ形容詞", "ナ形容詞"],
        count=count,
        level_min=level_min,
        level_max=level_max,
    )
    return VocabularyListResponse(items=items)

---
File: app/routers/html2md.py
---
import re
from urllib.parse import urlsplit

import autopager
import chardet
import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from httpx_curl_cffi import AsyncCurlTransport
from pydantic import BaseModel, Field, HttpUrl
from trafilatura import extract

from app.services.github_html2md import GithubMarkdownError, convert_github_url_to_markdown


class Html2MarkdownQuery(BaseModel):
    """HTML から Markdown へ変換するためのクエリパラメータ"""

    url: HttpUrl = Field(..., description="Markdown に変換する対象のページ URL")


router = APIRouter(
    prefix="/html2md",
    tags=["html2md"],
    responses={
        400: {"description": "ユーザー入力の不備"},
        500: {"description": "サーバー側のエラー"},
    },
)


async def _fetch(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    """指定 URL を取得する。失敗した場合は None を返す。

    Args:
        client: 再利用可能な HTTPX 非同期クライアント。
        url: 取得する URL。

    Returns:
        成功時はレスポンス、失敗時は None。
    """
    try:
        resp = await client.get(url, timeout=10)
    except httpx.HTTPError:
        return None
    if resp.status_code >= 400:
        return None
    return resp


def _decode_response(resp: httpx.Response) -> str:
    """レスポンスのコンテンツを適切な文字コードでデコードする。

    chardet を使用して文字コードを検出し、Shift-JIS などの非 UTF-8
    エンコーディングにも対応する。

    Args:
        resp: HTTPX レスポンスオブジェクト。

    Returns:
        デコードされた HTML 文字列。
    """
    # まず httpx の自動デコードを試す
    try:
        text = resp.text
        # デコード結果に文字化け（置換文字）が含まれている場合は chardet を使用
        if "" not in text:
            return text
    except Exception:
        pass

    # chardet で文字コードを検出
    detected = chardet.detect(resp.content)
    encoding = detected.get("encoding", "utf-8")
    confidence = detected.get("confidence", 0)

    # 信頼度が低い場合は UTF-8 を優先
    if confidence < 0.7:
        encoding = "utf-8"

    try:
        return resp.content.decode(encoding, errors="replace")
    except (UnicodeDecodeError, LookupError):
        # デコード失敗時は UTF-8 で再試行
        return resp.content.decode("utf-8", errors="replace")


def _extract_markdown(html: str, url: str) -> str | None:
    """HTML 文字列から trafilatura で Markdown を抽出する。

    Args:
        html: ページ HTML。
        url: ページの絶対 URL。

    Returns:
        成功時は Markdown、抽出失敗時は None。
    """
    result = extract(html, url=url, output_format="markdown", with_metadata=False)
    return result if isinstance(result, str) and result.strip() else None


_PAGE_IN_PATH_RE: re.Pattern[str] = re.compile(r"/(?:p|page)/(?P<num>\d+)(?:/|$)", re.IGNORECASE)
_PAGE_SUFFIX_RE: re.Pattern[str] = re.compile(r"(?:^|[^a-z])(?:p|page)[-]?(?P<num>\d+)(?:/|$)", re.IGNORECASE)
_PAGE_UNDERSCORE_RE: re.Pattern[str] = re.compile(r"_(?P<num>\d+)(?:\.html?|/|$)", re.IGNORECASE)


def _detect_page_number(url: str) -> int | None:
    """URL からページ番号を推定する。

    - クエリ文字列の `page` または `p` を優先
    - パスに含まれる `/page/2`, `/p/2`, `page-2`, `page2` などのパターンも対応
    - アンダースコアの後に数字が続くパターン（例: news051_2.html）も対応
    - 該当しない場合は None（呼び出し側で 1 とみなす）
    """
    from urllib.parse import parse_qs, urlsplit

    parts = urlsplit(url)
    q = parse_qs(parts.query)
    for key in ("page", "p"):
        vals = q.get(key)
        if vals:
            for v in vals:
                if v.isdigit():
                    return int(v)
    m = _PAGE_IN_PATH_RE.search(parts.path)
    if m:
        return int(m.group("num"))
    m = _PAGE_SUFFIX_RE.search(parts.path)
    if m:
        return int(m.group("num"))
    m = _PAGE_UNDERSCORE_RE.search(parts.path)
    if m:
        return int(m.group("num"))
    return None


def _build_pagination_urls(base_url: str, first_response: httpx.Response) -> list[tuple[int, str]]:
    """Autopager を用いてページネーション URL を抽出し、ページ番号順に整列する。

    - 同一ページ番号は先勝ちで重複排除
    - ページ番号が検出できない URL は 1 ページ扱い
    - 最低でも 1 ページ目（`base_url`）は含める

    Returns:
        (page_num, url) のタプルのリスト（page_num 昇順）。
    """
    urls: list[str] = []
    try:
        # httpx.Response をそのまま渡すと base URL 解決時に型不一致となる
        # ため、HTML 文字列と明示的な base_url を与える。
        # （requests.Response の .url は str だが httpx は httpx.URL 型）
        html_text = _decode_response(first_response)
        urls = autopager.urls(html_text, baseurl=base_url)
    except Exception:
        # autopager の解析に失敗した場合は後続で base_url のみ処理
        urls = []

    # 先頭に base_url を優先的に追加（1ページ目として扱う）
    candidates: list[str] = [base_url]
    # autopager は重複を返すことがあるため set で一度フィルタ（順序は保持）
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            candidates.append(u)
            seen.add(u)

    pages: dict[int, str] = {}
    for u in candidates:
        pnum = _detect_page_number(u) or 1
        # ページ番号ごとに先勝ち
        if pnum not in pages:
            pages[pnum] = u

    ordered: list[tuple[int, str]] = sorted(pages.items(), key=lambda t: t[0])
    return ordered


@router.get(
    "",
    summary="URL の HTML を Markdown に変換する",
    response_class=PlainTextResponse,
    response_description="変換された Markdown 文字列",
)
async def convert_html2md(query: Html2MarkdownQuery = Depends()) -> PlainTextResponse:
    """指定した URL の HTML を Markdown 形式のテキストとして返します。

    ページネーションが検出された場合は、各ページを取得・抽出して結合します。
    取得できなかったページはスキップし、ページ番号が検出できない URL は
    1 ページ目として扱います。
    """
    base_url = str(query.url)

    # GitHub の issue / discussion は専用ロジックで処理する
    parsed = urlsplit(base_url)
    host = parsed.netloc.lower()
    if host == "github.com" or host.endswith(".github.com"):
        try:
            markdown = await run_in_threadpool(convert_github_url_to_markdown, base_url)
        except GithubMarkdownError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - 想定外の例外
            msg = "GitHub からコンテンツを取得中にエラーが発生しました。"
            raise HTTPException(status_code=500, detail=msg) from exc

        return PlainTextResponse(content=markdown, media_type="text/markdown; charset=utf-8")

    headers = {"Accept-Language": "ja-JP"}
    transport = AsyncCurlTransport(
        impersonate="chrome",
        default_headers=True,
    )
    async with httpx.AsyncClient(transport=transport, headers=headers, follow_redirects=True) as client:
        # まず 1 ページ目を取得（失敗時は 400）
        first = await _fetch(client, base_url)
        if first is None:
            msg = "指定した URL からコンテンツを取得できませんでした。"
            raise HTTPException(status_code=400, detail=msg)

        # autopager で候補 URL を収集し、ページ番号順に整列
        ordered_pages: list[tuple[int, str]] = _build_pagination_urls(base_url, first)

        # 各ページを取得して markdown に変換（失敗はスキップ）
        markdown_parts: list[str] = []

        # 1ページ目は既にレスポンスがあるので先に処理
        first_html = _decode_response(first)
        first_md = await run_in_threadpool(_extract_markdown, first_html, base_url)
        if isinstance(first_md, str):
            markdown_parts.append(first_md)

        # 残りのページ（page_num > 1）を順に処理
        for page_num, page_url in ordered_pages:
            if page_num == 1:
                continue
            resp = await _fetch(client, page_url)
            if resp is None:
                continue  # 取得失敗はスキップ
            html = _decode_response(resp)
            md = await run_in_threadpool(_extract_markdown, html, page_url)
            if isinstance(md, str) and md:
                markdown_parts.append(md)

    # いずれのページも抽出できなかった場合はエラー
    if not markdown_parts:
        msg = "コンテンツの抽出に失敗しました。"
        raise HTTPException(status_code=400, detail=msg)

    combined = "\n\n".join(markdown_parts)
    return PlainTextResponse(content=combined, media_type="text/markdown; charset=utf-8")

---
File: app/routers/items.py
---
from fastapi import APIRouter, Body, HTTPException, Path
from pydantic import BaseModel, Field


class Item(BaseModel):
    """アイテムの基本情報"""

    name: str = Field(..., description="アイテム名")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "Plumbus"},
                {"name": "Portal Gun"},
            ]
        }
    }


class ItemWithId(Item):
    """アイテムの詳細情報（ID を含む）"""

    item_id: str = Field(..., description="アイテムの識別子（Path パラメータ）")


class ErrorResponse(BaseModel):
    """エラーレスポンスの共通形式"""

    detail: str = Field(..., description="エラーメッセージの詳細")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"detail": "Item not found"},
                {"detail": "You can only update the item: plumbus"},
            ]
        }
    }


router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={404: {"description": "Not found"}},
)
fake_items_db: dict[str, Item] = {"plumbus": Item(name="Plumbus"), "gun": Item(name="Portal Gun")}


@router.get(
    "/",
    summary="アイテム一覧の取得",
    description=("登録されているすべてのアイテムを返します。返却値はキーが item_id、値が Item モデルのオブジェクトである連想配列です。"),
    response_model=dict[str, Item],
    response_description="キーが item_id、値が Item モデルのオブジェクト",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "plumbus": {"name": "Plumbus"},
                        "gun": {"name": "Portal Gun"},
                    }
                }
            }
        }
    },
)
async def read_items() -> dict[str, Item]:
    """アイテムを全件取得します。"""
    return fake_items_db


@router.get(
    "/{item_id}",
    summary="アイテム詳細の取得",
    description="指定した item_id のアイテムを返します。存在しない場合は 404 を返します。",
    response_model=ItemWithId,
    response_description="指定した item_id のアイテム詳細",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "指定したアイテムが存在しません",
            "content": {"application/json": {"example": {"detail": "Item not found"}}},
        }
    },
)
async def read_item(item_id: str = Path(..., description="取得するアイテムID", examples={"例": {"value": "plumbus"}})) -> ItemWithId:
    """単一アイテムを取得します。"""
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemWithId(name=fake_items_db[item_id].name, item_id=item_id)


@router.put(
    "/{item_id}",
    tags=["custom"],
    summary="アイテムの更新（サンプル）",
    description="plumbus のみ更新可能なサンプル実装です。それ以外は 403 を返します。",
    response_model=ItemWithId,
    response_description="更新後のアイテム",
    responses={
        403: {
            "model": ErrorResponse,
            "description": "plumbus 以外の更新は禁止されています",
            "content": {"application/json": {"example": {"detail": "You can only update the item: plumbus"}}},
        }
    },
)
async def update_item(
    item_id: str = Path(..., description="更新するアイテムID", examples={"例": {"value": "plumbus"}}),
    item: Item = Body(..., description="更新するアイテムの内容"),
) -> ItemWithId:
    """アイテムを更新します（デモ用の制限あり）"""
    if item_id != "plumbus":
        raise HTTPException(status_code=403, detail="You can only update the item: plumbus")
    fake_items_db[item_id] = item
    return ItemWithId(**item.model_dump(), item_id=item_id)

---
File: app/routers/__init__.py
---

---
File: app/services/vocabulary_service.py
---
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

---
File: app/services/calculator.py
---
"""簡単な計算ユーティリティ。"""

from __future__ import annotations


def add(a: int | float, b: int | float) -> int | float:
    """2つの数値を足し算する。

    Args:
        a: 1つ目の数値。
        b: 2つ目の数値。

    Returns:
        足し算の結果。
    """
    return a + b


def subtract(a: int | float, b: int | float) -> int | float:
    """2つの数値を引き算する。

    Args:
        a: 1つ目の数値（被減数）。
        b: 2つ目の数値（減数）。

    Returns:
        引き算の結果。
    """
    return a - b


def multiply(a: int | float, b: int | float) -> int | float:
    """2つの数値を掛け算する。

    Args:
        a: 1つ目の数値。
        b: 2つ目の数値。

    Returns:
        掛け算の結果。
    """
    return a * b


def divide(a: int | float, b: int | float) -> float:
    """2つの数値を割り算する。

    Args:
        a: 1つ目の数値（被除数）。
        b: 2つ目の数値（除数）。

    Returns:
        割り算の結果。

    Raises:
        ZeroDivisionError: 除数が0の場合。
    """
    if b == 0:
        msg = "除数は0以外である必要があります。"
        raise ZeroDivisionError(msg)
    return a / b


def is_even(n: int) -> bool:
    """整数が偶数かどうかを判定する。

    Args:
        n: 判定する整数。

    Returns:
        偶数の場合はTrue、奇数の場合はFalse。
    """
    return n % 2 == 0


def factorial(n: int) -> int:
    """非負整数の階乗を計算する。

    Args:
        n: 非負整数。

    Returns:
        nの階乗。

    Raises:
        ValueError: nが負の数の場合。
    """
    if n < 0:
        msg = "nは0以上の整数である必要があります。"
        raise ValueError(msg)
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

---
File: app/services/puzzle_service.py
---
"""Knights and Knaves パズルの生成・解答サービス."""

import random

from z3 import And, Bool, Not, sat
from z3 import is_true as z3_is_true

from .puzzle_lib import (
    _COUNTING_TYPES_SET,
    _TWO_TARGET_TYPES_SET,
    STATEMENT_TYPES_ALL,
    STATEMENT_TYPES_EASY,
    add_constraint,
    build_solver,
    to_text,
)

MAX_VISIBLE_COUNT_N = 4


def solve_puzzle(puzzle: dict) -> dict[str, str] | None:
    """z3でパズルを解いて解答辞書を返す.

    Args:
        puzzle: persons と statements を含むパズル辞書

    Returns:
        {人名: "knight" | "knave"} の辞書、解なしの場合は None
    """
    solver, is_knight = build_solver(puzzle)

    if solver.check() != sat:
        return None

    model = solver.model()
    return {p: ("knight" if z3_is_true(model.eval(is_knight[p])) else "knave") for p in puzzle["persons"]}


def to_text_list(statements: list[dict]) -> list[str]:
    """発言リストをテキストリストに変換する.

    Args:
        statements: 発言辞書のリスト

    Returns:
        テキスト化された発言のリスト
    """
    return [to_text(s) for s in statements]


def _is_type_valid(t: str, sp_is_knight: bool, tg_is_knight: bool) -> bool:
    """発言タイプが解と整合するか確認する."""
    if t == "accusation":
        return sp_is_knight != tg_is_knight
    elif t == "affirmation":
        return sp_is_knight == tg_is_knight
    elif t == "sympathetic":
        return tg_is_knight
    elif t == "antithetic":
        return not tg_is_knight
    elif t == "disjoint":
        return sp_is_knight  # 騎士のみ
    elif t == "joint":
        return not sp_is_knight  # 悪党のみ
    return False


def _make_counting_statement(
    t: str,
    speaker: str,
    sp_is_knight: bool,
    knight_count: int,
    num_persons: int,
) -> dict | None:
    """カウント系発言dictを生成する. 生成不可の場合は None を返す."""
    knave_count = num_persons - knight_count
    visible_cap = min(MAX_VISIBLE_COUNT_N, num_persons)
    stmt: dict = {"type": t, "speaker": speaker}

    if t == "at_least_knight":
        if sp_is_knight:
            if knight_count == 0:
                return None  # 「少なくとも0人」は自明で使わない
            candidates = list(range(1, min(knight_count, visible_cap) + 1))
        else:
            candidates = list(range(knight_count + 1, visible_cap + 1))
        if not candidates:
            return None
        stmt["n"] = random.choice(candidates)
    elif t == "majority_knight":
        claim_true = knight_count * 2 > num_persons
        if sp_is_knight != claim_true:
            return None
    elif t == "at_least_knave":
        if sp_is_knight:
            if knave_count == 0:
                return None
            candidates = list(range(1, min(knave_count, visible_cap) + 1))
        else:
            candidates = list(range(knave_count + 1, visible_cap + 1))
        if not candidates:
            return None
        stmt["n"] = random.choice(candidates)
    elif t == "majority_knave":
        claim_true = knave_count * 2 > num_persons
        if sp_is_knight != claim_true:
            return None
    elif t == "odd_knight":
        claim_true = knight_count % 2 == 1
        if sp_is_knight != claim_true:
            return None
    elif t == "even_knight":
        claim_true = knight_count % 2 == 0
        if sp_is_knight != claim_true:
            return None
    elif t == "odd_knave":
        claim_true = knave_count % 2 == 1
        if sp_is_knight != claim_true:
            return None
    elif t == "even_knave":
        claim_true = knave_count % 2 == 0
        if sp_is_knight != claim_true:
            return None

    return stmt


def _make_two_target_statement(
    t: str,
    speaker: str,
    target1: str,
    target2: str,
    sp_is_knight: bool,
    tg1_is_knight: bool,
    tg2_is_knight: bool,
) -> dict | None:
    """2ターゲット型発言dictを生成する. 生成不可の場合は None を返す."""
    stmt: dict = {"type": t, "speaker": speaker, "target": target1, "target2": target2}

    if t == "pair_same":
        claim_true = tg1_is_knight == tg2_is_knight
        if sp_is_knight != claim_true:
            return None
    elif t == "pair_diff":
        claim_true = tg1_is_knight != tg2_is_knight
        if sp_is_knight != claim_true:
            return None
    elif t == "if_then":
        combos = [(ck, ek) for ck in [True, False] for ek in [True, False]]
        random.shuffle(combos)
        for ck, ek in combos:
            antecedent = tg1_is_knight == ck
            consequent = tg2_is_knight == ek
            claim_true = (not antecedent) or consequent
            if sp_is_knight == claim_true:
                stmt["condition_knight"] = ck
                stmt["conclusion_knight"] = ek
                return stmt
        return None
    elif t == "either_knight":
        claim_true = tg1_is_knight or tg2_is_knight
        if sp_is_knight != claim_true:
            return None
    elif t == "either_knave":
        claim_true = (not tg1_is_knight) or (not tg2_is_knight)
        if sp_is_knight != claim_true:
            return None

    return stmt


def _make_statement(t: str, speaker: str, target: str, tg_is_knight: bool) -> dict:
    """発言dictを生成する."""
    stmt: dict = {"type": t, "speaker": speaker, "target": target}
    if t == "disjoint":
        stmt["claimed_is_knight"] = tg_is_knight  # 真実を述べる
    elif t == "joint":
        stmt["claimed_is_knight"] = not tg_is_knight  # 嘘をつく
    return stmt


def _generate_statements(
    persons: list[str],
    solution: dict[str, str],
    types: list[str],
    num_statements: int,
) -> list[dict]:
    """解と整合する発言リストを生成する."""
    speakers = list(persons)
    extra = num_statements - len(speakers)
    for _ in range(extra):
        speakers.append(random.choice(persons))
    random.shuffle(speakers)

    knight_count = sum(1 for p in persons if solution[p] == "knight")
    num_persons = len(persons)

    majority_pair_used = False
    at_least_knight_used = False
    at_least_knave_used = False
    parity_group_used = False
    if_then_used = False

    statements = []
    for speaker in speakers:
        targets = [p for p in persons if p != speaker]
        target = random.choice(targets)

        sp_is_knight = solution[speaker] == "knight"
        tg_is_knight = solution[target] == "knight"

        if len(targets) >= 2:
            remaining = [p for p in targets if p != target]
            target2 = random.choice(remaining)
            tg2_is_knight = solution[target2] == "knight"
        else:
            target2 = None
            tg2_is_knight = False

        _parity_types = ("odd_knight", "even_knight", "odd_knave", "even_knave")
        available = [
            t for t in types
            if not (majority_pair_used and t in ("majority_knight", "majority_knave"))
            and not (at_least_knight_used and t == "at_least_knight")
            and not (at_least_knave_used and t == "at_least_knave")
            and not (parity_group_used and t in _parity_types)
            and not (if_then_used and t == "if_then")
            and not (target2 is None and t in _TWO_TARGET_TYPES_SET)
        ]
        random.shuffle(available)

        stmt = None
        for t in available:
            if t in _COUNTING_TYPES_SET:
                stmt = _make_counting_statement(t, speaker, sp_is_knight, knight_count, num_persons)
                if stmt is not None:
                    break
                stmt = None
            elif t in _TWO_TARGET_TYPES_SET:
                stmt = _make_two_target_statement(t, speaker, target, target2, sp_is_knight, tg_is_knight, tg2_is_knight)
                if stmt is not None:
                    break
                stmt = None
            elif _is_type_valid(t, sp_is_knight, tg_is_knight):
                stmt = _make_statement(t, speaker, target, tg_is_knight)
                break

        if stmt is None:
            # フォールバック: accusation / affirmation は必ず成立する
            t = "accusation" if sp_is_knight != tg_is_knight else "affirmation"
            stmt = _make_statement(t, speaker, target, tg_is_knight)

        used_type = stmt["type"]
        if used_type in ("majority_knight", "majority_knave"):
            majority_pair_used = True
        elif used_type == "at_least_knight":
            at_least_knight_used = True
        elif used_type == "at_least_knave":
            at_least_knave_used = True
        elif used_type in ("odd_knight", "even_knight", "odd_knave", "even_knave"):
            parity_group_used = True
        elif used_type == "if_then":
            if_then_used = True

        statements.append(stmt)

    return statements


def _is_unique_solution(puzzle: dict) -> bool:
    """z3でパズルの解が一意かどうか確認する."""
    persons = puzzle["persons"]
    solution = puzzle["solution"]

    is_knight = {p: Bool(f"is_knight_{p}") for p in persons}
    import z3
    solver_inst = z3.Solver()

    for stmt in puzzle["statements"]:
        add_constraint(solver_inst, stmt, is_knight)

    # 既知解の否定を追加 → 別解が存在しなければ unsat
    known = And([is_knight[p] if solution[p] == "knight" else Not(is_knight[p]) for p in persons])
    solver_inst.add(Not(known))

    return solver_inst.check() != sat


def generate_puzzle(num_persons: int, level: str, max_retries: int = 1000) -> dict:
    """パズルを生成して返す.

    Args:
        num_persons: 人数（2以上）
        level: 難易度（"easy" | "normal" | "hard"）
        max_retries: 最大リトライ回数

    Returns:
        生成されたパズル辞書（version, level, num_persons, persons, statements, solution を含む）

    Raises:
        ValueError: num_persons が 2 未満、または level が不正な場合
        RuntimeError: リトライ上限を超えた場合
    """
    if num_persons < 2:
        raise ValueError("num_persons は 2 以上が必要です")

    persons = [f"{chr(ord('A') + i)}さん" for i in range(num_persons)]

    if level == "easy":
        types = STATEMENT_TYPES_EASY
        num_statements = num_persons
    elif level == "normal":
        types = STATEMENT_TYPES_ALL
        num_statements = num_persons
    elif level == "hard":
        types = STATEMENT_TYPES_ALL
        num_statements = num_persons + num_persons // 2
    else:
        raise ValueError(f"不明な難易度: {level}")

    for _ in range(max_retries):
        solution = {p: random.choice(["knight", "knave"]) for p in persons}
        statements = _generate_statements(persons, solution, types, num_statements)

        puzzle = {
            "version": "1.0",
            "level": level,
            "num_persons": num_persons,
            "persons": persons,
            "statements": statements,
            "solution": solution,
        }

        if _is_unique_solution(puzzle):
            return puzzle

    raise RuntimeError(f"{max_retries} 回リトライしても一意なパズルを生成できませんでした")

---
File: app/services/puzzle_lib.py
---
"""Knights and Knaves パズルの共通ロジック."""

from z3 import Bool, BoolVal, If, Implies, Not, Or, Solver, Sum

STATEMENT_TYPES_EASY = ["accusation", "affirmation", "sympathetic", "antithetic"]
STATEMENT_TYPES_ALL = [
    "accusation", "affirmation", "sympathetic", "antithetic",
    "disjoint", "joint",
    "at_least_knight", "majority_knight", "at_least_knave", "majority_knave",
    "odd_knight", "even_knight", "odd_knave", "even_knave",
    "pair_same", "pair_diff", "if_then", "either_knight", "either_knave",
]
STATEMENT_TYPES_COUNTING = ["at_least_knight", "majority_knight", "at_least_knave", "majority_knave", "odd_knight", "even_knight", "odd_knave", "even_knave"]
STATEMENT_TYPES_TWO_TARGET = ["pair_same", "pair_diff", "if_then", "either_knight", "either_knave"]

_COUNTING_TYPES_SET = set(STATEMENT_TYPES_COUNTING)
_TWO_TARGET_TYPES_SET = set(STATEMENT_TYPES_TWO_TARGET)


def to_text(stmt: dict) -> str:
    """発言をテキストに変換する."""
    sp = stmt["speaker"]
    tg = stmt.get("target")
    t = stmt["type"]

    if t == "accusation":
        return f"{sp}は言った: 「{tg}は悪党だ。」"
    elif t == "affirmation":
        return f"{sp}は言った: 「{tg}は騎士だ。」"
    elif t == "sympathetic":
        return f"{sp}は言った: 「{tg}は私と同じ種類だ。」"
    elif t == "antithetic":
        return f"{sp}は言った: 「{tg}は私と異なる種類だ。」"
    elif t == "disjoint":
        ck = stmt["claimed_is_knight"]
        kind = "騎士" if ck else "悪党"
        return f"{sp}は言った: 「{tg}は{kind}だ、または私は悪党だ。」"
    elif t == "joint":
        ck = stmt["claimed_is_knight"]
        kind = "騎士" if ck else "悪党"
        return f"{sp}は言った: 「{tg}は{kind}だ、そして私は悪党だ。」"
    elif t == "at_least_knight":
        n = stmt["n"]
        return f"{sp}は言った: 「この中に騎士は少なくとも{n}人いる。」"
    elif t == "majority_knight":
        return f"{sp}は言った: 「騎士のほうが多い。」"
    elif t == "at_least_knave":
        n = stmt["n"]
        return f"{sp}は言った: 「この中に悪党は少なくとも{n}人いる。」"
    elif t == "majority_knave":
        return f"{sp}は言った: 「悪党のほうが多い。」"
    elif t == "odd_knight":
        return f"{sp}は言った: 「この中に騎士は奇数人いる。」"
    elif t == "even_knight":
        return f"{sp}は言った: 「この中に騎士は偶数人いる。」"
    elif t == "odd_knave":
        return f"{sp}は言った: 「この中に悪党は奇数人いる。」"
    elif t == "even_knave":
        return f"{sp}は言った: 「この中に悪党は偶数人いる。」"
    elif t == "pair_same":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}と{tg2}は同じ種類だ。」"
    elif t == "pair_diff":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}と{tg2}は異なる種類だ。」"
    elif t == "if_then":
        tg2 = stmt["target2"]
        ck = stmt["condition_knight"]
        ek = stmt["conclusion_knight"]
        cond_kind = "騎士" if ck else "悪党"
        conc_kind = "騎士" if ek else "悪党"
        return f"{sp}は言った: 「もし{tg}が{cond_kind}なら、{tg2}は{conc_kind}だ。」"
    elif t == "either_knight":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}か{tg2}の少なくとも一方は騎士だ。」"
    elif t == "either_knave":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}か{tg2}の少なくとも一方は悪党だ。」"
    else:
        raise ValueError(f"未知のタイプ: {t}")


def add_constraint(solver: Solver, stmt: dict, is_knight: dict) -> None:
    """z3ソルバーに発言の制約を追加する.

    各発言タイプの制約:
    - accusation:  speaker と target は異なる種類
    - affirmation: speaker と target は同じ種類
    - sympathetic: target は騎士（speaker の種類に関わらず成立）
    - antithetic:  target は悪党（speaker の種類に関わらず成立）
    - disjoint:    speaker は騎士、target は claimed_is_knight の通り
    - joint:       speaker は悪党、target は claimed_is_knight と逆
    """
    sp = stmt["speaker"]
    tg = stmt.get("target")
    t = stmt["type"]

    if t == "accusation":
        solver.add(is_knight[sp] != is_knight[tg])
    elif t == "affirmation":
        solver.add(is_knight[sp] == is_knight[tg])
    elif t == "sympathetic":
        solver.add(is_knight[tg])
    elif t == "antithetic":
        solver.add(Not(is_knight[tg]))
    elif t == "disjoint":
        ck = stmt["claimed_is_knight"]
        solver.add(is_knight[sp])
        solver.add(is_knight[tg] == BoolVal(ck))
    elif t == "joint":
        ck = stmt["claimed_is_knight"]
        solver.add(Not(is_knight[sp]))
        solver.add(is_knight[tg] == BoolVal(not ck))
    elif t in _COUNTING_TYPES_SET:
        num_persons = len(is_knight)
        knight_count = Sum([If(is_knight[p], 1, 0) for p in is_knight])
        knave_count = num_persons - knight_count
        if t == "at_least_knight":
            n = stmt["n"]
            solver.add(is_knight[sp] == (knight_count >= n))
        elif t == "majority_knight":
            solver.add(is_knight[sp] == (knight_count * 2 > num_persons))
        elif t == "at_least_knave":
            n = stmt["n"]
            solver.add(is_knight[sp] == (knave_count >= n))
        elif t == "majority_knave":
            solver.add(is_knight[sp] == (knave_count * 2 > num_persons))
        elif t == "odd_knight":
            solver.add(is_knight[sp] == (knight_count % 2 == 1))
        elif t == "even_knight":
            solver.add(is_knight[sp] == (knight_count % 2 == 0))
        elif t == "odd_knave":
            solver.add(is_knight[sp] == (knave_count % 2 == 1))
        elif t == "even_knave":
            solver.add(is_knight[sp] == (knave_count % 2 == 0))
    elif t in _TWO_TARGET_TYPES_SET:
        tg2 = stmt["target2"]
        if t == "pair_same":
            solver.add(is_knight[sp] == (is_knight[tg] == is_knight[tg2]))
        elif t == "pair_diff":
            solver.add(is_knight[sp] == (is_knight[tg] != is_knight[tg2]))
        elif t == "if_then":
            ck = stmt["condition_knight"]
            ek = stmt["conclusion_knight"]
            solver.add(is_knight[sp] == Implies(is_knight[tg] == BoolVal(ck), is_knight[tg2] == BoolVal(ek)))
        elif t == "either_knight":
            solver.add(is_knight[sp] == Or(is_knight[tg], is_knight[tg2]))
        elif t == "either_knave":
            solver.add(is_knight[sp] == Or(Not(is_knight[tg]), Not(is_knight[tg2])))
    else:
        raise ValueError(f"未知のタイプ: {t}")


def build_solver(puzzle: dict) -> tuple[Solver, dict]:
    """パズルからz3ソルバーと変数辞書を構築する."""
    persons = puzzle["persons"]
    is_knight = {p: Bool(f"is_knight_{p}") for p in persons}
    solver = Solver()
    for stmt in puzzle["statements"]:
        add_constraint(solver, stmt, is_knight)
    return solver, is_knight

---
File: app/services/roleplay_repository.py
---
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, isna

BASE_DIR: Path = Path(__file__).resolve().parents[2]
PARQUET_PATH: Path = BASE_DIR / "hoge.parquet"


def _load_dataframe() -> DataFrame:
    """hoge.parquet を読み込み、DataFrame を返します。"""
    if not PARQUET_PATH.is_file():
        msg = f"hoge.parquet が見つかりません: {PARQUET_PATH}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(PARQUET_PATH)


ROLEPLAY_DF: DataFrame = _load_dataframe()


def _normalize_value(value: Any) -> str | None:
    """NaN を None に変換し、文字列として返します。"""
    if isna(value):
        return None
    return str(value)


def get_random_roleplay_record() -> dict[str, str | None]:
    """ロールプレイテンプレートを 1 件ランダムに取得します。"""
    if ROLEPLAY_DF.empty:
        msg = "hoge.parquet が空のため、ロールプレイデータを取得できません。"
        raise RuntimeError(msg)

    row = ROLEPLAY_DF.sample(n=1).iloc[0]

    return {
        "major_genre": _normalize_value(row.get("major_genre")),
        "minor_genre": _normalize_value(row.get("minor_genre")),
        "world_setting": _normalize_value(row.get("world_setting")),
        "scene_setting": _normalize_value(row.get("scene_setting")),
        "user_setting": _normalize_value(row.get("user_setting")),
        "assistant_setting": _normalize_value(row.get("assistant_setting")),
    }


---
File: app/services/__init__.py
---
"""アプリケーション内部で利用するサービスモジュール群。"""


---
File: app/services/github_html2md.py
---
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlsplit

from github import Auth, Github, GithubException

EM_SPACE: Final[str] = "\u2003"
_DETAILS_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"<details\b[^>]*>.*?</details>",
    re.IGNORECASE | re.DOTALL,
)


class GithubMarkdownError(Exception):
    """GitHub から Markdown を生成する際の例外。"""


@dataclass(frozen=True)
class GithubResource:
    """GitHub 上の issue / discussion を表す情報。"""

    owner: str
    repo: str | None
    kind: str  # "issues" or "discussions"
    is_org_level: bool
    number: int


def convert_github_url_to_markdown(url: str) -> str:
    """GitHub の issue / discussion URL から Markdown を生成する。

    Args:
        url: GitHub の issue または discussion の URL。

    Returns:
        生成された Markdown 文字列。

    Raises:
        GithubMarkdownError: URL 形式や GitHub API 呼び出しに問題があった場合。
    """
    resource = _parse_github_url(url)
    token = os.getenv("GITHUB_ACCESS_TOKEN")
    if not token:
        msg = "環境変数 GITHUB_ACCESS_TOKEN が設定されていません。"
        raise GithubMarkdownError(msg)

    client = Github(auth=Auth.Token(token))

    try:
        if resource.kind == "issues":
            if resource.is_org_level:
                msg = "Organization レベルの issue URL には対応していません。"
                raise GithubMarkdownError(msg)
            if not resource.repo:
                msg = "Issue のリポジトリ情報が不足しています。"
                raise GithubMarkdownError(msg)

            repo = client.get_repo(f"{resource.owner}/{resource.repo}")
            return _render_issue_markdown(repo, resource.number)
        if resource.kind == "discussions":
            if resource.is_org_level:
                return _render_org_discussion_markdown(client, resource.owner, resource.number)
            if not resource.repo:
                msg = "Discussion のリポジトリ情報が不足しています。"
                raise GithubMarkdownError(msg)

            repo = client.get_repo(f"{resource.owner}/{resource.repo}")
            return _render_discussion_markdown(repo, resource.number)
    except GithubException as exc:
        msg = "GitHub API 呼び出し中にエラーが発生しました。"
        raise GithubMarkdownError(msg) from exc

    msg = "issue と discussions の URL のみサポートしています。"
    raise GithubMarkdownError(msg)


def _parse_github_url(url: str) -> GithubResource:
    """GitHub issue / discussion の URL をパースする。

    Args:
        url: GitHub の issue または discussion の URL。

    Returns:
        分解された GitHubResource。

    Raises:
        GithubMarkdownError: GitHub 以外の URL や、対応外のパス形式の場合。
    """
    parsed = urlsplit(url)
    host = parsed.netloc.lower()
    if not (host == "github.com" or host.endswith(".github.com")):
        msg = "GitHub の URL ではありません。"
        raise GithubMarkdownError(msg)

    segments = [seg for seg in parsed.path.split("/") if seg]
    if len(segments) < 4:
        msg = "GitHub の issue / discussion の URL 形式が不正です。"
        raise GithubMarkdownError(msg)

    # /owner/repo/issues/123 や /owner/repo/discussions/123 形式
    if segments[0] != "orgs":
        owner, repo, kind, number_str = segments[:4]
        is_org_level = False
    else:
        # /orgs/ORG/discussions/123 形式（Organization レベルの Discussion）
        if len(segments) < 4:
            msg = "GitHub の issue / discussion の URL 形式が不正です。"
            raise GithubMarkdownError(msg)
        _, owner, kind, number_str = segments[:4]
        repo = None
        is_org_level = True

    if kind not in {"issues", "discussions"}:
        msg = "issue と discussions の URL のみサポートしています。"
        raise GithubMarkdownError(msg)

    try:
        number = int(number_str)
    except ValueError as exc:
        msg = "issue / discussion 番号が数値ではありません。"
        raise GithubMarkdownError(msg) from exc

    return GithubResource(owner=owner, repo=repo, kind=kind, is_org_level=is_org_level, number=number)


def _format_datetime_utc(dt: datetime) -> str:
    """日時を UTC に変換して sample.md と同形式の文字列にする。"""
    if dt.tzinfo is None:
        utc_dt = dt.replace(tzinfo=UTC)
    else:
        utc_dt = dt.astimezone(UTC)
    return utc_dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_issue_header_line(author_login: str, created_at: datetime, state: str) -> str:
    """Issue / discussion 本文のヘッダー行を整形する。"""
    created_str = _format_datetime_utc(created_at)
    # 例: "@tommhuth \u2003 Created at: 2025-12-12 13:03:48 UTC  \u2003 OPEN"
    return f"@{author_login} {EM_SPACE} Created at: {created_str} UTC  {EM_SPACE} {state}"


def _format_comment_header_line(index: int, author_login: str, created_at: datetime) -> str:
    """コメントのヘッダー行を整形する。"""
    created_str = _format_datetime_utc(created_at)
    # 例: "## Comment 1, @sing-deep \u2003 at: 2025-12-12 15:08:17 UTC"
    return f"## Comment {index}, @{author_login} {EM_SPACE} at: {created_str} UTC"


def _remove_details_sections(markdown: str) -> str:
    """Markdown テキストから <details> ブロックを削除する。

    GitHub 上で頻出する `<details><summary>Details</summary>...</details>` のような
    折りたたみセクションを正規表現ベースで取り除く。
    """
    if not markdown:
        return markdown
    return _DETAILS_BLOCK_RE.sub("", markdown)


def _render_issue_markdown(repo: object, number: int) -> str:
    """Issue を取得して sample.md に近い形式の Markdown を生成する。"""
    issue = repo.get_issue(number=number)

    author_login = issue.user.login if getattr(issue, "user", None) is not None else "unknown"
    state = (issue.state or "").upper() or "UNKNOWN"

    lines: list[str] = []
    lines.append(f"# {issue.title}")
    lines.append("")
    lines.append(_format_issue_header_line(author_login, issue.created_at, state))
    lines.append("")
    lines.append("")

    body = _remove_details_sections(issue.body or "")
    if body:
        lines.extend(body.rstrip("\n").splitlines())
        lines.append("")

    comments = issue.get_comments()
    for idx, comment in enumerate(comments, start=1):
        comment_author = comment.user.login if getattr(comment, "user", None) is not None else "unknown"

        lines.append("---")
        lines.append("")
        lines.append(_format_comment_header_line(idx, comment_author, comment.created_at))
        lines.append("")
        lines.append("")

        comment_body = _remove_details_sections(comment.body or "")
        if comment_body:
            lines.extend(comment_body.rstrip("\n").splitlines())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_discussion_markdown(repo: object, number: int) -> str:
    """Discussion を取得して sample.md に近い形式の Markdown を生成する。"""
    discussion_graphql_schema = """
id
title
body
createdAt
url
author { login }
"""
    comment_graphql_schema = """
id
body
createdAt
url
author { login }
"""

    # discussion / comments ともに GraphQL で必要なフィールドだけ取得する
    discussion = repo.get_discussion(number, discussion_graphql_schema)
    return _format_discussion_markdown_with_comments(discussion, comment_graphql_schema)


def _render_org_discussion_markdown(client: Github, org_login: str, number: int) -> str:
    """Organization レベルの Discussion を取得して Markdown を生成する。

    GitHub GraphQL には ``organization.discussion`` フィールドが存在しないため、
    Discussions 検索（``search(type: DISCUSSION)``）で該当ノードを特定してから
    ``get_repository_discussion`` で本体を取得する。
    """
    node_id = _search_discussion_node_id_for_org(client, org_login, number)

    discussion_graphql_schema = """
id
title
body
createdAt
url
author { login }
"""
    comment_graphql_schema = """
id
body
createdAt
url
author { login }
"""

    # Discussion の node_id が分かれば、PyGithub のヘルパーで
    # RepositoryDiscussion オブジェクトを取得できる。
    discussion = client.get_repository_discussion(node_id, discussion_graphql_schema)
    return _format_discussion_markdown_with_comments(discussion, comment_graphql_schema)


def _search_discussion_node_id_for_org(client: Github, org_login: str, number: int) -> str:
    """Organization レベル Discussion の node_id を Discussions 検索で取得する。

    GraphQL の search フィールドを使い、org 単位で Discussions を検索したうえで
    Discussion.number が一致するノードを 1 件特定する。
    """
    search_query = f"org:{org_login} {number}"
    graphql_query = """
query Q($query: String!) {
  search(query: $query, type: DISCUSSION, first: 50) {
    discussionCount
    nodes {
      __typename
      ... on Discussion {
        id
        number
      }
    }
  }
}
"""
    variables: dict[str, Any] = {"query": search_query}
    _, data = client.requester.graphql_query(graphql_query, variables)

    search_data = data.get("data", {}).get("search")
    if not isinstance(search_data, dict):
        msg = "Organization レベルの discussion 検索結果を解釈できませんでした。"
        raise GithubMarkdownError(msg)

    nodes = search_data.get("nodes") or []
    if not isinstance(nodes, list):
        msg = "Organization レベルの discussion 検索結果の形式が不正です。"
        raise GithubMarkdownError(msg)

    candidates: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("__typename") != "Discussion":
            continue
        if int(node.get("number", -1)) != number:
            continue
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id:
            candidates.append(node_id)

    if not candidates:
        msg = f"Organization レベルの discussion (org={org_login}, number={number}) を GitHub 検索で特定できませんでした。"
        raise GithubMarkdownError(msg)
    if len(candidates) > 1:
        msg = f"Organization レベルの discussion (org={org_login}, number={number}) に複数の候補が見つかりました。"
        raise GithubMarkdownError(msg)

    return candidates[0]


def _format_discussion_markdown_with_comments(discussion: object, comment_graphql_schema: str) -> str:
    """Discussion 本文とコメントを Markdown 文字列に整形する。"""
    author = discussion.author
    author_login = getattr(author, "login", None) or "unknown"

    # Discussion に state フィールドは無いので固定のラベルを付与する
    state_label = "DISCUSSION"

    lines: list[str] = []
    lines.append(f"# {discussion.title}")
    lines.append("")
    lines.append(_format_issue_header_line(author_login, discussion.created_at, state_label))
    lines.append("")
    lines.append("")

    body = _remove_details_sections(discussion.body or "")
    if body:
        lines.extend(body.rstrip("\n").splitlines())
        lines.append("")

    comments = discussion.get_comments(comment_graphql_schema)
    for idx, comment in enumerate(comments, start=1):
        comment_author = comment.author
        comment_author_login = getattr(comment_author, "login", None) or "unknown"

        lines.append("---")
        lines.append("")
        lines.append(_format_comment_header_line(idx, comment_author_login, comment.created_at))
        lines.append("")
        lines.append("")

        comment_body = _remove_details_sections(comment.body or "")
        if comment_body:
            lines.extend(comment_body.rstrip("\n").splitlines())
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"

---
File: pyproject.toml
---
[project]
name = "fastapi-template"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "fastapi[standard]",
    "ruff",
    "pytest",
    "trafilatura[all]>=2.0.0",
    "httpx>=0.28.1",
    "autopager>=0.3.1",
    "ipython>=9.6.0",
    "curl-cffi>=0.13.0",
    "httpx-curl-cffi>=0.1.4",
    "newspaper4k>=0.9.4.1",
    "tinysegmenter>=0.4",
    "chardet>=5.2.0",
    "pygithub>=2.8.1",
    "pandas>=2.3.3",
    "pyarrow>=22.0.0",
    "openpyxl>=3.1.5",
    "z3-solver>=4.16.0.0",
]

[tool.ruff]
line-length = 300
target-version = "py313"
exclude = [".git", ".ruff_cache", ".venv", ".vscode"]

[tool.ruff.lint]
select = [
    "ANN", # 型アノテーション関連
    "B",   # flake8-bugbearルール
    "D",   # pydocstyleルール
    "E",   # pycodestyleエラー
    "F",   # pyflakesルール
    "I",   # isort互換ルール
    "PTH", # os.pathではなくpathlib.Pathの利用を促す
    "RUF", # Ruff固有ルール
    "SIM", # flake8-simplifyルール
    "UP",  # pyupgrade
    "W",   # pycodestyle警告
]
ignore = [
    "RUF001", # 文字列内の曖昧なUnicode文字を許容
    "ANN401", # 関数引数の型アノテーションがAnyでも許容
    "B007",   # ループ変数の未使用を許容
    "B008",   # デフォルト引数での関数呼び出しを許容
    "B905",   # strict=Trueなしのzip()使用を許容
    "COM812", # カンマの付け忘れを許容
    "COM819", # カンマ禁止違反を許容
    "D1",     # 公開モジュール・クラス・関数・メソッドのdocstring省略を許容
    "D203",   # クラスdocstring前の空行数（GoogleスタイルではD211優先のため無視）
    "D205",   # docstringの要約行と説明の間の空行数を無視
    "D212",   # 複数行docstringの要約行の位置（1行目開始）を無視
    "D213",   # 複数行docstringの要約行の位置（2行目開始）を無視
    "D400",   # docstringの1行目の末尾ピリオドを無視
    "D415",   # docstringの1行目の末尾句読点（ピリオド等）を無視
    "E114",   # コメント行のインデントが4の倍数でない場合を許容
    "G004",   # ログ出力でのf-string使用を許容
    "ISC001", # 1行での暗黙的な文字列連結を許容
    "ISC002", # 複数行での暗黙的な文字列連結を許容
    "PTH123", # open()のPath.open()置き換えを強制しない
    # "Q000",   # シングルクォート使用を許容（ダブルクォート推奨違反）
    "Q001",   # 複数行文字列でのシングルクォート使用を許容
    "Q002",   # docstringでのシングルクォート使用を許容
    "RUF002", # docstring内の曖昧なUnicode文字を許容
    "RUF003", # コメント内の曖昧なUnicode文字を許容
    "SIM105", # try-except-passをcontextlib.suppressで置き換えなくても許容
    "SIM108", # if-elseブロックを三項演算子にしなくても許容
    "SIM116", # 連続したif文を辞書にしなくても許容
]
unfixable = [
    "F401", # 未使用インポート
    "F841", # 未使用変数
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]

[tool.ruff.lint.pydocstyle]
convention = "google"

