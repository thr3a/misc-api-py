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
