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

