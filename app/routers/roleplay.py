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

