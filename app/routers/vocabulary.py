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
