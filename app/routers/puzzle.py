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
