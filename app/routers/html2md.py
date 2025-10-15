from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl
from trafilatura import extract, fetch_url


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


async def _convert_html_to_markdown(url: str) -> str:
    """HTML を取得し、Markdown に変換する内部処理"""

    def _extract() -> str:
        downloaded = fetch_url(url)
        if downloaded is None:
            msg = "指定した URL からコンテンツを取得できませんでした。"
            raise HTTPException(status_code=400, detail=msg)

        result = extract(downloaded, output_format="markdown", with_metadata=True)
        if result is None:
            msg = "コンテンツの抽出に失敗しました。"
            raise HTTPException(status_code=400, detail=msg)
        return result

    return await run_in_threadpool(_extract)


@router.get(
    "",
    summary="URL の HTML を Markdown に変換する",
    response_class=PlainTextResponse,
    response_description="変換された Markdown 文字列",
)
async def convert_html2md(query: Html2MarkdownQuery = Depends()) -> PlainTextResponse:
    """指定した URL の HTML を Markdown 形式のテキストとして返します。"""
    try:
        markdown = await _convert_html_to_markdown(str(query.url))
    except HTTPException:
        raise
    except Exception as exc:
        msg = "変換処理中に予期しないエラーが発生しました。"
        raise HTTPException(status_code=500, detail=msg) from exc
    return PlainTextResponse(content=markdown, media_type="text/markdown; charset=utf-8")
