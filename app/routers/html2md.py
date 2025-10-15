import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl
from trafilatura import extract


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

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"


async def _convert_html_to_markdown(url: str) -> str:
    """HTML を取得し、Markdown に変換する内部処理"""
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "ja-JP"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            response = await client.get(url, timeout=10)
        except httpx.HTTPError as exc:
            msg = "指定した URL からコンテンツを取得できませんでした。"
            raise HTTPException(status_code=400, detail=msg) from exc
    if response.status_code >= 400:
        msg = f"指定した URL の取得に失敗しました（HTTP {response.status_code}）。"
        raise HTTPException(status_code=400, detail=msg)

    html = response.text
    if not html.strip():
        msg = "取得したコンテンツが空でした。"
        raise HTTPException(status_code=400, detail=msg)

    def _extract() -> str:
        result = extract(html, url=url, output_format="markdown", with_metadata=True)
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
