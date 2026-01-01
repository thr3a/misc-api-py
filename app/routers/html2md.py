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
