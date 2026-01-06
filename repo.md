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
│   │   └── roleplay.py
│   └── services
│       ├── __init__.py
│       ├── github_html2md.py
│       └── roleplay_repository.py
└── pyproject.toml

```

---
File: app/main.py
---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import html2md, items, roleplay

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
from fastapi import APIRouter, HTTPException, Path
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
async def read_item(item_id: str = Path(..., description="取得するアイテムID", example="plumbus")) -> ItemWithId:
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
async def update_item(item_id: str = Path(..., description="更新するアイテムID", example="plumbus")) -> ItemWithId:
    """アイテムを更新します（デモ用の制限あり）"""
    if item_id != "plumbus":
        raise HTTPException(status_code=403, detail="You can only update the item: plumbus")
    return ItemWithId(item_id=item_id, name="The great Plumbus")

---
File: app/routers/__init__.py
---

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

