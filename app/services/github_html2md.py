from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final
from urllib.parse import urlsplit

import github
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
