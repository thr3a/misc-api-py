from fastapi import FastAPI

from .routers import html2md, items, roleplay

app = FastAPI()

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
