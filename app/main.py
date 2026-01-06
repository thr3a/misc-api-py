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
