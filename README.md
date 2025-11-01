# repo.md

```
curl https://cdn.jsdelivr.net/gh/thr3a/fastapi-template@repo2txt/repo.md
```

```
APIの可読性と保守性を向上させることを目指します。
app/routers/items.pyで以下のサンプルコードを参考に
OpenAPI (Swagger UI) で表示されるための詳細な日本語のドキュメンテーションを追加してください。

サンプルコード
@/memo.md
```

```
import httpx, autopager; html=httpx.get('https://gendai.media/articles/-/159330?imp=0').text;
print(autopager.urls(html, baseurl='https://gendai.media/articles/-/159330?imp=0'))

['https://gendai.media/articles/-/159330?page=2', 'https://gendai.media/articles/-/159330?imp=0', 'https://gendai.media/articles/-/159330?page=2', 'https://gendai.media/articles/-/159330?page=3', 'https://gendai.media/articles/-/159330?page=4', 'https://gendai.media/articles/-/159330?page=5']
```

