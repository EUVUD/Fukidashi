# Fukidashi

多 agent 漫画翻译编辑器（日 → 简体中文）。设计见 [OUTLINE.md](OUTLINE.md)，范围见 [SCOPE.md](SCOPE.md)。

## Setup

```bash
uv sync
git clone --depth 1 https://github.com/mantra-inc/open-mantra-dataset.git data/open-mantra-dataset
cp .env.example .env   # 填 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
```

OpenMantra 是 CC BY-NC 4.0，图片不进本仓库。数据笔记见 [docs/openmantra.md](docs/openmantra.md)。

## Run the app（M1 vertical slice）

```bash
uv run python scripts/import_openmantra.py          # 214 页导入 runs/fukidashi.db，可重复运行
uv run uvicorn app.main:app --app-dir backend --port 8000
npm install --prefix frontend                       # 第一次
npm run dev --prefix frontend                       # http://localhost:5173/tojime_no_siora/002
```

在浏览器里选一页，点 Translate：后端对这一页跑一次 `baseline`（一次调用，不用 tools 和 KB），把 run 和 zh-Hans 译文存进 SQLite。Translate 会真实调用 `.env` 里配置的模型，约 $0.0006 / 页。DB 路径可以用环境变量 `FUKIDASHI_DB` 改。记录见 [docs/m1.md](docs/m1.md)。

## M0 first call

```bash
uv run python scripts/m0_first_call.py --dry-run   # 只看请求内容，不发送
uv run python scripts/m0_first_call.py             # 真实调用一页，结果写到 runs/m0/
uv run pytest
```

每次调用的 token 用量追加到 `runs/usage.jsonl`；provider 没返回 usage 的调用单独计数，不记为 0。
