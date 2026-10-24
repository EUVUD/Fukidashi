# Fukidashi

多 agent 漫画翻译编辑器（日 → 简体中文）。设计见 [OUTLINE.md](OUTLINE.md)，范围见 [SCOPE.md](SCOPE.md)。

## Setup

```bash
uv sync
git clone --depth 1 https://github.com/mantra-inc/open-mantra-dataset.git data/open-mantra-dataset
cp .env.example .env   # 填 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
```

OpenMantra 是 CC BY-NC 4.0，图片不进本仓库。数据笔记见 [docs/openmantra.md](docs/openmantra.md)。

## M0 first call

```bash
uv run python scripts/m0_first_call.py --dry-run   # 只看请求内容，不发送
uv run python scripts/m0_first_call.py             # 真实调用一页，结果写到 runs/m0/
uv run pytest
```

每次调用的 token 用量追加到 `runs/usage.jsonl`；provider 没返回 usage 的调用单独计数，不记为 0。
