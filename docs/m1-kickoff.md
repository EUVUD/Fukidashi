# M1 Kickoff Prompt

你在 Fukidashi 仓库里开始 **M1 vertical slice**。仓库：`/Users/junninghu/Documents/Fukidashi`（GitHub 私有仓库 `EUVUD/Fukidashi`；`main` 上只有 M0 的 commit `f8395f2`）。

## 先读

- `OUTLINE.md`：§0 Terminology（术语一律用这里的英文写法）、§1.3 Design Principles、§3 Architecture、§7 的 M1 行、§8 已定的 decisions。
- `SCOPE.md`：§2 Never cut（vertical slice 在这里）。
- `docs/openmantra.md`（数据字段和坑）、`docs/m0.md`（第一次真实调用的结果）。
- M0 已有代码：
  - `backend/app/datasets/openmantra.py`：OpenMantra loader
  - `backend/app/llm/client.py` + `metering.py`：OpenAI-compatible client；unknown usage 不记 0；per-page token budget；账本 `runs/usage.jsonl`
  - `scripts/m0_first_call.py`：baseline 的 prompt、JSON 解析、解析失败重试一次

## 目标（10/25 checkpoint）

**Done when：** 在浏览器里打开 `tojime_no_siora/002`，点 Translate，看到 `deepseek-flash` 返回的真实 zh-Hans 译文，每条译文都和图上的 oracle box 对得上。

## M1-a：Backend + 数据（约 1 h）

- FastAPI + SQLite。用标准库 `sqlite3` 和一个 `schema.sql`，先不引 ORM。
- 表：`books`、`pages`、`segments`（oracle box、`text_ja`、zh-Hant 和 en 参考译文）、`runs`（config、model、usage、cost、status、时间）、`translations`（run → segment → zh-Hans）。
- `scripts/import_openmantra.py`：把 214 页导入 SQLite，重复运行不产生重复行。DB 文件放在 gitignored 的位置。
- API：
  - `GET /api/pages`：页面列表（`book_title`、`page_index`、segment 数）
  - `GET /api/pages/{book_title}/{page_index}`：segments + 最新一次 run 的译文
  - `GET /api/pages/{book_title}/{page_index}/image`：从 `data/` 读 jpg
  - `POST /api/pages/{book_title}/{page_index}/translate`：跑 `baseline`，存 run 和译文，返回结果

## M1-b：Baseline + 最简 frontend（约 2.5 h）

- 把 baseline 逻辑从 `scripts/m0_first_call.py` 挪进 backend（例如 `backend/app/pipeline/baseline.py`），script 和 API 共用；M0 script 改完要还能跑。
- 所有 LLM 调用都走 `LLMClient`，不能绕过 metering。
- Frontend：`frontend/` 下用 Vite + React + TypeScript。页面列表 → 单页视图：
  - 页面图片 + 绝对定位的 box overlay，按图片实际显示的尺寸缩放
  - 一列 segment：ja / zh-Hans output / zh-Hant reference
  - Translate 按钮，有 loading 和错误状态
  - dev server 代理到 FastAPI
- 写 `.claude/launch.json`（backend 和 frontend 两个配置），用内置浏览器预览验证。

## 不做（留给后面的 milestone）

SSE、agent / tools、KB、Reviewer、编辑功能、react-konva、eval harness、OpenCC、Langfuse / Phoenix。

## 约束

- 文档正文中文、术语英文；新术语加进 OUTLINE §0。
- `.env` 里有 API key：不要读取或打印它的值。`.env`、`data/`、`runs/` 都是 gitignored。
- 整个项目费用上限 $100。`.env` 按高峰价记账，baseline 实测约 $0.0006 / 页。测试里不发真实请求（mock LLM）；验证时真实调用只跑 1–2 页。
- DeepSeek：thinking 默认开，可以和 JSON mode 一起用；偶尔返回空 content，已有的重试一次能处理；一页图约 591 input tokens。
- 测试只测会出事的地方：import 重复运行不产生重复行、API smoke test（LLM mocked）。
- 在 feature branch（例如 `m1-vertical-slice`）上工作。做完先给我看 diff 和浏览器截图，我确认后再 commit / push。

## 如果 10/25 前做不完

按 OUTLINE §9 的 pivot rule：先放下 frontend，保证 CLI 路径能跑，编辑器推到 M7。

## 做完后告诉我

- 怎么启动（命令）
- 浏览器里看到的结果（截图）
- 真实调用花了多少（看 `runs/usage.jsonl`）
- 哪些没做、哪些没验证
