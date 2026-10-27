# Design Diagrams

> 2026-10-24 · Matches the code after M1, once Run handling moved from the router into a service layer. Diagrams are Mermaid and render on GitHub.
> **Solid = implemented; dashed = planned** (milestones in [OUTLINE.md](../OUTLINE.md) §7). Terms follow OUTLINE §0.

Fukidashi translates Japanese manga into Simplified Chinese (zh-Hans). Each page has several Segments (one line of dialogue in one oracle box); the model looks at the whole page image and translates them together, and the user checks the translations against the boxes in the browser.

**How to read this:** diagrams 1 and 2 answer "what is this system and how is it used" in about a minute; diagrams 3 and 4 answer "how is it built", for when you change code.

---

## 1. Domain Model

Only the concepts of the problem domain and their relationships; no code.

```mermaid
classDiagram
    direction LR

    class Book {
        book_title
    }
    class Page {
        page_index
        image
    }
    class Segment {
        oracle box
        text_ja
        reference translations
    }
    class Run {
        config
        model
        status
        tokens and cost
    }
    class Translation {
        text_zh_hans
    }
    class KnowledgeBase:::planned {
        Glossary
        Character Profiles
    }
    class ReviewComment:::planned {
        text
    }

    Book "1" --> "*" Page : contains
    Book "1" --> "0..1" KnowledgeBase : has
    Page "1" --> "*" Segment : has
    Page "1" --> "*" Run : translated by
    Run "1" --> "*" Translation : produces
    Segment "1" <-- "*" Translation : translates
    Translation "1" --> "*" ReviewComment : gets

    classDef planned fill:#f4f4f4,stroke:#999,stroke-dasharray:5 5,color:#777
```

- A page has many **Segments**. Every click on Translate creates a **Run**, and the Run gives each Segment one **Translation**. The page shows its latest successful Run.
- **Reference translations** (OpenMantra's zh-Hant and en translations) belong to the Segment and are used only for comparison and evaluation, never in a prompt.
- `Run.config` is only `baseline` for now (one call per page, no tools, no KB); `full` and `full_no_kb` arrive in M2–M3.
- The dashed **KnowledgeBase** (M3) and **ReviewComment** (M3) do not exist yet.

---

## 2. System Sequence Diagram

The system as a black box: what the user asks for, what comes back, and which external system is involved.

```mermaid
sequenceDiagram
    actor U as User
    participant S as Fukidashi<br/>(system)
    participant M as LLM provider<br/>(deepseek-flash)

    U->>S: openPage(book, page)
    S-->>U: image, Segments (box, ja, reference), latest Translations

    U->>S: translate(book, page)
    S->>M: page image + Japanese text + boxes
    M-->>S: zh-Hans text for every Segment
    alt ok
        S-->>U: Segments with zh-Hans + Run (model, tokens, cost)
    else call failed or output unparseable
        S-->>U: error (earlier Translations stay on screen)
    end
```

---

## 3. Object Interaction Diagram

Inside the system: which objects one `translate` request passes through, and in what order (`translate_page` in `api/pages.py` and `translate_page` in `runs.py`).

```mermaid
sequenceDiagram
    autonumber
    participant V as PageView<br/>(React)
    participant R as pages router<br/>(FastAPI)
    participant U as runs<br/>(service)
    participant D as SQLite<br/>(via store)
    participant B as baseline<br/>(pipeline)
    participant C as LLMClient
    participant G as Meter,<br/>PageBudget
    participant M as LLM provider

    V->>R: POST /api/pages/:book/:index/translate
    R->>U: translate_page(book, index, make_client)
    U->>D: read Page and Segments
    U->>C: make_client() (config from .env, new Meter)
    U->>D: INSERT Run (status running)
    U->>B: run_baseline(page, client, budget)
    B->>B: build_messages (image, ja, boxes only)
    loop up to 2 attempts (retry only if the output is empty or unparseable)
        B->>C: chat(messages, budget)
        C->>G: budget.check()
        C->>M: chat.completions.create
        M-->>C: content + usage
        C->>G: meter.record(), budget.charge()
        C-->>B: LLMResult
        B->>B: parse_translations()
    end
    B-->>U: BaselineResult (translations or error)
    U->>D: UPDATE Run (usage, cost, status), INSERT Translations
    U-->>R: Outcome (run id, result)
    alt ok
        R->>D: read Segments with latest Translations
        R-->>V: 200 PageDetail (Segments, zh-Hans, Run)
    else failed
        R-->>V: 502 error detail
    end
    Note over G: Meter appends every call to runs/usage.jsonl
```

---

## 4. Object Model

The objects in the code and how they depend on each other (the actual M1 structure).

### 4.1 Backend

Colors mark layers: blue = API, pink = service, green = pipeline, orange = LLM, purple = data.

```mermaid
classDiagram
    direction LR

    class PagesRouter:::api {
        list_pages()
        get_page_detail()
        get_page_image()
        translate_page()
    }
    class PageDetail:::api
    class Runs:::service {
        translate_page()
    }
    class Baseline:::pipeline {
        run_baseline()
    }
    class BaselineResult:::pipeline {
        translations
        attempts
        error
    }
    class LLMClient:::llm {
        chat()
    }
    class LLMConfig:::llm
    class Meter:::llm {
        records CallRecords
    }
    class PageBudget:::llm
    class Page:::data
    class Segment:::data
    class Store:::data {
        find_page()
        start_run()
        finish_run()
    }
    class SQLite:::data {
        books
        pages
        segments
        runs
        translations
    }
    class Importer:::data {
        import_openmantra()
    }

    PagesRouter ..> PageDetail : returns
    PagesRouter ..> Runs : translate
    PagesRouter ..> Store : reads
    Runs ..> Store : Runs, Translations
    Runs ..> Baseline : run_baseline
    Runs ..> Page : builds
    Baseline ..> Page : reads
    Baseline ..> LLMClient : chat
    Baseline ..> BaselineResult : returns
    LLMClient *-- LLMConfig
    LLMClient o-- Meter
    LLMClient ..> PageBudget : check, charge
    Page *-- Segment
    Store ..> SQLite : SQL
    Importer ..> Page : loads
    Importer ..> SQLite : fills

    classDef api fill:#e8f0fe,stroke:#4a78d0,color:#222
    classDef service fill:#fde8f1,stroke:#c4527f,color:#222
    classDef pipeline fill:#e6f4ea,stroke:#3a9d5d,color:#222
    classDef llm fill:#fff4e0,stroke:#d08a2a,color:#222
    classDef data fill:#f1e9fb,stroke:#8a5fc4,color:#222
```

- `PagesRouter` only parses parameters, maps errors to status codes and shapes responses; it has no SQL. "Translate a page and record a Run" lives in `Runs.translate_page`, and the SQL lives in `Store`.
- `Runs.translate_page` does not depend on HTTP, so the eval harness can call it directly (OUTLINE §1.3, Same code path). An invalid page raises `PageNotFound` and similar errors, which the router maps to 404 / 422; a failed model call is not an exception but a `failed` Run.
- `PagesRouter` gets the SQLite connection, the dataset path and the `LLMClient` factory through FastAPI's `Depends` (`get_db`, `get_dataset_root`, `get_client_factory`). Tests swap them for a temporary DB and a fake `LLMClient`, so tests never make real requests.
- `Runs` receives an `LLMClient` factory, not a client: every Run builds its own `LLMClient` and `Meter`, so each Run's usage is its own.
- `Baseline` touches neither HTTP nor SQLite.

### 4.2 Frontend

```mermaid
classDiagram
    direction LR

    class App {
        pages
        route
    }
    class PageList
    class PageView {
        detail
        translating
        error
    }
    class api {
        <<module>>
        fetchPages()
        fetchPage()
        translatePage()
    }
    class FastAPI {
        <<backend /api>>
    }

    App *-- PageList
    App *-- PageView
    App ..> api : fetchPages
    PageView ..> api : fetchPage, translatePage
    api ..> FastAPI : HTTP
```

### 4.3 Concept → table → object

| Domain concept | SQLite table | Backend object | API / frontend type |
|---|---|---|---|
| Book | `books` | `book_title` (string) | `PageSummary.book_title` |
| Page | `pages` | `Page` | `PageSummary`, `PageDetail` |
| Segment | `segments` | `Segment` | `SegmentOut` / `Segment` |
| Run | `runs` | No class of its own: `Runs.translate_page` writes the row through `Store.start_run` / `finish_run`, using `Meter.summary()` and `BaselineResult` | `RunOut` / `Run` |
| Translation | `translations` | `BaselineResult.translations` | `Segment.text_zh_hans` |

### 4.4 Where later milestones plug in

- **M2:** an Orchestrator and the Translator Agent (a tool loop; directories `orchestrator/` and `agents/` in OUTLINE §6) join `Baseline`. `LLMClient`, `Meter` and `PageBudget` stay as they are, and every agent call goes through them; `Runs.translate_page` picks Baseline or the Orchestrator by `config`.
- **M3:** the `KB` (Glossary, Character Profiles, Proposal Queue) and the Reviewer / Curator Agents; `runs.config` gains `full` and `full_no_kb`.
- **M7:** the frontend adds a canvas and editing to `PageView`, plus SSE, which adds an event subscription to the `api` module.
