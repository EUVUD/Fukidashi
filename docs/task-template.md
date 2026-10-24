---
id: "NNN"
title: <one line, imperative>
status: proposed
milestone: M?
branch:
created: YYYY-MM-DD
depends_on: []
pr:
---

# Task NNN: <title>

<!-- Architect-owned: everything down to "## Execution report". Executor and reviewer do not edit it. -->

## Goal

Done when: <one sentence someone can observe, e.g. "in the browser, ... shows ...">

## Context

Read first: <OUTLINE sections, docs, and the code paths this task touches. Cite them, do not copy them.>

## Scope

- <what to build or change>

## Out of scope

- <what looks related but must not be done here>

## Constraints

- Real LLM calls: <none | N pages on <book/page>, at most $X>. Tests mock the LLM.
- Do not touch: <files or areas>

## Acceptance criteria

Each one must be checkable by a command or an observable behaviour.

- [ ] AC1: <criterion> (check: <command or observation>)
- [ ] AC2: ...

## Verification

Commands the executor runs and the reviewer re-runs:

```bash
uv run pytest
```

---

## Execution report

<!-- Executor-owned. Add one "### Round N" block per execution:
     Summary and files changed · each acceptance criterion: met / not met + evidence · verification output ·
     deviations from the task · not done / not verified · noticed outside scope · LLM spend · commit SHAs.
     If blocked, say why under "Blocked" and set status back to proposed. -->

## Review

<!-- Reviewer-owned. Add one "### Round N" block per review:
     Verdict · findings by severity (blocker / major / minor / nit), each with file:line, what is wrong,
     how to reproduce, suggested fix · acceptance criteria table with your own evidence ·
     what you verified and what you could not. -->
