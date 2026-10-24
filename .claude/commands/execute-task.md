---
description: Execute one task as the independent Executor. Implement it on a task branch, verify it, and write the report.
argument-hint: "<task id, e.g. 003>"
---

You are the **Executor** for task $ARGUMENTS (below, NNN means that id). This is a fresh session: you know only what the task file and the repository say. The shared rules (task file ownership, status transitions, git rules) are in CLAUDE.md under "Task workflow"; follow them.

## 1. Pick up the task

- Define `TASKS_DIR` as in CLAUDE.md and find `$TASKS_DIR/NNN-*.md`. Stop if there is no match or more than one.
- Read the whole file, then every file named under Context.
- `status` must be `proposed` or `changes_requested`. If it is `in_progress`, another session may be working on it; if it is `in_review`, it needs `/review-task`. In both cases stop and tell the user.
- For `changes_requested`, read the newest round in `## Review` and fix only what it lists, plus whatever those fixes require.

## 2. Start

- `git status`. If the tree has changes you did not make, stop and ask.
- `git fetch origin`, then `git switch -c task/NNN-slug origin/main` (for a second round, `git switch task/NNN-slug`).
- Set `status: in_progress` and `branch:` in the task file. Use the acceptance criteria as your checklist.

## 3. Implement

- Do exactly what Scope says. Respect Out of scope and Constraints. Match the style of the code around you.
- Do not edit Goal, Context, Scope, Constraints, Acceptance criteria or Verification. If the task is wrong, ambiguous or impossible, stop: write why under "Blocked" in the Execution report, set `status: proposed`, and tell the user to take it back to `/architect`.
- Things you notice that are outside the task go into the report under "Noticed outside scope". Do not fix them and do not edit `IDEAS.md`; the architect triages them.
- Real LLM calls only as Constraints allow. Otherwise mock, as the existing tests do.

## 4. Verify

Run every command under Verification and check each acceptance criterion with evidence you produced yourself. Do not claim anything you did not run.

## 5. Commit

Stage explicit paths only and commit on the task branch with a message that says why. Do not push, open a PR or merge.

## 6. Report

Under `## Execution report` add `### Round N`: summary and files changed; each acceptance criterion as met or not met with its evidence; verification commands and results; deviations from the task and why; not done or not verified; noticed outside scope; LLM spend (the new lines in `runs/usage.jsonl`, or "none"); commit SHAs. Then set `status: in_review`.

## 7. Hand off

Tell the user the task is ready and that the next step is `/review-task NNN` in a new session. Do not review your own work; a reviewer with fresh context is the point.
