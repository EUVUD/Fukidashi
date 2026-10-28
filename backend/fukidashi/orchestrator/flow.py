"""The Orchestrator's flow for one page: translate, review, and send back at most twice.

Plain code, not an LLM (OUTLINE §1.3): this file decides who runs next and when the loop ends, and the agents only
answer. It takes plain values (the page, the earlier pages, an `LLMClient`, the budgets) and returns a plain result;
`runs.py` reads the earlier pages before it starts and stores the Run after it ends. It imports no LangChain and no
database.

    translate  the flagging Translator on the page (it may fall back to `baseline`, as in M2)
    review     the Reviewer on the current translation, with the flags of the run that produced it
    revise     a Translator run that also gets the translation and the Reviewer's Issues; it never falls back
               (a revision that gives no usable answer leaves the translation it was meant to replace)

A `revise` verdict sends the page back, at most `MAX_SEND_BACKS` times. After the last revision there is no further
review, because another `revise` could not be acted on. A provider error or a spent budget in any stage fails the Run
with that stage's error, and nothing more is sent (OUTLINE §3.6); the translations of the earlier stages are dropped,
so a `full_no_kb` page is the whole pipeline or nothing.

The Knowledge Base is a switch (OUTLINE §4.2): a `KBSnapshot` turns it on, `None` leaves `full_no_kb`. The caller takes
the snapshot once, before the page starts, and the whole page uses it, so what the Translator proposes on this page is
invisible to this page's lookups. The Orchestrator makes the page's one `ProposalCollector`, gives it to every
Translator run, and returns the proposals in the order of the calls (see `run_orchestrator`). Writing them is the
Curator's, later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..agents import reviewer, translator
from ..agents.loop import DEFAULT_MAX_STEPS
from ..datasets.openmantra import Page
from ..kb.collector import ProposalCollector
from ..kb.entities import Proposal
from ..kb.snapshot import KBSnapshot
from ..llm.client import LLMClient
from ..pipeline import baseline

CONFIG_NAME = "full_no_kb"
KB_CONFIG_NAME = "full"  # the config name in the attempts record when a KB snapshot is given
MAX_SEND_BACKS = 2  # OUTLINE §3.2: fixed, not tuned

# The kinds of stage, in the order of the flow.
TRANSLATE = "translate"
REVIEW = "review"
REVISE = "revise"

# A stage's status. The Reviewer's own words, which the Translator stages share: `unusable` means no usable answer and
# nothing more sent, with the previous translation standing; `failed` means a provider error or a spent budget.
OK = reviewer.OK
UNUSABLE = reviewer.UNUSABLE
FAILED = reviewer.FAILED

# Why the loop stopped. `FAILED` is the stop of a Run that failed, where no translation is stored.
APPROVED = "approved"
REVIEW_UNUSABLE = "review_unusable"
REVISION_UNUSABLE = "revision_unusable"
SEND_BACK_LIMIT = "send_back_limit"
FINAL_STOPS = (APPROVED, REVIEW_UNUSABLE, REVISION_UNUSABLE, SEND_BACK_LIMIT)


@dataclass
class OrchestratorResult:
    translations: dict[int, str] | None  # region id -> zh-Hans; None when the Run failed
    attempts: dict  # what `runs.py` stores: see run_orchestrator
    error: str | None = None  # why translations is None
    proposals: tuple[Proposal, ...] = ()  # the page's KB proposals, once each, in call order; empty when the Run failed

    @property
    def ok(self) -> bool:
        return self.translations is not None

    @property
    def send_backs(self) -> int:
        return self.attempts["send_backs"]

    @property
    def final_stop(self) -> str:
        return self.attempts["final"]["stop"]

    @property
    def fallback_reason(self) -> str | None:
        """Why the first pass fell back to `baseline`; None when it did not (a revision never falls back)."""
        fallback = self.attempts["stages"][0]["attempts"].get("fallback")
        return None if fallback is None else fallback["reason"]


def run_orchestrator(
    page: Page,
    client: LLMClient,
    *,
    previous: Sequence[translator.PreviousPage] = (),
    agent_budget: int = translator.DEFAULT_PAGE_BUDGET,
    review_budget: int = reviewer.DEFAULT_REVIEW_BUDGET,
    fallback_budget: int = baseline.DEFAULT_PAGE_BUDGET,
    max_tokens: int = baseline.DEFAULT_MAX_TOKENS,
    max_steps: int = DEFAULT_MAX_STEPS,
    kb: KBSnapshot | None = None,
) -> OrchestratorResult:
    """Run one page through translate → review → revise. Never raises for provider errors or bad output.

    Every Translator run (the first pass and each revision) has `agent_budget` tokens to itself, every review has
    `review_budget`, and the first pass's fallback has `fallback_budget`; the most a page can use is their sum.
    `kb` is the one switch between `full_no_kb` (None) and `full`: with a snapshot the Translator also has the KB's
    lookups and `propose_entry`, the Reviewer has the lookups, and both get the KB's entries for the page in their user
    message; without one, every request is what it is without the KB.

    `proposals` of the result are the proposals the Translator's `propose_entry` calls made, each id once, ordered by
    where its call stands: the Translator run (the first pass, then each revision in turn), then the step, then the
    call's position in the model's reply. When several calls give one id (the same kind, source and target) with other
    details, the payload kept is the first call's in that order. The calls of one reply run in parallel, so the order
    they arrive in is a race, and neither the order nor the payload depends on it (`translator.step_proposals`). Every
    call that `make_proposal` accepted counts, from every Translator run of the page, including a first pass that fell
    back to `baseline` and a revision that was unusable: whether a proposal fits the stored translation is the Curator's
    judgment. A Run that failed has none.

    The attempts record: {"config": "full_no_kb" ("full" with a `kb`),
      "stages": [{"kind": "translate" | "review" | "revise", "status": "ok" | "unusable" | "failed", "error",
                  "attempts": <the agent's own record>,
                  # a translate or revise stage also has "flags"; a review stage also has "verdict" and "issues"
                 }, ...],                                            # in the order they ran
      "send_backs": <the number of revise stages>,
      "final": {"stage": <index in stages of the stage that produced the stored translation>, "kind",
                "reviewed": <whether a review gave a verdict on the stored translation>,
                "stop": "approved" | "review_unusable" | "revision_unusable" | "send_back_limit" | "failed"}}
    When the Run failed, the failed stage is the last one and `final` has no stage (`stage`, `kind` null).
    """
    stages: list[dict] = []
    collector = ProposalCollector(kb.kb_id, page.page_index) if kb is not None else None
    made: list[Sequence[Proposal]] = []  # the proposals of each Translator run, in the order the runs happened

    def finish(stop: str, *, translations: dict[int, str] | None = None, produced_by: int | None = None,
               reviewed: bool = False, error: str | None = None) -> OrchestratorResult:
        final = {"stage": produced_by, "kind": None if produced_by is None else stages[produced_by]["kind"],
                 "reviewed": reviewed, "stop": stop}
        send_backs = sum(s["kind"] == REVISE for s in stages)
        attempts = {"config": CONFIG_NAME if kb is None else KB_CONFIG_NAME, "stages": stages,
                    "send_backs": send_backs, "final": final}
        return OrchestratorResult(translations, attempts, error,
                                  merge_proposals(*made) if translations is not None else ())

    first = translator.run_translator(
        page, client, previous=previous, agent_budget=agent_budget, fallback_budget=fallback_budget,
        max_tokens=max_tokens, max_steps=max_steps, flagging=True, kb=kb, collector=collector,
    )
    stages.append(_translator_stage(TRANSLATE, first))
    made.append(first.proposals)
    if not first.ok:  # a provider error or a spent budget, or the fallback failed too
        return finish(FAILED, error=first.error)

    current, flags, produced_by = first.translations, first.flags, len(stages) - 1
    while True:
        review = reviewer.run_reviewer(
            page, current, flags, client, review_budget=review_budget, max_tokens=max_tokens, max_steps=max_steps,
            kb=kb,
        )
        stages.append({"kind": REVIEW, "status": review.status, "error": review.error, "verdict": review.verdict,
                       "issues": review.issues, "attempts": review.attempts})
        if review.failed:
            return finish(FAILED, error=review.error)
        if review.unusable:  # no verdict: the translation stands as it is
            return finish(REVIEW_UNUSABLE, translations=current, produced_by=produced_by)
        if review.verdict == reviewer.APPROVE:
            return finish(APPROVED, translations=current, produced_by=produced_by, reviewed=True)

        revision = translator.run_translator(
            page, client, previous=previous, agent_budget=agent_budget, max_tokens=max_tokens, max_steps=max_steps,
            flagging=True, revision=translator.Revision(current, review.issues), kb=kb, collector=collector,
        )
        stages.append(_translator_stage(REVISE, revision))
        made.append(revision.proposals)  # before anything else: an unusable revision's calls were made too
        if revision.unusable:  # the translation the Reviewer sent back stands
            return finish(REVISION_UNUSABLE, translations=current, produced_by=produced_by, reviewed=True)
        if not revision.ok:
            return finish(FAILED, error=revision.error)
        current, flags, produced_by = revision.translations, revision.flags, len(stages) - 1
        if sum(s["kind"] == REVISE for s in stages) == MAX_SEND_BACKS:  # no further review: see the module docstring
            return finish(SEND_BACK_LIMIT, translations=current, produced_by=produced_by)


def merge_proposals(*runs: Sequence[Proposal]) -> tuple[Proposal, ...]:
    """The proposals of one page's Translator runs (each run's `proposals`, in the order the runs happened) as one
    list: each id once, in the order of the calls, with the payload of the first call that made it. So a repeat in a
    revision, with other details, does not replace the first pass's."""
    kept: dict[str, Proposal] = {}
    for proposals in runs:
        for proposal in proposals:
            kept.setdefault(proposal.proposal_id, proposal)
    return tuple(kept.values())


def _translator_stage(kind: str, result: translator.TranslatorResult) -> dict:
    status = OK if result.ok else UNUSABLE if result.unusable else FAILED
    return {"kind": kind, "status": status, "error": result.error, "flags": result.flags, "attempts": result.attempts}
