// Mirrors the response models in backend/fukidashi/api/pages.py, and, below the live trace, those in api/kb.py.

export interface PageSummary {
  book_title: string
  page_index: number
  segment_count: number
  translated: boolean
}

export interface Segment {
  index: number
  x: number
  y: number
  w: number
  h: number
  text_ja: string
  text_zh_hant: string // reference
  text_en: string // reference
  text_zh_hans: string | null // latest succeeded run
}

export interface Run {
  id: number
  config: string
  model: string
  response_model: string | null
  status: string
  calls: number
  unknown_usage_calls: number
  input_tokens: number | null
  output_tokens: number | null
  cost_usd: number | null
  started_at: string
  finished_at: string | null
}

// What the Orchestrator's stages say about the displayed translation (backend/fukidashi/review_notes.py builds it from the
// stored run, so a page the dev books already show has it with no model call). A note is a Flag or an Issue.
export interface ReviewFlag {
  region: number
  reason: string // Simplified Chinese
  pass: number // 0: the first pass; n: the nth revision
  current: boolean // made by the stage that produced the displayed translation, and the region is not edited
}
export interface ReviewIssue {
  region: number
  comment: string // Simplified Chinese
  suggestion: string | null
  verdict: string | null // the verdict of the review that gave it: 'approve' or 'revise'; null: that review gave none
  review: number // 1 for the first review, 2 for the second
  current: boolean // from the review that judged the displayed translation, and the region is not edited
}
export interface PageReview {
  run_id: number // the review's run: the `full` or `full_no_kb` run whose stages produced the translation
  config: string
  send_backs: number
  final_stop: string // 'approved', 'review_unusable', 'revision_unusable' or 'send_back_limit'
  edited: number[] // the regions a person edited since that run, sorted
  flags: ReviewFlag[]
  issues: ReviewIssue[]
}

export interface PageDetail {
  book_title: string
  page_index: number
  image_url: string
  segments: Segment[]
  run: Run | null // latest succeeded run
  running_run_id: number | null // the page's running run, if any: a reload watches it
  review: PageReview | null // null: the displayed translation has no review (`baseline`, `translator`, no run yet)
}

// The configs the editor runs (OUTLINE §8, "Running pages from the editor").
export type TranslateConfig = 'baseline' | 'full'

// ---- The live trace: the events of one run (backend/fukidashi/live_trace.py; its docstring is the contract) ----------
// An event carries no page image, no reasoning text (only its length) and no reference translation.

// A model call as the backend's Meter has it (backend/fukidashi/llm/metering.py).
export interface CallRecord {
  page_id: string
  requested_model: string
  response_model: string | null
  usage: { input_tokens: number; output_tokens: number; cached_input_tokens: number | null } | null // null: unknown, not zero
  latency_s: number
  cost_usd: number | null // null: unknown
  error: string | null
}

export type StageKind = 'translate' | 'review' | 'revise' | 'finalize'
export type StageStatus = 'ok' | 'unusable' | 'failed' | 'skipped' // finalize's Curator can also be skipped

export interface Flag { id: number; reason: string } // a region the Translator is not sure of
export interface Issue { id: number; comment: string; suggestion: string | null } // the Reviewer's remark on one region
export interface AskedToolCall { id: string; name: string; arguments: string | null } // arguments: the JSON text the model sent

export interface RunStartData { run_id: number; config: string; model: string }
export interface StageStartData { stage: StageKind; index: number | null } // index: null for finalize
export interface ModelCallData {
  step: number // 1-based, within the agent
  record: CallRecord | null
  finish_reason: string | null // null: the call failed
  text: string
  reasoning_chars: number | null // null: the reply had no reasoning
  error: string | null
  tool_calls: AskedToolCall[]
}
export interface ToolResultData {
  tool_call_id: string
  name: string
  result: string | null // one of result and error is null
  error: string | null
  ran: boolean // the tool's handler was called
  duplicate: boolean // refused as a repeat of an earlier call
}
export interface RepairData { record: CallRecord | null; error: string | null; parse_error: string }
export interface FallbackData { reason: string }
export interface StageEndData {
  stage: StageKind
  index: number | null
  status: StageStatus
  error: string | null
  flags?: Flag[] // translate and revise stages
  verdict?: 'approve' | 'revise' | null // review stages (null: no verdict)
  issues?: Issue[] // review stages
  results?: Record<string, number> // finalize: how many proposals got each result
}

// The data of a run's last event, `run_end`: the stored row's outcome.
export interface RunEnd {
  status: 'succeeded' | 'failed'
  error: string | null
  calls: number
  unknown_usage_calls: number
  input_tokens: number | null
  output_tokens: number | null
  cost_usd: number | null
  // only for the Orchestrator's configs (`full`)
  send_backs?: number
  final_stop?: string
  final_reviewed?: boolean
}

// One event as the stream delivers it: `seq` rises by 1 from 1 within the run.
export type TraceEvent =
  | { seq: number; kind: 'run_start'; data: RunStartData }
  | { seq: number; kind: 'stage_start'; data: StageStartData }
  | { seq: number; kind: 'model_call'; data: ModelCallData }
  | { seq: number; kind: 'tool_result'; data: ToolResultData }
  | { seq: number; kind: 'repair'; data: RepairData }
  | { seq: number; kind: 'fallback'; data: FallbackData }
  | { seq: number; kind: 'stage_end'; data: StageEndData }
  | { seq: number; kind: 'run_end'; data: RunEnd }
export type EventKind = TraceEvent['kind']
export type EventOf<K extends EventKind> = Extract<TraceEvent, { kind: K }>
export const EVENT_KINDS: readonly EventKind[] = [
  'run_start', 'stage_start', 'model_call', 'tool_result', 'repair', 'fallback', 'stage_end', 'run_end',
]

// An event from the wire: the SSE `event:` name, the `id:` the browser hands over as `lastEventId` (the event's seq) and
// the `data:` text. Null when it cannot be one (no positive whole number as id, data that is not a JSON object): the
// event is dropped, and a later one shows the hole as a gap. The shape of `data` is the backend's word, not checked here.
export function parseEvent(kind: EventKind, lastEventId: string, data: string): TraceEvent | null {
  const seq = /^[0-9]+$/.test(lastEventId) ? Number(lastEventId) : 0
  if (!Number.isSafeInteger(seq) || seq < 1) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(data)
  } catch {
    return null
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null
  return { seq, kind, data: parsed } as TraceEvent
}

// A failed request, with the status the backend answered (0: it never answered).
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new ApiError('Cannot reach the backend. Is it running on port 8000?', 0)
  }
  if (!res.ok) {
    // A 5xx without the API's {"detail": ...} body comes from the dev server's proxy: the backend is down.
    let detail = res.status >= 500 ? `Cannot reach the backend (${res.status}). Is it running on port 8000?` : `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // not JSON: keep the message above
    }
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

const pagePath = (book: string, index: number) => `/api/pages/${encodeURIComponent(book)}/${index}`

export const fetchPages = () => request<PageSummary[]>('/api/pages')
export const fetchPage = (book: string, index: number, signal?: AbortSignal) =>
  request<PageDetail>(pagePath(book, index), { signal })
// Starts the page in the background and answers with the run's id; the run's end comes over `watchRun`.
export const translatePage = (book: string, index: number, config: TranslateConfig) =>
  request<{ run_id: number }>(`${pagePath(book, index)}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  })

// Opens the run's live trace (GET /api/runs/:id/events). Every event goes to `onEvent` as the stream sends it (`run_end`
// too, before `onEnd`), and `onEnd` is called once, with `run_end`'s data, or with null when the stream is over without
// one (the run was finished already, or the backend refused the stream). The browser reconnects by itself after a
// dropped stream and sends the last event id, so nothing is missed; a stream opened anew, after a reload say, replays
// the run from seq 1, so a caller that kept events must ignore a seq it has. Returns a function that closes the
// stream; after it neither callback is called.
export function watchRun(
  runId: number,
  onEvent: (event: TraceEvent) => void,
  onEnd: (end: RunEnd | null) => void,
): () => void {
  const source = new EventSource(`/api/runs/${runId}/events`)
  let over = false
  const finish = (end: RunEnd | null) => {
    if (over) return
    over = true
    source.close()
    onEnd(end)
  }
  // The browser hands a named event only to a listener of that name, so each kind has its own.
  for (const kind of EVENT_KINDS) {
    source.addEventListener(kind, (e) => {
      if (over) return
      const { lastEventId, data } = e as MessageEvent<string>
      const event = parseEvent(kind, lastEventId, data)
      if (event) onEvent(event)
      if (kind === 'run_end') finish(event ? (event.data as RunEnd) : null)
    })
  }
  // A 204 or an error status closes the source for good; a dropped connection leaves it connecting again.
  source.addEventListener('error', () => {
    if (source.readyState === EventSource.CLOSED) finish(null)
  })
  return () => {
    over = true
    source.close()
  }
}

// A person's edit of one segment's translation: the page, with the edit as its latest run.
export const editTranslation = (book: string, index: number, segment: number, text: string) =>
  request<PageDetail>(`${pagePath(book, index)}/segments/${segment}/translation`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text_zh_hans: text }),
  })

// ---- A book's Editor KB (backend/fukidashi/api/kb.py; each field is commented there) -----------------------------------
// The Proposal Queue, the Human Review Queue, the entries and the profiles of the KB the editor keeps for a book (OUTLINE §0,
// Editor KB), and what a person's decision answers with. kbView.ts says what each field means in words.

// One change of a row's history. The backend sends a dict: `at` (UTC, seconds) and `event` are always there; a person's
// decision also has `by` ('human') and `from` (the status it replaced), and the rest depends on the event (`page`, `run_id`,
// `proposal_id`, `reason`, `aliases`). kbView.ts reads the keys it knows and checks their types.
export type HistoryLine = { at: string; event: string } & Record<string, unknown>

// A glossary entry, in any status: a rejected one stays listed, so a person can approve it again.
export interface KBEntry {
  id: number
  source_term: string // as written in the Japanese text
  target_term: string // its rendering in Simplified Chinese
  category: string // a short label such as place or organisation; may be empty
  status: string // 'approved' or 'pending_human' (the agents can find it), 'rejected' or 'proposed' (they cannot)
  provenance: string // 'agent', or 'human' once a person has decided the entry (approved, rejected, or made it)
  evidence_pages: number[] // the pages that show it, ascending
  history: HistoryLine[] // one object per change, oldest first
}

// A character profile, in any status.
export interface KBProfile {
  id: number
  name_src: string // the name as written in the Japanese text
  name_tgt: string // its rendering in Simplified Chinese
  aliases: string[] // other Japanese names the character is called by
  notes: string // speech style and anything else worth keeping; may be empty
  status: string // as for an entry
  provenance: string // derived, a profile has no such column: 'human' when a history line has `by` 'human', else 'agent'
  evidence_pages: number[]
  history: HistoryLine[] // as for an entry
}

// A proposal in the Proposal Queue (status 'proposed': a term nobody decided yet).
export interface KBProposal {
  proposal_id: string
  kind: string // 'term' or 'character'
  payload: Record<string, unknown> // a term: source_term, target_term, category; a character: name_src, name_tgt, aliases, notes
  proposer: string // who proposed it ('translator')
  page_index: number // the page it was proposed on
  run_id: number // the Run that proposed it
  reason: string | null // null while the proposal is undecided; later what was said when it was decided or sent to a person
}

// A proposal in the Human Review Queue (status 'pending_human'), and what it waits for.
export interface ReviewProposal extends KBProposal {
  waits_for: string // 'new_name': a profile was made and waits for a person; 'conflict': nothing was written for it
  profile_id: number | null // the new name's profile (a `KBProfile.id`); null for a conflict
}

// How many of the KB's proposals have each status, decided ones included; every status is there, 0 when none.
export interface ProposalCounts {
  proposed: number
  approved: number
  rejected: number
  pending_human: number
}

// The book's Editor KB as it is now (`KBOut`). `kb_id` is null while the book has no Editor KB: every list is empty then.
export interface KBView {
  kb_id: number | null
  entries: KBEntry[] // every status, by id
  profiles: KBProfile[] // every status, by id
  proposal_queue: KBProposal[] // the Proposal Queue, in the order the proposals were queued
  human_review_queue: ReviewProposal[] // the Human Review Queue, in the same order
  proposal_counts: ProposalCounts
}

// What a person decides (the body of every decision is exactly `{ decision }`).
export type KBDecision = 'approve' | 'reject'

// A proposal's decision answers what it did and the KB after it. `outcome` is one of `curator.HUMAN_OUTCOMES`; kbView.ts words
// the seven the proposals route sends and shows any other as it is.
export interface ProposalDecision {
  outcome: string
  kb: KBView
}

const kbPath = (book: string) => `/api/books/${encodeURIComponent(book)}/kb`

const decisionRequest = (decision: KBDecision): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ decision }),
})

// The book's Editor KB. A book that has none yet answers `kb_id` null and empty lists: reading never makes one.
export const fetchKB = (book: string, signal?: AbortSignal) => request<KBView>(kbPath(book), { signal })

// A person's decisions. A 409 carries the reason the rules refused it, in the error's message; a decision writes nothing when
// it is refused. The proposals route answers the outcome and the KB, the other two the KB.
export const decideProposal = (book: string, proposalId: string, decision: KBDecision) =>
  request<ProposalDecision>(
    `${kbPath(book)}/proposals/${encodeURIComponent(proposalId)}/decision`,
    decisionRequest(decision),
  )
export const decideEntry = (book: string, entryId: number, decision: KBDecision) =>
  request<KBView>(`${kbPath(book)}/entries/${entryId}/decision`, decisionRequest(decision))
export const decideProfile = (book: string, profileId: number, decision: KBDecision) =>
  request<KBView>(`${kbPath(book)}/profiles/${profileId}/decision`, decisionRequest(decision))
