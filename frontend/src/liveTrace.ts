// The live trace's logic, with no React and no DOM: what the trace panel shows is built here from the events received so
// far (the events and their order are the backend's, backend/fukidashi/live_trace.py). Only types come from './api', so
// the module loads anywhere and Vitest checks it directly (OUTLINE §8, "Frontend tests").
import type { CallRecord, EventOf, StageKind, TraceEvent } from './api'

// ---- Holding the events ------------------------------------------------------------------------------------------

/** The events by `seq`, each `seq` once. They may have arrived in any order, and a replay (a stream opened anew for a run
 *  that was watched already) sends events again: the first one received for a `seq` stays. */
export function orderEvents(received: readonly TraceEvent[]): TraceEvent[] {
  const seen = new Set<number>()
  const kept: TraceEvent[] = []
  for (const event of received) {
    if (seen.has(event.seq)) continue
    seen.add(event.seq)
    kept.push(event)
  }
  return kept.sort((a, b) => a.seq - b.seq)
}

/** `held` (in `seq` order, each `seq` once, as this function keeps it) with `event` added. A `seq` that is held already is
 *  ignored, and then `held` itself comes back, so a state that holds the events does not change. */
export function addEvent(held: readonly TraceEvent[], event: TraceEvent): readonly TraceEvent[] {
  let at = held.length
  while (at > 0 && held[at - 1].seq > event.seq) at--
  if (at > 0 && held[at - 1].seq === event.seq) return held
  return [...held.slice(0, at), event, ...held.slice(at)]
}

/** Events `from` to `to`, both included, that never arrived. */
export interface Gap {
  from: number
  to: number
}

/** The `seq`s below the highest one held that are not held: a run's first event is seq 1 and each rises by 1, so none
 *  should be missing. A hole at the end is not known (the next event may be on its way), so it is not a gap. */
export function findGaps(events: readonly TraceEvent[]): Gap[] {
  const seqs = [...new Set(events.map((e) => e.seq))].sort((a, b) => a - b)
  const gaps: Gap[] = []
  let expected = 1
  for (const seq of seqs) {
    if (seq > expected) gaps.push({ from: expected, to: seq - 1 })
    expected = seq + 1
  }
  return gaps
}

/** The gaps as a person reads them: "3–4, 9". */
export const describeGaps = (gaps: readonly Gap[]): string =>
  gaps.map((g) => (g.from === g.to ? `${g.from}` : `${g.from}–${g.to}`)).join(', ')

// ---- The view: events grouped by stage ---------------------------------------------------------------------------

/** One stage of the Orchestrator's flow, or finalize: its `stage_start`, what happened inside, and its `stage_end`. */
export interface Stage {
  kind: StageKind
  number: number | null // review and revise stages are numbered from 1 in the order they began; null for the others
  title: string // "Translate", "Review 1", "Revision 2", "Finalize"
  start: EventOf<'stage_start'>
  events: TraceEvent[] // between the start and the end, in seq order
  end: EventOf<'stage_end'> | null // null while the stage runs, and for a run that failed inside it
}

export type Block = { type: 'stage'; stage: Stage } | { type: 'event'; event: TraceEvent }

export interface TraceView {
  start: EventOf<'run_start'> | null // the run's start and end are outside the stages
  blocks: Block[] // the stages in order; an event outside every stage (a run with no stages) is a block of its own
  end: EventOf<'run_end'> | null
  gaps: Gap[]
  count: number // the events held, each seq once
}

const TITLES: Record<string, string> = { translate: 'Translate', review: 'Review', revise: 'Revision', finalize: 'Finalize' }
const NUMBERED = new Set<string>(['review', 'revise'])

export function stageTitle(kind: string, number: number | null): string {
  const name = TITLES[kind] ?? kind
  return number === null ? name : `${name} ${number}`
}

/** Every event of `received` lands in exactly one place: the run's start or end, a stage, or a block of its own.
 *  - `received` may be in any order and may repeat a seq (see `orderEvents`).
 *  - A stage runs from its `stage_start` to the `stage_end` of the same stage and index; what comes between belongs to it.
 *  - `run_end` is never inside a stage: a run that failed in the middle of one leaves that stage without an end.
 *  - A `stage_end` that matches no open stage, or a second `run_start` or `run_end`, is kept where it falls (it can only
 *    happen after a gap, which `gaps` reports). */
export function buildView(received: readonly TraceEvent[]): TraceView {
  const events = orderEvents(received)
  const view: TraceView = { start: null, blocks: [], end: null, gaps: findGaps(events), count: events.length }
  const began = new Map<string, number>() // stages begun, by kind
  let open: Stage | null = null
  for (const event of events) {
    if (event.kind === 'run_start' && view.start === null) {
      view.start = event
    } else if (event.kind === 'run_end' && view.end === null) {
      view.end = event
    } else if (event.kind === 'stage_start') {
      const kind = event.data.stage
      const n = (began.get(kind) ?? 0) + 1
      began.set(kind, n)
      const number = NUMBERED.has(kind) ? n : null
      open = { kind, number, title: stageTitle(kind, number), start: event, events: [], end: null }
      view.blocks.push({ type: 'stage', stage: open })
    } else if (event.kind === 'stage_end' && open !== null && open.kind === event.data.stage && open.start.data.index === event.data.index) {
      open.end = event
      open = null
    } else if (open !== null) {
      open.events.push(event)
    } else {
      view.blocks.push({ type: 'event', event })
    }
  }
  return view
}

// ---- Words for the rows -----------------------------------------------------------------------------------------

/** `text` cut to its first `max` characters (code points, so no character is split), and whether anything was cut. */
export function shorten(text: string, max: number): { text: string; cut: boolean } {
  const chars = Array.from(text)
  return chars.length <= max ? { text, cut: false } : { text: chars.slice(0, max).join(''), cut: true }
}

/** A tool call's arguments in short: the JSON the model sent on one line, cut to `max` characters. */
export function shortArguments(args: string | null, max = 80): string {
  if (args === null) return ''
  let line: string
  try {
    line = JSON.stringify(JSON.parse(args))
  } catch {
    line = args.replace(/\s+/g, ' ').trim() // not JSON (the model sent something else): as it came
  }
  const { text, cut } = shorten(line, max)
  return cut ? `${text}…` : text
}

/** A call's tokens; usage the provider did not send is unknown, never 0. */
export function usageText(record: CallRecord | null): string {
  const usage = record?.usage
  if (!usage) return 'usage unknown'
  const cached = usage.cached_input_tokens ? ` (${usage.cached_input_tokens} cached)` : ''
  return `${usage.input_tokens} in${cached} / ${usage.output_tokens} out tokens`
}

export const costText = (cost: number | null): string => (cost === null ? 'cost unknown' : `$${cost.toFixed(5)}`)

/** Finalize's counts of proposal results, without the zeros. */
export const nonZeroCounts = (results: Record<string, number> | undefined): [string, number][] =>
  Object.entries(results ?? {}).filter(([, n]) => n > 0)
