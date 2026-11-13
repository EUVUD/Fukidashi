import { describe, expect, it } from 'vitest'
import type { CallRecord, EventKind, StageEndData, StageKind, TraceEvent } from './api'
import {
  addEvent,
  buildView,
  costText,
  describeGaps,
  findGaps,
  nonZeroCounts,
  orderEvents,
  shortArguments,
  shorten,
  stageTitle,
  usageText,
  type Block,
  type Stage,
  type TraceView,
} from './liveTrace'

// ---- Fixtures: events as the backend sends them (backend/fukidashi/live_trace.py) ----------------------------------

type Spec = [EventKind, object]

const record = (over: Partial<CallRecord> = {}): CallRecord => ({
  page_id: 'book/1',
  requested_model: 'model',
  response_model: 'model',
  usage: { input_tokens: 1000, output_tokens: 200, cached_input_tokens: null },
  latency_s: 1.5,
  cost_usd: 0.0004,
  error: null,
  ...over,
})

const runStart = (config = 'full'): Spec => ['run_start', { run_id: 7, config, model: 'model' }]
const stageStart = (stage: StageKind, index: number | null): Spec => ['stage_start', { stage, index }]
const stageEnd = (stage: StageKind, index: number | null, over: Partial<StageEndData> = {}): Spec => [
  'stage_end',
  { stage, index, status: 'ok', error: null, ...over },
]
const modelCall = (step: number, tools: string[] = []): Spec => [
  'model_call',
  {
    step,
    record: record(),
    finish_reason: tools.length ? 'tool_calls' : 'stop',
    text: '',
    reasoning_chars: null,
    error: null,
    tool_calls: tools.map((name, n) => ({ id: `call_${step}_${n}`, name, arguments: '{}' })),
  },
]
const toolResult = (name: string): Spec => [
  'tool_result',
  { tool_call_id: 'call_1_0', name, result: 'ok', error: null, ran: true, duplicate: false },
]
const repair: Spec = ['repair', { record: record(), error: null, parse_error: 'no JSON object in the response' }]
const fallback: Spec = ['fallback', { reason: 'step_limit' }]
const runEnd = (over: object = {}): Spec => [
  'run_end',
  { status: 'succeeded', error: null, calls: 3, unknown_usage_calls: 0, input_tokens: 3000, output_tokens: 600, cost_usd: 0.0012, ...over },
]

/** The specs as events of one run: seq 1, 2, 3 in the order given. */
const trace = (...specs: Spec[]): TraceEvent[] =>
  specs.map(([kind, data], n) => ({ seq: n + 1, kind, data }) as TraceEvent)

/** A `full` run that is sent back twice: translate, review 1, revision 1, review 2, revision 2 (no third review), finalize. */
const sentBackTwice = (): TraceEvent[] =>
  trace(
    runStart(), // 1
    stageStart('translate', 0), // 2
    modelCall(1, ['get_previous_context']), // 3
    toolResult('get_previous_context'), // 4
    modelCall(2), // 5
    stageEnd('translate', 0, { flags: [{ id: 2, reason: 'a pun' }] }), // 6
    stageStart('review', 1), // 7
    modelCall(1), // 8
    stageEnd('review', 1, { verdict: 'revise', issues: [{ id: 2, comment: 'c', suggestion: 's' }] }), // 9
    stageStart('revise', 2), // 10
    modelCall(1), // 11
    stageEnd('revise', 2, { flags: [] }), // 12
    stageStart('review', 3), // 13
    modelCall(1), // 14
    stageEnd('review', 3, { verdict: 'revise', issues: [{ id: 4, comment: 'c', suggestion: null }] }), // 15
    stageStart('revise', 4), // 16
    modelCall(1), // 17
    stageEnd('revise', 4, { flags: [] }), // 18
    stageStart('finalize', null), // 19
    repair, // 20
    stageEnd('finalize', null, { results: { accepted: 1, undecided: 0 } }), // 21
    runEnd({ send_backs: 2, final_stop: 'send_back_limit', final_reviewed: false }), // 22
  )

const seqs = (events: readonly TraceEvent[]) => events.map((e) => e.seq)
const stages = (blocks: Block[]): Stage[] => blocks.flatMap((b) => (b.type === 'stage' ? [b.stage] : []))
/** Every seq the view holds, wherever it put it. */
const placed = (view: TraceView): number[] =>
  [
    ...(view.start ? [view.start.seq] : []),
    ...view.blocks.flatMap((b) =>
      b.type === 'event'
        ? [b.event.seq]
        : [b.stage.start.seq, ...seqs(b.stage.events), ...(b.stage.end ? [b.stage.end.seq] : [])],
    ),
    ...(view.end ? [view.end.seq] : []),
  ].sort((a, b) => a - b)

// ---- Order and duplicates ---------------------------------------------------------------------------------------

describe('events in seq order', () => {
  it('puts events in seq order whatever order they came in', () => {
    const run = trace(runStart(), modelCall(1), modelCall(2), modelCall(3), modelCall(4), modelCall(5), runEnd())
    const shuffled = [5, 2, 7, 1, 4, 6, 3].map((seq) => run[seq - 1])
    expect(seqs(orderEvents(shuffled))).toEqual([1, 2, 3, 4, 5, 6, 7])
    const view = buildView(shuffled)
    expect(view.start?.seq).toBe(1)
    expect(view.blocks.map((b) => (b.type === 'event' ? b.event.seq : -1))).toEqual([2, 3, 4, 5, 6])
    expect(view.end?.seq).toBe(7)
  })

  it('builds the same view from a run that arrived in reverse', () => {
    const run = sentBackTwice()
    expect(buildView([...run].reverse())).toEqual(buildView(run))
  })

  it('addEvent puts a late event where it belongs', () => {
    const run = trace(runStart(), modelCall(1), modelCall(2), modelCall(3))
    let held: readonly TraceEvent[] = []
    for (const event of [run[0], run[3], run[1], run[2]]) held = addEvent(held, event)
    expect(seqs(held)).toEqual([1, 2, 3, 4])
    expect(findGaps(held)).toEqual([])
  })

  it('does not change what it was given', () => {
    const run = trace(runStart(), modelCall(1), modelCall(2))
    const arrived = [run[2], run[0], run[1]]
    orderEvents(arrived)
    buildView(arrived)
    expect(seqs(arrived)).toEqual([3, 1, 2])
    const held = [run[0], run[1]]
    addEvent(held, run[2])
    expect(seqs(held)).toEqual([1, 2])
  })
})

describe('a seq seen twice', () => {
  it('is ignored, and the first event received for it stays', () => {
    const [first, second] = trace(modelCall(1), modelCall(2))
    const again = { ...second, seq: 1 } as TraceEvent // another event under seq 1
    const kept = orderEvents([first, again, second])
    expect(seqs(kept)).toEqual([1, 2])
    expect(kept[0]).toBe(first)
  })

  it('addEvent returns the very same array for a seq it holds', () => {
    const run = trace(runStart(), modelCall(1), modelCall(2))
    const held = run as readonly TraceEvent[]
    expect(addEvent(held, run[1])).toBe(held)
    expect(addEvent(held, { ...run[0], seq: 3 } as TraceEvent)).toBe(held) // held by another event, still ignored
  })

  it('counts a repeated seq once in the view', () => {
    const run = sentBackTwice()
    const view = buildView([...run, run[3], run[3], run[0]])
    expect(view.count).toBe(22)
    expect(placed(view)).toEqual(seqs(run))
  })
})

describe('a replay that starts again from 1 after events were seen', () => {
  it('leaves each event once and no gap, and goes on live', () => {
    const run = sentBackTwice()
    let held: readonly TraceEvent[] = []
    for (const event of run.slice(0, 9)) held = addEvent(held, event) // the first connection, up to seq 9
    for (const event of run.slice(0, 15)) held = addEvent(held, event) // a stream opened anew replays 1..15
    for (const event of run.slice(15)) held = addEvent(held, event) // and goes on with the rest
    expect(seqs(held)).toEqual(seqs(run))
    expect(findGaps(held)).toEqual([])
    expect(buildView(held)).toEqual(buildView(run))
  })

  it('does the same through buildView alone, which takes the events as they were received', () => {
    const run = sentBackTwice()
    const received = [...run.slice(0, 9), ...run] // seqs 1..9, then all of them again
    expect(buildView(received)).toEqual(buildView(run))
  })
})

// ---- Gaps -------------------------------------------------------------------------------------------------------

describe('gaps', () => {
  it('finds none in a run that arrived whole, and none for no events', () => {
    expect(buildView(sentBackTwice()).gaps).toEqual([])
    expect(findGaps([])).toEqual([])
  })

  it('names the seqs that never arrived', () => {
    const run = trace(...Array(10).fill(modelCall(1)))
    const arrived = run.filter((e) => ![3, 4, 7].includes(e.seq))
    expect(buildView(arrived).gaps).toEqual([{ from: 3, to: 4 }, { from: 7, to: 7 }])
    expect(describeGaps(buildView(arrived).gaps)).toBe('3–4, 7')
  })

  it('counts events missing at the start, since a run starts at seq 1', () => {
    const run = trace(...Array(5).fill(modelCall(1)))
    expect(buildView(run.slice(2)).gaps).toEqual([{ from: 1, to: 2 }])
  })

  it('closes a gap when the late event arrives', () => {
    const run = trace(runStart(), modelCall(1), modelCall(2))
    expect(buildView([run[0], run[2]]).gaps).toEqual([{ from: 2, to: 2 }])
    expect(buildView([run[0], run[2], run[1]]).gaps).toEqual([])
  })

  it('does not call the end of a trace that is still coming a gap', () => {
    const run = trace(runStart(), stageStart('translate', 0), modelCall(1))
    expect(buildView(run).gaps).toEqual([])
  })

  it('keeps an event whose neighbours are missing instead of dropping it', () => {
    const run = sentBackTwice()
    const view = buildView(run.filter((e) => e.seq !== 9)) // review 1's stage_end is lost
    expect(view.gaps).toEqual([{ from: 9, to: 9 }])
    expect(placed(view)).toEqual(seqs(run).filter((seq) => seq !== 9))
  })
})

// ---- Grouping by stage ------------------------------------------------------------------------------------------

describe('events grouped by stage', () => {
  it('puts the run start and end outside the stages and each event in its stage', () => {
    const view = buildView(sentBackTwice())
    expect(view.start?.seq).toBe(1)
    expect(view.end?.seq).toBe(22)
    const all = stages(view.blocks)
    expect(view.blocks).toHaveLength(6)
    expect(all.map((s) => [s.start.seq, seqs(s.events), s.end?.seq])).toEqual([
      [2, [3, 4, 5], 6],
      [7, [8], 9],
      [10, [11], 12],
      [13, [14], 15],
      [16, [17], 18],
      [19, [20], 21],
    ])
    expect(placed(view)).toEqual(Array.from({ length: 22 }, (_, n) => n + 1))
  })

  it('numbers reviews and revisions in the order they began', () => {
    const all = stages(buildView(sentBackTwice()).blocks)
    expect(all.map((s) => s.title)).toEqual(['Translate', 'Review 1', 'Revision 1', 'Review 2', 'Revision 2', 'Finalize'])
    expect(all.map((s) => s.number)).toEqual([null, 1, 1, 2, 2, null])
    expect(all.map((s) => s.kind)).toEqual(['translate', 'review', 'revise', 'review', 'revise', 'finalize'])
  })

  it('numbers by order, not by the stage index, so a run approved at once has Review 1 alone', () => {
    const view = buildView(
      trace(runStart(), stageStart('translate', 0), stageEnd('translate', 0), stageStart('review', 1), stageEnd('review', 1, { verdict: 'approve', issues: [] }), runEnd()),
    )
    expect(stages(view.blocks).map((s) => s.title)).toEqual(['Translate', 'Review 1'])
  })

  it('files every kind of event under the stage it happened in', () => {
    const view = buildView(
      trace(
        runStart(), // 1
        stageStart('translate', 0), // 2
        modelCall(1, ['lookup_term']), // 3
        toolResult('lookup_term'), // 4
        repair, // 5
        fallback, // 6
        stageEnd('translate', 0), // 7
        stageStart('review', 1), // 8
        repair, // 9
        stageEnd('review', 1, { verdict: 'approve', issues: [] }), // 10
        runEnd(), // 11
      ),
    )
    const [translate, review] = stages(view.blocks)
    expect(translate.events.map((e) => e.kind)).toEqual(['model_call', 'tool_result', 'repair', 'fallback'])
    expect(translate.end?.kind).toBe('stage_end')
    expect(review.events.map((e) => e.kind)).toEqual(['repair'])
    expect(view.start?.kind).toBe('run_start')
    expect(view.end?.kind).toBe('run_end')
  })

  it('has no stages for a baseline run: its start and end only', () => {
    const view = buildView(trace(runStart('baseline'), runEnd({ calls: 1 })))
    expect(view.blocks).toEqual([])
    expect(view.start?.seq).toBe(1)
    expect(view.end?.seq).toBe(2)
    expect(view.gaps).toEqual([])
  })

  it('shows a run that has not begun, and one that has no events', () => {
    expect(buildView([])).toEqual({ start: null, blocks: [], end: null, gaps: [], count: 0 })
    const started = buildView(trace(runStart()))
    expect(started.start?.seq).toBe(1)
    expect(started.end).toBeNull()
  })

  it('leaves the last stage without an end when the run failed inside it, and keeps run_end outside it', () => {
    const view = buildView(
      trace(
        runStart(), // 1
        stageStart('translate', 0), // 2
        stageEnd('translate', 0), // 3
        stageStart('review', 1), // 4
        modelCall(1), // 5, then the provider failed
        runEnd({ status: 'failed', error: 'Cannot reach the provider' }), // 6
      ),
    )
    const [translate, review] = stages(view.blocks)
    expect(translate.end?.seq).toBe(3)
    expect(review.title).toBe('Review 1')
    expect(review.end).toBeNull()
    expect(seqs(review.events)).toEqual([5])
    expect(view.end?.seq).toBe(6)
    expect(view.end?.data.status).toBe('failed')
  })

  it('shows a stage that is still running as one without an end', () => {
    const view = buildView(trace(runStart(), stageStart('translate', 0), modelCall(1, ['lookup_term'])))
    const [translate] = stages(view.blocks)
    expect(translate.end).toBeNull()
    expect(seqs(translate.events)).toEqual([3])
    expect(view.end).toBeNull()
  })

  it('keeps events that belong to no stage, in order', () => {
    // A run of a config with no stages (a Translator run) has model calls between its start and its end.
    const view = buildView(trace(runStart('translator'), modelCall(1, ['get_previous_context']), toolResult('get_previous_context'), modelCall(2), runEnd()))
    expect(view.blocks.map((b) => (b.type === 'event' ? [b.event.seq, b.event.kind] : 'stage'))).toEqual([
      [2, 'model_call'],
      [3, 'tool_result'],
      [4, 'model_call'],
    ])
  })

  it('loses no event when a stage_end has no stage, or a stage began before the last one ended', () => {
    const lostStart = buildView(trace(runStart(), stageEnd('review', 1, { verdict: 'approve', issues: [] }), runEnd()))
    expect(lostStart.blocks.map((b) => b.type)).toEqual(['event'])
    expect(placed(lostStart)).toEqual([1, 2, 3])

    const lostEnd = buildView(trace(runStart(), stageStart('translate', 0), modelCall(1), stageStart('review', 1), stageEnd('review', 1), runEnd()))
    const [translate, review] = stages(lostEnd.blocks)
    expect(translate.end).toBeNull()
    expect(seqs(translate.events)).toEqual([3])
    expect(review.end?.seq).toBe(5)
    expect(placed(lostEnd)).toEqual([1, 2, 3, 4, 5, 6])
  })

  it('does not close a stage with the end of another one', () => {
    const view = buildView(trace(runStart(), stageStart('review', 1), stageEnd('translate', 0), stageEnd('review', 1)))
    const [review] = stages(view.blocks)
    expect(seqs(review.events)).toEqual([3]) // the stray stage_end is inside, not its end
    expect(review.end?.seq).toBe(4)
  })
})

// ---- Words for the rows ------------------------------------------------------------------------------------------

describe('the words of a row', () => {
  it('titles the stages', () => {
    expect(stageTitle('translate', null)).toBe('Translate')
    expect(stageTitle('review', 2)).toBe('Review 2')
    expect(stageTitle('revise', 1)).toBe('Revision 1')
    expect(stageTitle('finalize', null)).toBe('Finalize')
    expect(stageTitle('curate', null)).toBe('curate') // a stage the panel does not know is shown as it is named
  })

  it('shortens by characters, never in the middle of one', () => {
    expect(shorten('abc', 3)).toEqual({ text: 'abc', cut: false })
    expect(shorten('abcd', 3)).toEqual({ text: 'abc', cut: true })
    expect(shorten('你好吗？', 3)).toEqual({ text: '你好吗', cut: true })
    expect(shorten('a😀b', 2)).toEqual({ text: 'a😀', cut: true }) // 😀 is two UTF-16 units but one character
    expect(shorten('', 5)).toEqual({ text: '', cut: false })
  })

  it('shows arguments on one line, and cut when long', () => {
    expect(shortArguments(null)).toBe('')
    expect(shortArguments('{ "term" :  "コップ", "n": 1 }')).toBe('{"term":"コップ","n":1}')
    expect(shortArguments('not json,\n  as sent')).toBe('not json, as sent')
    const long = JSON.stringify({ reason: 'x'.repeat(200) })
    const shown = shortArguments(long, 40)
    expect(Array.from(shown)).toHaveLength(41)
    expect(shown.endsWith('…')).toBe(true)
  })

  it('says when usage or cost is unknown, never 0', () => {
    expect(usageText(null)).toBe('usage unknown')
    expect(usageText(record({ usage: null }))).toBe('usage unknown')
    expect(usageText(record())).toBe('1000 in / 200 out tokens')
    expect(usageText(record({ usage: { input_tokens: 1000, output_tokens: 200, cached_input_tokens: 800 } }))).toBe('1000 in (800 cached) / 200 out tokens')
    expect(usageText(record({ usage: { input_tokens: 0, output_tokens: 0, cached_input_tokens: null } }))).toBe('0 in / 0 out tokens')
    expect(costText(null)).toBe('cost unknown')
    expect(costText(0)).toBe('$0.00000')
    expect(costText(0.00791)).toBe('$0.00791')
  })

  it('lists only the proposal results that happened', () => {
    expect(nonZeroCounts({ duplicate: 0, accepted: 2, undecided: 1 })).toEqual([['accepted', 2], ['undecided', 1]])
    expect(nonZeroCounts({ accepted: 0 })).toEqual([])
    expect(nonZeroCounts(undefined)).toEqual([])
  })
})
