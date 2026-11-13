import { useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import type { ModelCallData, RepairData, RunEnd, StageEndData, ToolResultData, TraceEvent } from './api'
import { hue } from './hue'
import {
  buildView,
  costText,
  describeGaps,
  nonZeroCounts,
  shortArguments,
  shorten,
  stageTitle,
  usageText,
  type Stage,
} from './liveTrace'

const RESULT_CHARS = 200 // a tool result longer than this is cut, with a control to show all of it

interface Props {
  events: readonly TraceEvent[] // as received, in any order and with repeats: `buildView` sorts them out
  open: boolean // the stream is still open
  live: boolean // the run is being waited for; false: the trace of a finished run is replayed
}

// What a run did, step by step (OUTLINE §0, Live trace event): one row per event, each with its seq, the stages as groups.
export function TracePanel({ events, open, live }: Props) {
  const view = useMemo(() => buildView(events), [events])
  const [shownAll, setShownAll] = useState<ReadonlySet<number>>(new Set()) // seqs of the tool results shown in full
  const toggle = (seq: number) =>
    setShownAll((current) => {
      const next = new Set(current)
      if (!next.delete(seq)) next.add(seq)
      return next
    })

  // While the run is live, the newest row stays in view unless the reader scrolled up.
  const log = useRef<HTMLDivElement>(null)
  const following = useRef(true)
  useLayoutEffect(() => {
    const el = log.current
    if (live && el && following.current) el.scrollTop = el.scrollHeight
  }, [view.count, live])
  const onScroll = () => {
    const el = log.current
    if (el) following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24
  }

  const row = (event: TraceEvent, stage?: Stage) => (
    <EventRow key={event.seq} event={event} stage={stage} shownAll={shownAll.has(event.seq)} onToggle={() => toggle(event.seq)} />
  )

  return (
    <section className="trace" aria-label="Live trace">
      <header className="trace-head">
        <h3>Live trace</h3>
        <span className="muted">
          {[
            view.count > 0 ? `${view.count} ${view.count === 1 ? 'event' : 'events'}` : null,
            open && !view.end ? (live ? 'running' : 'loading') : null,
          ]
            .filter((part) => part !== null)
            .join(' · ')}
        </span>
      </header>
      {view.gaps.length > 0 && (
        <p className="error trace-gap" role="alert">
          {view.gaps.length === 1 && view.gaps[0].from === view.gaps[0].to ? 'Event' : 'Events'} {describeGaps(view.gaps)}{' '}
          never arrived, so this trace has holes. This should not happen.
        </p>
      )}
      <div
        ref={log}
        className="trace-log"
        role="log"
        aria-label="Events of the run"
        aria-live={live ? 'polite' : 'off'}
        tabIndex={0}
        onScroll={onScroll}
      >
        {view.count === 0 ? (
          <p className="muted">{open ? 'Waiting for the first event…' : 'No live trace was recorded for this Run.'}</p>
        ) : (
          <ol className="trace-list">
            {view.start && row(view.start)}
            {view.blocks.map((block) =>
              block.type === 'event' ? (
                row(block.event)
              ) : (
                <li key={block.stage.start.seq} className="trace-stage">
                  <ol className="trace-list">
                    {row(block.stage.start, block.stage)}
                    {block.stage.events.map((event) => row(event))}
                    {block.stage.end && row(block.stage.end, block.stage)}
                  </ol>
                </li>
              ),
            )}
            {view.end && row(view.end)}
          </ol>
        )}
        {view.count > 0 && !open && !view.end && (
          <p className="muted">The stream ended without the run's end event.</p>
        )}
      </div>
    </section>
  )
}

function Row({ event, children }: { event: TraceEvent; children: ReactNode }) {
  return (
    <li className={`trace-row kind-${event.kind}`}>
      <span className="seq" title={`seq ${event.seq}`}>{event.seq}</span>
      <div className="row-body">{children}</div>
    </li>
  )
}

// A region id, in the colour of its box and its row in the list.
const Region = ({ id }: { id: number }) => (
  <span className="badge" style={{ ['--c' as string]: hue(id) }} title={`region ${id}`}>{id}</span>
)

const Tag = ({ tone, children }: { tone?: 'ok' | 'bad'; children: ReactNode }) => (
  <span className={`tag${tone ? ` ${tone}` : ''}`}>{children}</span>
)
const statusTone = (status: string) => (status === 'ok' || status === 'succeeded' ? 'ok' : status === 'failed' ? 'bad' : undefined)
const ErrorText = ({ children }: { children: ReactNode }) => <p className="trace-error">{children}</p>

function EventRow({ event, stage, shownAll, onToggle }: { event: TraceEvent; stage?: Stage; shownAll: boolean; onToggle: () => void }) {
  switch (event.kind) {
    case 'run_start': {
      const { run_id, config, model } = event.data
      return (
        <Row event={event}>
          <strong>Run {run_id}</strong> <span className="muted">· {config} · {model}</span>
        </Row>
      )
    }
    case 'stage_start':
      return (
        <Row event={event}>
          <strong className="stage-title">{stage?.title ?? stageTitle(event.data.stage, null)}</strong>
        </Row>
      )
    case 'model_call':
      return <Row event={event}><ModelCall data={event.data} /></Row>
    case 'tool_result':
      return <Row event={event}><ToolResult data={event.data} shownAll={shownAll} onToggle={onToggle} /></Row>
    case 'repair':
      return <Row event={event}><Repair data={event.data} /></Row>
    case 'fallback':
      return (
        <Row event={event}>
          <strong>Fallback</strong> <span className="muted">to baseline: {event.data.reason}</span>
        </Row>
      )
    case 'stage_end':
      return <Row event={event}><StageEnd data={event.data} title={stage?.title ?? stageTitle(event.data.stage, null)} /></Row>
    case 'run_end':
      return <Row event={event}><RunEndRow data={event.data} /></Row>
  }
}

function ModelCall({ data }: { data: ModelCallData }) {
  const facts = [
    data.finish_reason,
    usageText(data.record),
    data.record ? `${data.record.latency_s.toFixed(1)} s` : null,
    data.reasoning_chars !== null ? `reasoning ${data.reasoning_chars} chars` : null,
  ].filter((fact) => fact !== null)
  return (
    <>
      <strong>Model call {data.step}</strong> <span className="muted">· {facts.join(' · ')}</span>
      {data.tool_calls.length > 0 && (
        <ul className="trace-calls">
          {data.tool_calls.map((call) => (
            <li key={call.id}>
              <code>{call.name}</code> <code className="args">{shortArguments(call.arguments)}</code>
            </li>
          ))}
        </ul>
      )}
      {data.error && <ErrorText>{data.error}</ErrorText>}
    </>
  )
}

function ToolResult({ data, shownAll, onToggle }: { data: ToolResultData; shownAll: boolean; onToggle: () => void }) {
  const text = data.error ?? data.result ?? ''
  const { text: head, cut } = shorten(text, RESULT_CHARS)
  return (
    <>
      <strong>Result of <code>{data.name}</code></strong>
      {!data.ran && <> <Tag tone="bad">did not run</Tag></>}
      {data.duplicate && <> <Tag tone="bad">duplicate</Tag></>}
      <p className={data.error !== null ? 'trace-error' : 'trace-result'}>
        {shownAll || !cut ? text : `${head}…`}
      </p>
      {cut && (
        <button type="button" className="link-button" onClick={onToggle} aria-expanded={shownAll}>
          {shownAll ? 'Show less' : `Show all ${Array.from(text).length} characters`}
        </button>
      )}
    </>
  )
}

function Repair({ data }: { data: RepairData }) {
  return (
    <>
      <strong>Repair</strong> <span className="muted">· the answer could not be used: {data.parse_error}</span>
      {data.error && <ErrorText>The repair call failed: {data.error}</ErrorText>}
    </>
  )
}

function StageEnd({ data, title }: { data: StageEndData; title: string }) {
  const counts = nonZeroCounts(data.results)
  return (
    <>
      <strong>{title} ended</strong> <Tag tone={statusTone(data.status)}>{data.stage === 'finalize' ? `Curator ${data.status}` : data.status}</Tag>
      {data.verdict !== undefined && <> <Tag tone={data.verdict === 'approve' ? 'ok' : undefined}>verdict: {data.verdict ?? 'none'}</Tag></>}
      {data.error && <ErrorText>{data.error}</ErrorText>}
      {data.issues && data.issues.length > 0 && (
        <ul className="trace-notes" aria-label="Issues">
          {data.issues.map((issue) => (
            <li key={issue.id}>
              <Region id={issue.id} /> <span lang="zh-Hans">{issue.comment}</span>
              {issue.suggestion && <> <span className="muted">→</span> <span lang="zh-Hans" className="suggestion">{issue.suggestion}</span></>}
            </li>
          ))}
        </ul>
      )}
      {data.flags && data.flags.length > 0 && (
        <ul className="trace-notes" aria-label="Flags">
          {data.flags.map((flag) => (
            <li key={flag.id}>
              <span className="muted">unsure</span> <Region id={flag.id} /> {flag.reason}
            </li>
          ))}
        </ul>
      )}
      {counts.length > 0 && (
        <p className="muted">proposals: {counts.map(([result, n]) => `${result} ${n}`).join(' · ')}</p>
      )}
    </>
  )
}

function RunEndRow({ data }: { data: RunEnd }) {
  const tokens =
    data.input_tokens === null || data.output_tokens === null
      ? 'usage unknown'
      : `${data.input_tokens} in / ${data.output_tokens} out tokens`
  const facts = [
    `${data.calls} ${data.calls === 1 ? 'call' : 'calls'}`,
    tokens,
    costText(data.cost_usd),
    data.unknown_usage_calls > 0 ? `${data.unknown_usage_calls} without usage` : null,
    data.send_backs !== undefined ? `${data.send_backs} ${data.send_backs === 1 ? 'send-back' : 'send-backs'}` : null,
    data.final_stop !== undefined ? `stop: ${data.final_stop}` : null,
    data.final_reviewed !== undefined ? (data.final_reviewed ? 'final translation reviewed' : 'final translation not reviewed') : null,
  ].filter((fact) => fact !== null)
  return (
    <>
      <strong>Run ended</strong> <Tag tone={statusTone(data.status)}>{data.status}</Tag>
      {data.error && <ErrorText>{data.error}</ErrorText>}
      <p className="muted">{facts.join(' · ')}</p>
    </>
  )
}
