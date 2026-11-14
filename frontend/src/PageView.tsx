import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  editTranslation,
  fetchPage,
  translatePage,
  watchRun,
  type PageDetail,
  type PageSummary,
  type Run,
  type RunEnd,
  type Segment,
  type TraceEvent,
  type TranslateConfig,
} from './api'
import { measureText } from './canvasText'
import { hue } from './hue'
import { Link } from './Link'
import { addEvent } from './liveTrace'
import { NoteList, NoteMarks, ReviewLine } from './PageNotes'
import { describeRegion, groupNotes } from './reviewNotes'
import { routePath } from './route'
import { TracePanel } from './TracePanel'
import { TranslationEditor } from './TranslationEditor'
import { MIN_FONT_PX, describeFigures, overflowRate, typesetPage } from './typeset'

// Konva and react-konva are most of the bundle, and a page needs them only once it has a translation to draw: the canvas
// layer is its own chunk, loaded the first time it shows.
const TypesetLayer = lazy(() => import('./TypesetLayer').then((m) => ({ default: m.TypesetLayer })))

interface Props {
  book: string
  pageIndex: number
  pages: PageSummary[]
  navigate: (path: string) => void
  onTranslated: (book: string, pageIndex: number) => void
}

// The row being edited: the text typed so far, and why the last save failed.
interface Editing {
  index: number
  draft: string
  error: string | null
}

// The run whose trace the panel shows. `events` are those received so far (a seq is held once); `open`: the stream is
// still open; `live`: the run is being waited for, not a finished run's trace replayed.
interface Followed {
  runId: number
  events: readonly TraceEvent[]
  open: boolean
  live: boolean
}

const isEditorConfig = (config: string): config is TranslateConfig => config === 'baseline' || config === 'full'

const NO_SEGMENTS: Segment[] = [] // one array for every render without a page, so the typesetting below is not redone for it

const when = (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'medium' })

function runSummary(run: Run): string {
  // A person's edit is a run with no call: no tokens and no cost to show.
  if (run.config === 'human') return `human edit · ${when(run.finished_at ?? run.started_at)}`
  const parts = [run.config, run.response_model ?? run.model]
  parts.push(
    run.input_tokens === null || run.output_tokens === null
      ? 'usage unknown'
      : `${run.input_tokens} in / ${run.output_tokens} out tokens`,
  )
  parts.push(run.cost_usd === null ? 'cost unknown' : `$${run.cost_usd.toFixed(5)}`)
  if (run.finished_at) {
    parts.push(`${Math.round((Date.parse(run.finished_at) - Date.parse(run.started_at)) / 1000)} s`)
  }
  return parts.join(' · ')
}

export function PageView({ book, pageIndex, pages, navigate, onTranslated }: Props) {
  const [detail, setDetail] = useState<PageDetail | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [translating, setTranslating] = useState(false) // a run of this page is being watched, started here or found on load
  const [translateError, setTranslateError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [config, setConfig] = useState<TranslateConfig>('baseline')
  const stopWatching = useRef<(() => void) | null>(null)
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [active, setActive] = useState<number | null>(null)
  const [editing, setEditing] = useState<Editing | null>(null) // one row at a time
  const [saving, setSaving] = useState(false)
  const [followed, setFollowed] = useState<Followed | null>(null) // the trace on show: it stays until the next Translate
  const [original, setOriginal] = useState(false) // the reader asked for the original: the typeset translations are hidden
  const [selected, setSelected] = useState<number | null>(null) // the region whose box was clicked: it stays until cleared
  const rows = useRef(new Map<number, HTMLLIElement>()) // the rows of the list by region, to scroll the selected one into view

  // Follow a run's trace. With `live` (the default) also wait for its `run_end`, then show what it left on the page: the
  // run goes on in the backend if this view goes away. Without it, replay the trace of a run that is over, and nothing else.
  const watch = useCallback(
    (runId: number, live = true) => {
      stopWatching.current?.()
      // The same run again (a stream opened anew replays it from seq 1) keeps the events it has; another run starts empty.
      setFollowed((f) => ({ runId, events: f?.runId === runId ? f.events : [], open: true, live }))
      if (live) setTranslating(true)
      stopWatching.current = watchRun(
        runId,
        (event) => {
          setFollowed((f) => {
            if (f === null || f.runId !== runId) return f
            const events = addEvent(f.events, event)
            return events === f.events ? f : { ...f, events }
          })
          // After a reload the select shows the config of the run being waited for, as that run's `run_start` says.
          if (live && event.kind === 'run_start' && isEditorConfig(event.data.config)) setConfig(event.data.config)
        },
        async (end: RunEnd | null) => {
          stopWatching.current = null
          setFollowed((f) => (f !== null && f.runId === runId ? { ...f, open: false } : f))
          if (!live) return
          setNotice(null) // the run is over: "already being translated" is no longer true
          try {
            const page = await fetchPage(book, pageIndex)
            setDetail(page)
            if (page.run) onTranslated(book, pageIndex)
            if (end?.status === 'failed') setTranslateError(end.error ?? 'the run failed')
            // The stream ended without a run_end (the server restarted mid-run, say): the page cannot say whether it failed.
            else if (end === null) setNotice('The run ended without reporting its result. The page shows its latest translation.')
          } catch (e) {
            setTranslateError((e as Error).message)
          } finally {
            setTranslating(false)
          }
        },
      )
    },
    [book, pageIndex, onTranslated],
  )

  // The parent keys this component by page, so every page starts from fresh state.
  useEffect(() => {
    const ctrl = new AbortController()
    fetchPage(book, pageIndex, ctrl.signal).then(
      (page) => {
        if (ctrl.signal.aborted) return
        setDetail(page)
        if (page.run) onTranslated(book, pageIndex) // a run that ended while this page was off screen still marks it
        if (page.running_run_id !== null) watch(page.running_run_id) // a reload: the page goes on waiting for its run
        else if (page.run) watch(page.run.id, false) // else the trace of the run on show, if one was recorded
      },
      (e: Error) => {
        if (!ctrl.signal.aborted) setLoadError(e.message)
      },
    )
    return () => {
      ctrl.abort()
      stopWatching.current?.()
      stopWatching.current = null
    }
  }, [book, pageIndex, watch, onTranslated])

  const onTranslate = async () => {
    setTranslating(true)
    setTranslateError(null)
    setNotice(null)
    try {
      watch((await translatePage(book, pageIndex, config)).run_id)
    } catch (e) {
      setTranslating(false)
      if (e instanceof ApiError && e.status === 409) {
        setNotice('This page is already being translated.')
        fetchPage(book, pageIndex).then((page) => {
          setDetail(page)
          if (page.running_run_id !== null) watch(page.running_run_id)
          else setNotice(null) // the run ended between the 409 and this read
        }, () => {})
      } else {
        setTranslateError((e as Error).message)
      }
    }
  }

  const startEdit = (s: Segment) => setEditing({ index: s.index, draft: s.text_zh_hans ?? '', error: null })

  const onSave = async () => {
    if (!editing || saving) return
    const { index, draft } = editing
    setSaving(true)
    setEditing({ ...editing, error: null })
    try {
      setDetail(await editTranslation(book, pageIndex, index, draft))
      setEditing(null)
    } catch (e) {
      setEditing((current) => current && { ...current, error: (e as Error).message }) // the draft stays
    } finally {
      setSaving(false)
    }
  }

  // Clicking a box selects its region: the box and its row are shown selected, and the row scrolls into view. The selection stays
  // until another box is clicked; clicking the selected box again, or pressing Esc, clears it.
  const select = (index: number) => {
    if (selected === index) {
      setSelected(null)
      return
    }
    setSelected(index)
    const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    rows.current.get(index)?.scrollIntoView({ block: 'center', behavior: calm ? 'auto' : 'smooth' })
  }
  useEffect(() => {
    if (selected === null) return
    // An editor's own Esc (cancel) has handled the key already, and an input method's Esc is not ours.
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !e.defaultPrevented && !e.isComposing) setSelected(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selected])

  const siblings = pages.filter((p) => p.book_title === book)
  const at = siblings.findIndex((p) => p.page_index === pageIndex)
  const prev = at > 0 ? siblings[at - 1] : null
  const next = at >= 0 && at < siblings.length - 1 ? siblings[at + 1] : null

  const segments = detail?.segments ?? NO_SEGMENTS
  const label = `${book}/${String(pageIndex).padStart(3, '0')}`
  // Typeset again whenever the displayed translations change: an edit saved, a run that ended, another page (OUTLINE §8).
  const results = useMemo(() => typesetPage(segments, measureText), [segments])
  const figures = useMemo(() => overflowRate(results), [results])
  const showTranslated = figures.typeset > 0 && !original // the default view is the translated one, once there is one
  const review = detail?.review ?? null
  const notes = useMemo(() => groupNotes(review), [review]) // the review's notes by region

  return (
    <div className="page-view">
      <header className="page-header">
        <h2>{label}</h2>
        <div className="pager">
          {prev && <Link to={routePath(book, prev.page_index)} navigate={navigate}>← {prev.page_index}</Link>}
          {next && <Link to={routePath(book, next.page_index)} navigate={navigate}>{next.page_index} →</Link>}
        </div>
        <select
          value={config}
          onChange={(e) => setConfig(e.target.value as TranslateConfig)}
          disabled={translating}
          aria-label="Configuration to translate with"
        >
          <option value="baseline">baseline</option>
          <option value="full">full</option>
        </select>
        <button onClick={onTranslate} disabled={translating || saving || !detail || segments.length === 0}>
          {translating ? 'Translating…' : detail?.run ? 'Translate again' : 'Translate'}
        </button>
      </header>

      {loadError && <p className="error">{loadError}</p>}
      {translateError && <p className="error" role="alert">Translation failed: {translateError}</p>}
      {notice && <p className="muted" role="status">{notice}</p>}
      {translating && <p className="muted">Waiting for the model. The page updates when the run ends.</p>}
      {detail && segments.length === 0 && <p className="muted">This page has no text regions.</p>}
      {detail?.run && <p className="muted run-info">{runSummary(detail.run)}</p>}
      {review && <ReviewLine review={review} />}
      {detail && segments.length > 0 && (
        <div className="typeset-bar">
          <p
            className="muted typeset-line"
            title={`A box overflows when its translation does not fit at ${MIN_FONT_PX} px, in image pixels. Boxes with no translation are left out of the rate.`}
          >
            Typesetting: {describeFigures(figures)}
          </p>
          <div className="view-toggle" role="group" aria-label="What the page shows">
            <button type="button" aria-pressed={showTranslated} disabled={figures.typeset === 0} onClick={() => setOriginal(false)}>
              Translated
            </button>
            <button type="button" aria-pressed={!showTranslated} onClick={() => setOriginal(true)}>
              Original
            </button>
          </div>
        </div>
      )}
      {!detail && !loadError && <p className="muted">Loading…</p>}

      {detail && (
        <div className={followed ? 'columns with-trace' : 'columns'}>
          <div className="stage">
            <img
              src={detail.image_url}
              alt={label}
              onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            />
            {natural && showTranslated && (
              <Suspense fallback={null}>
                <TypesetLayer segments={segments} results={results} natural={natural} />
              </Suspense>
            )}
            {natural &&
              segments.map((s, i) => {
                const overflow = results[i]?.overflow
                const regionNotes = notes.get(s.index)
                const words = regionNotes && describeRegion(regionNotes)
                return (
                  <div
                    key={s.index}
                    className={`box${active === s.index ? ' active' : ''}${selected === s.index ? ' selected' : ''}${overflow ? ' overflow' : ''}`}
                    role="button"
                    tabIndex={0}
                    aria-pressed={selected === s.index}
                    aria-label={[`Region ${s.index}`, overflow ? 'overflows' : null, words].filter((part) => part).join(', ')}
                    title={[overflow ? `Overflows: its translation does not fit at ${MIN_FONT_PX} px` : null, words].filter((part) => part).join(' · ') || undefined}
                    style={{
                      left: `${(s.x / natural.w) * 100}%`,
                      top: `${(s.y / natural.h) * 100}%`,
                      width: `${(s.w / natural.w) * 100}%`,
                      height: `${(s.h / natural.h) * 100}%`,
                      ['--c' as string]: hue(s.index),
                    }}
                    onMouseEnter={() => setActive(s.index)}
                    onMouseLeave={() => setActive(null)}
                    onClick={() => select(s.index)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        select(s.index)
                      }
                    }}
                  >
                    <span className="badge">{s.index}</span>
                    {regionNotes && <NoteMarks notes={regionNotes} />}
                  </div>
                )
              })}
          </div>

          <ol className="segments">
            {segments.map((s, i) => (
              <li
                key={s.index}
                ref={(row) => {
                  if (row) rows.current.set(s.index, row)
                  else rows.current.delete(s.index)
                }}
                className={[active === s.index ? 'active' : '', selected === s.index ? 'selected' : ''].filter((c) => c).join(' ') || undefined}
                aria-current={selected === s.index ? 'true' : undefined}
                style={{ ['--c' as string]: hue(s.index) }}
                onMouseEnter={() => setActive(s.index)}
                onMouseLeave={() => setActive(null)}
              >
                <span className="badge">{s.index}</span>
                <div className="texts">
                  <p className="ja" lang="ja">{s.text_ja}</p>
                  {editing?.index === s.index ? (
                    <TranslationEditor
                      index={s.index}
                      draft={editing.draft}
                      error={editing.error}
                      saving={saving}
                      disabled={translating}
                      onChange={(draft) => setEditing((current) => current && { ...current, draft })}
                      onSave={onSave}
                      onCancel={() => setEditing(null)}
                    />
                  ) : (
                    <div className="out-line">
                      <p className={s.text_zh_hans ? 'out' : 'out empty'} lang="zh-Hans">
                        {s.text_zh_hans ?? (translating ? '…' : 'not translated yet')}
                      </p>
                      {results[i]?.overflow && (
                        <span className="tag bad" title={`Does not fit its box at ${MIN_FONT_PX} px`}>
                          overflows
                        </span>
                      )}
                      {s.text_zh_hans !== null && (
                        <button
                          type="button"
                          className="secondary"
                          onClick={() => startEdit(s)}
                          disabled={translating || saving || editing !== null}
                          aria-label={`Edit the translation of region ${s.index}`}
                        >
                          Edit
                        </button>
                      )}
                    </div>
                  )}
                  <NoteList region={s.index} notes={notes.get(s.index)?.notes ?? []} />
                  <p className="ref" lang="zh-Hant">
                    <span className="tag">zh-Hant reference</span> {s.text_zh_hant}
                  </p>
                </div>
              </li>
            ))}
          </ol>

          {followed && <TracePanel key={followed.runId} events={followed.events} open={followed.open} live={followed.live} />}
        </div>
      )}
    </div>
  )
}
