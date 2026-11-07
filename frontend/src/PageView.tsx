import { useCallback, useEffect, useRef, useState } from 'react'
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
  type TranslateConfig,
} from './api'
import { Link } from './Link'
import { routePath } from './route'
import { TranslationEditor } from './TranslationEditor'

interface Props {
  book: string
  pageIndex: number
  pages: PageSummary[]
  navigate: (path: string) => void
  onTranslated: (book: string, pageIndex: number) => void
}

// One hue per segment, shared by its box on the image and its row in the list.
const hue = (index: number) => `hsl(${(index * 67) % 360} 70% 48%)`

// The row being edited: the text typed so far, and why the last save failed.
interface Editing {
  index: number
  draft: string
  error: string | null
}

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

  // Watch a run until its `run_end`, then show what it left on the page. The run goes on in the backend if this view goes away.
  const watch = useCallback(
    (runId: number) => {
      stopWatching.current?.()
      setTranslating(true)
      stopWatching.current = watchRun(runId, async (end: RunEnd | null) => {
        stopWatching.current = null
        try {
          const page = await fetchPage(book, pageIndex)
          setDetail(page)
          if (page.run) onTranslated(book, pageIndex)
          if (end?.status === 'failed') setTranslateError(end.error ?? 'the run failed')
        } catch (e) {
          setTranslateError((e as Error).message)
        } finally {
          setTranslating(false)
        }
      })
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
        if (page.running_run_id !== null) watch(page.running_run_id) // a reload: the page goes on waiting for its run
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
  }, [book, pageIndex, watch])

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

  const siblings = pages.filter((p) => p.book_title === book)
  const at = siblings.findIndex((p) => p.page_index === pageIndex)
  const prev = at > 0 ? siblings[at - 1] : null
  const next = at >= 0 && at < siblings.length - 1 ? siblings[at + 1] : null

  const segments = detail?.segments ?? []
  const label = `${book}/${String(pageIndex).padStart(3, '0')}`

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
      {!detail && !loadError && <p className="muted">Loading…</p>}

      {detail && (
        <div className="columns">
          <div className="stage">
            <img
              src={detail.image_url}
              alt={label}
              onLoad={(e) => setNatural({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            />
            {natural &&
              segments.map((s) => (
                <div
                  key={s.index}
                  className={`box${active === s.index ? ' active' : ''}`}
                  style={{
                    left: `${(s.x / natural.w) * 100}%`,
                    top: `${(s.y / natural.h) * 100}%`,
                    width: `${(s.w / natural.w) * 100}%`,
                    height: `${(s.h / natural.h) * 100}%`,
                    ['--c' as string]: hue(s.index),
                  }}
                  onMouseEnter={() => setActive(s.index)}
                  onMouseLeave={() => setActive(null)}
                >
                  <span className="badge">{s.index}</span>
                </div>
              ))}
          </div>

          <ol className="segments">
            {segments.map((s) => (
              <li
                key={s.index}
                className={active === s.index ? 'active' : undefined}
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
                  <p className="ref" lang="zh-Hant">
                    <span className="tag">zh-Hant reference</span> {s.text_zh_hant}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}
