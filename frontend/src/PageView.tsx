import { useEffect, useState } from 'react'
import { fetchPage, translatePage, type PageDetail, type PageSummary, type Run } from './api'
import { Link } from './Link'
import { routePath } from './route'

interface Props {
  book: string
  pageIndex: number
  pages: PageSummary[]
  navigate: (path: string) => void
  onTranslated: (book: string, pageIndex: number) => void
}

// One hue per segment, shared by its box on the image and its row in the list.
const hue = (index: number) => `hsl(${(index * 67) % 360} 70% 48%)`

function runSummary(run: Run): string {
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
  const [translating, setTranslating] = useState(false)
  const [translateError, setTranslateError] = useState<string | null>(null)
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [active, setActive] = useState<number | null>(null)

  // The parent keys this component by page, so every page starts from fresh state.
  useEffect(() => {
    const ctrl = new AbortController()
    fetchPage(book, pageIndex, ctrl.signal).then(setDetail, (e: Error) => {
      if (!ctrl.signal.aborted) setLoadError(e.message)
    })
    return () => ctrl.abort()
  }, [book, pageIndex])

  const onTranslate = async () => {
    setTranslating(true)
    setTranslateError(null)
    try {
      setDetail(await translatePage(book, pageIndex))
      onTranslated(book, pageIndex)
    } catch (e) {
      setTranslateError((e as Error).message)
    } finally {
      setTranslating(false)
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
        <button onClick={onTranslate} disabled={translating || !detail || segments.length === 0}>
          {translating ? 'Translating…' : detail?.run ? 'Translate again' : 'Translate'}
        </button>
      </header>

      {loadError && <p className="error">{loadError}</p>}
      {translateError && <p className="error" role="alert">Translation failed: {translateError}</p>}
      {translating && <p className="muted">Waiting for the model. This usually takes a few seconds.</p>}
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
                  <p className={s.text_zh_hans ? 'out' : 'out empty'} lang="zh-Hans">
                    {s.text_zh_hans ?? (translating ? '…' : 'not translated yet')}
                  </p>
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
