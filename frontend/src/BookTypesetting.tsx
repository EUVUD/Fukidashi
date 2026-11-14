import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchPage, type PageSummary } from './api'
import { measureText } from './canvasText'
import { Link } from './Link'
import { routePath } from './route'
import { MIN_FONT_PX, describeFigures, summarizeBook, typesetPage, type BookFigures } from './typeset'

interface Props {
  book: string
  pages: readonly PageSummary[] // every page of every book, as the page list has them
  navigate: (path: string) => void
}

type State =
  | { phase: 'idle' }
  | { phase: 'measuring'; done: number; total: number }
  | { phase: 'failed'; error: string }
  | { phase: 'done'; summary: BookFigures; at: Date }

// The book's overflow rate (OUTLINE §0, Overflow rate; §8, "Typesetting and the overflow rate"). One click fetches every page
// of the book, typesets what each page displays with the code and the measure the page itself uses, and shows the pooled
// figures and where to look. It calls no model and writes nothing. The result is a snapshot of that moment, kept while the
// reader moves between the book's pages (the parent keys this component by book) and measured again on the next click.
export function BookTypesetting({ book, pages, navigate }: Props) {
  const indexes = useMemo(() => pages.filter((p) => p.book_title === book).map((p) => p.page_index), [pages, book])
  const [state, setState] = useState<State>({ phase: 'idle' })
  const running = useRef<AbortController | null>(null)
  useEffect(() => () => running.current?.abort(), []) // leaving the book stops a measuring that is under way

  const measure = async () => {
    const ctrl = new AbortController()
    running.current = ctrl
    let at = 0 // the page being read, for the error message
    setState({ phase: 'measuring', done: 0, total: indexes.length })
    try {
      const measured = []
      for (const pageIndex of indexes) {
        at = pageIndex
        const page = await fetchPage(book, pageIndex, ctrl.signal)
        measured.push({ pageIndex, results: typesetPage(page.segments, measureText) })
        setState({ phase: 'measuring', done: measured.length, total: indexes.length })
      }
      setState({ phase: 'done', summary: summarizeBook(measured), at: new Date() })
    } catch (e) {
      // Nothing is shown for a book that was only partly read: its figures would say less than the book.
      if (!ctrl.signal.aborted) setState({ phase: 'failed', error: `Page ${at} could not be read: ${(e as Error).message}` })
    }
  }

  const measuring = state.phase === 'measuring'
  return (
    <section className="book-typeset" aria-label="Book typesetting">
      <div className="book-typeset-head">
        <h3>Typesetting of {book}</h3>
        <button type="button" onClick={measure} disabled={measuring || indexes.length === 0}>
          {measuring ? `Measuring… ${state.done} of ${state.total} pages` : 'Measure the book'}
        </button>
      </div>
      {state.phase === 'idle' && (
        <p className="muted">Fetches every page of the book and typesets what it displays. No model is called.</p>
      )}
      {state.phase === 'failed' && <p className="error" role="alert">{state.error}</p>}
      {/* Always there, so a reader's screen reader announces the result when it arrives. */}
      <div aria-live="polite">
        {state.phase === 'done' && <Result book={book} summary={state.summary} at={state.at} navigate={navigate} />}
      </div>
    </section>
  )
}

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many)

function Result({ book, summary, at, navigate }: { book: string; summary: BookFigures; at: Date; navigate: (path: string) => void }) {
  const overflowing = summary.pages.filter((p) => p.figures.overflowing > 0)
  const link = (pageIndex: number, title: string, overflows = false) => (
    <Link key={pageIndex} to={routePath(book, pageIndex)} navigate={navigate} className={overflows ? 'chip overflowing' : 'chip'} title={title}>
      {pageIndex}
    </Link>
  )
  return (
    <div className="book-result">
      <p className="book-figures">
        <strong>{describeFigures(summary.figures)}</strong>
      </p>
      <p className="muted">
        {summary.pages.length} {plural(summary.pages.length, 'page', 'pages')} measured at {at.toLocaleTimeString()}, a snapshot: measure again after
        an edit. A box overflows when its translation does not fit at {MIN_FONT_PX} px.
      </p>
      <div className="book-list">
        <span>Pages with an overflowing box ({overflowing.length}){overflowing.length === 0 && ': none'}</span>
        {overflowing.length > 0 && (
          <div className="chips">
            {overflowing.map((p) =>
              link(p.pageIndex, `${p.figures.overflowing} of ${p.figures.typeset} ${plural(p.figures.typeset, 'box', 'boxes')} overflow`, true),
            )}
          </div>
        )}
      </div>
      <div className="book-list">
        <span>Pages with regions but no translation ({summary.untranslatedPages.length}){summary.untranslatedPages.length === 0 && ': none'}</span>
        {summary.untranslatedPages.length > 0 && (
          <div className="chips">
            {summary.untranslatedPages.map((pageIndex) => {
              const p = summary.pages.find((page) => page.pageIndex === pageIndex)
              return link(pageIndex, `${p?.regions ?? 0} ${plural(p?.regions ?? 0, 'box', 'boxes')}, none translated`)
            })}
          </div>
        )}
      </div>
      <details>
        <summary>Per page</summary>
        <table className="book-pages">
          <thead>
            <tr>
              <th scope="col">Page</th>
              <th scope="col" title="Boxes on the page">Boxes</th>
              <th scope="col" title="Boxes with a translation, typeset">Typeset</th>
              <th scope="col" title={`Typeset boxes that do not fit at ${MIN_FONT_PX} px`}>Overflow</th>
              <th scope="col" title="Boxes with no translation">Not translated</th>
            </tr>
          </thead>
          <tbody>
            {summary.pages.map((p) => (
              <tr key={p.pageIndex}>
                <th scope="row">
                  <Link to={routePath(book, p.pageIndex)} navigate={navigate}>{p.pageIndex}</Link>
                </th>
                <td>{p.regions}</td>
                <td>{p.figures.typeset}</td>
                <td>{p.figures.overflowing}</td>
                <td>{p.figures.untranslated}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  )
}
