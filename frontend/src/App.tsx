import { useCallback, useEffect, useState } from 'react'
import { fetchPages, type PageSummary, type RenamedPage } from './api'
import { BookTypesetting } from './BookTypesetting'
import { KBPanel } from './KBPanel'
import { PageList } from './PageList'
import { PageView, type RenameSignal } from './PageView'
import { useRoute } from './route'

export default function App() {
  const [route, navigate] = useRoute()
  const [pages, setPages] = useState<PageSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [runsEnded, setRunsEnded] = useState(0) // a Run that ended in the editor may have written the book's KB: the panel reads it again
  const [renamed, setRenamed] = useState<RenameSignal>({ n: 0, book: '', pages: [] }) // the last rename: the open page reads again if it was written

  useEffect(() => {
    fetchPages().then(setPages, (e: Error) => setError(e.message))
  }, [])

  const markTranslated = useCallback((book: string, pageIndex: number) => {
    setPages((all) =>
      all && all.map((p) => (p.book_title === book && p.page_index === pageIndex ? { ...p, translated: true } : p)),
    )
  }, [])

  // Stable, like markTranslated: PageView's `watch` depends on it, and a new one would watch the page's Run again.
  const countRunEnd = useCallback(() => setRunsEnded((n) => n + 1), [])

  // The KB panel's rename wrote a Human edit on each of these pages of the book. Stable, like the two above: the panel's requests close over it.
  const noteRename = useCallback(
    (book: string, runs: readonly RenamedPage[]) =>
      setRenamed((last) => ({ n: last.n + 1, book, pages: runs.map((run) => run.page_index) })),
    [],
  )

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>Fukidashi</h1>
        {error && <p className="error">{error}</p>}
        {!error && !pages && <p className="muted">Loading pages…</p>}
        {pages?.length === 0 && (
          <p className="muted">
            No pages yet. Run <code>uv run python scripts/import_openmantra.py</code>, then reload.
          </p>
        )}
        {pages && <PageList pages={pages} route={route} navigate={navigate} />}
      </aside>
      <main className="main">
        {/* Keyed by book, not by page: a book's measurement stays on screen while the reader moves between its pages. */}
        {route && pages && <BookTypesetting key={route.book} book={route.book} pages={pages} navigate={navigate} />}
        {/* Keyed by book too: the KB it loaded and whether the reader opened it stay while the reader moves between the book's pages.
            Its own key, not the book's: siblings with one key are not unique, and React says so on every render. */}
        {route && (
          <KBPanel key={`kb-panel:${route.book}`} book={route.book} navigate={navigate} runsEnded={runsEnded} onRenamed={noteRename} />
        )}
        {route ? (
          <PageView
            key={`${route.book}/${route.pageIndex}`}
            book={route.book}
            pageIndex={route.pageIndex}
            pages={pages ?? []}
            navigate={navigate}
            onTranslated={markTranslated}
            onRunEnd={countRunEnd}
            renamed={renamed}
          />
        ) : (
          <p className="muted">Pick a page on the left.</p>
        )}
      </main>
    </div>
  )
}
