import { useCallback, useEffect, useState } from 'react'
import { fetchPages, type PageSummary } from './api'
import { BookTypesetting } from './BookTypesetting'
import { PageList } from './PageList'
import { PageView } from './PageView'
import { useRoute } from './route'

export default function App() {
  const [route, navigate] = useRoute()
  const [pages, setPages] = useState<PageSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPages().then(setPages, (e: Error) => setError(e.message))
  }, [])

  const markTranslated = useCallback((book: string, pageIndex: number) => {
    setPages((all) =>
      all && all.map((p) => (p.book_title === book && p.page_index === pageIndex ? { ...p, translated: true } : p)),
    )
  }, [])

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
        {route ? (
          <PageView
            key={`${route.book}/${route.pageIndex}`}
            book={route.book}
            pageIndex={route.pageIndex}
            pages={pages ?? []}
            navigate={navigate}
            onTranslated={markTranslated}
          />
        ) : (
          <p className="muted">Pick a page on the left.</p>
        )}
      </main>
    </div>
  )
}
