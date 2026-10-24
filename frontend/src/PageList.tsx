import type { PageSummary } from './api'
import { Link } from './Link'
import { routePath, type Route } from './route'

interface Props {
  pages: PageSummary[]
  route: Route | null
  navigate: (path: string) => void
}

export function PageList({ pages, route, navigate }: Props) {
  const books = new Map<string, PageSummary[]>()
  for (const p of pages) books.set(p.book_title, [...(books.get(p.book_title) ?? []), p])

  return (
    <nav className="page-list">
      {[...books].map(([book, items]) => (
        <section key={book}>
          <h2>{book}</h2>
          <div className="chips">
            {items.map((p) => {
              const current = route?.book === book && route.pageIndex === p.page_index
              const classes = ['chip', current && 'current', p.translated && 'translated', p.segment_count === 0 && 'empty']
              return (
                <Link
                  key={p.page_index}
                  to={routePath(book, p.page_index)}
                  navigate={navigate}
                  className={classes.filter(Boolean).join(' ')}
                  title={`${p.segment_count} segments${p.translated ? ', translated' : ''}`}
                >
                  {p.page_index}
                </Link>
              )
            })}
          </div>
        </section>
      ))}
    </nav>
  )
}
