import { useCallback, useEffect, useState } from 'react'

export interface Route {
  book: string
  pageIndex: number
}

// /tojime_no_siora/002 -> { book: 'tojime_no_siora', pageIndex: 2 }. Anything else is "no page selected".
export function parseRoute(pathname: string): Route | null {
  const m = /^\/([^/]+)\/(\d+)\/?$/.exec(pathname)
  return m ? { book: decodeURIComponent(m[1]), pageIndex: Number(m[2]) } : null
}

export const routePath = (book: string, pageIndex: number) =>
  `/${encodeURIComponent(book)}/${String(pageIndex).padStart(3, '0')}`

export function useRoute(): [Route | null, (path: string) => void] {
  const [pathname, setPathname] = useState(window.location.pathname)
  useEffect(() => {
    const onPop = () => setPathname(window.location.pathname)
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])
  const navigate = useCallback((path: string) => {
    window.history.pushState(null, '', path)
    setPathname(path)
  }, [])
  return [parseRoute(pathname), navigate]
}
