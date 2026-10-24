// Mirrors the response models in backend/app/api/pages.py.

export interface PageSummary {
  book_title: string
  page_index: number
  segment_count: number
  translated: boolean
}

export interface Segment {
  index: number
  x: number
  y: number
  w: number
  h: number
  text_ja: string
  text_zh_hant: string // reference
  text_en: string // reference
  text_zh_hans: string | null // latest succeeded run
}

export interface Run {
  id: number
  config: string
  model: string
  response_model: string | null
  status: string
  calls: number
  unknown_usage_calls: number
  input_tokens: number | null
  output_tokens: number | null
  cost_usd: number | null
  started_at: string
  finished_at: string | null
}

export interface PageDetail {
  book_title: string
  page_index: number
  image_url: string
  segments: Segment[]
  run: Run | null
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new Error('Cannot reach the backend. Is it running on port 8000?')
  }
  if (!res.ok) {
    // A 5xx without the API's {"detail": ...} body comes from the dev server's proxy: the backend is down.
    let detail = res.status >= 500 ? `Cannot reach the backend (${res.status}). Is it running on port 8000?` : `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // not JSON: keep the message above
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const pagePath = (book: string, index: number) => `/api/pages/${encodeURIComponent(book)}/${index}`

export const fetchPages = () => request<PageSummary[]>('/api/pages')
export const fetchPage = (book: string, index: number, signal?: AbortSignal) =>
  request<PageDetail>(pagePath(book, index), { signal })
export const translatePage = (book: string, index: number) =>
  request<PageDetail>(`${pagePath(book, index)}/translate`, { method: 'POST' })
