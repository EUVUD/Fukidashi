// Mirrors the response models in backend/fukidashi/api/pages.py.

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
  run: Run | null // latest succeeded run
  running_run_id: number | null // the page's running run, if any: a reload watches it
}

// The configs the editor runs (OUTLINE §8, "Running pages from the editor").
export type TranslateConfig = 'baseline' | 'full'

// The data of a run's last event, `run_end` (backend/fukidashi/live_trace.py): the stored row's outcome.
export interface RunEnd {
  status: 'succeeded' | 'failed'
  error: string | null
  calls: number
  cost_usd: number | null
}

// A failed request, with the status the backend answered (0: it never answered).
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new ApiError('Cannot reach the backend. Is it running on port 8000?', 0)
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
    throw new ApiError(detail, res.status)
  }
  return res.json() as Promise<T>
}

const pagePath = (book: string, index: number) => `/api/pages/${encodeURIComponent(book)}/${index}`

export const fetchPages = () => request<PageSummary[]>('/api/pages')
export const fetchPage = (book: string, index: number, signal?: AbortSignal) =>
  request<PageDetail>(pagePath(book, index), { signal })
// Starts the page in the background and answers with the run's id; the run's end comes over `watchRun`.
export const translatePage = (book: string, index: number, config: TranslateConfig) =>
  request<{ run_id: number }>(`${pagePath(book, index)}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  })

// Opens the run's live trace (GET /api/runs/:id/events) and calls `onEnd` once, with `run_end`'s data, or with null when
// the stream is over without one (the run was finished already, or the backend refused the stream). The browser
// reconnects by itself after a dropped stream and sends the last event id, so nothing is missed. Returns a function that
// closes the stream; after it, `onEnd` is not called. No event is shown yet.
export function watchRun(runId: number, onEnd: (end: RunEnd | null) => void): () => void {
  const source = new EventSource(`/api/runs/${runId}/events`)
  let over = false
  const finish = (end: RunEnd | null) => {
    if (over) return
    over = true
    source.close()
    onEnd(end)
  }
  source.addEventListener('run_end', (e) => {
    try {
      finish(JSON.parse((e as MessageEvent<string>).data) as RunEnd)
    } catch {
      finish(null)
    }
  })
  // A 204 or an error status closes the source for good; a dropped connection leaves it connecting again.
  source.addEventListener('error', () => {
    if (source.readyState === EventSource.CLOSED) finish(null)
  })
  return () => {
    over = true
    source.close()
  }
}

// A person's edit of one segment's translation: the page, with the edit as its latest run.
export const editTranslation = (book: string, index: number, segment: number, text: string) =>
  request<PageDetail>(`${pagePath(book, index)}/segments/${segment}/translation`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text_zh_hans: text }),
  })
