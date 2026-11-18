import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  ApiError,
  EVENT_KINDS,
  decideEntry,
  decideProfile,
  decideProposal,
  fetchKB,
  parseEvent,
  previewRename,
  renameRow,
  type RenameKind,
} from './api'

// What the browser hands over for one event of the stream: its `event:` name, `lastEventId` (the `id:` field, the event's
// seq) and `data`. The importing of `api.ts` needs no DOM: `EventSource` is only touched inside `watchRun`.
describe('parseEvent', () => {
  it('reads the seq from the last event id and the data as JSON', () => {
    expect(parseEvent('stage_start', '12', '{"stage":"review","index":1}')).toEqual({
      seq: 12,
      kind: 'stage_start',
      data: { stage: 'review', index: 1 },
    })
  })

  it('keeps Chinese and Japanese text as it is', () => {
    const event = parseEvent('tool_result', '3', '{"name":"lookup_term","result":"コップ → 杯子","error":null}')
    expect(event?.kind === 'tool_result' && event.data.result).toBe('コップ → 杯子')
  })

  it('drops an event whose id is not a positive whole number', () => {
    for (const id of ['', '0', '-1', '1.5', '1e3', ' 4', '4 ', '0x3', 'abc', '99999999999999999999']) {
      expect(parseEvent('model_call', id, '{}'), JSON.stringify(id)).toBeNull()
    }
    expect(parseEvent('model_call', '007', '{}')?.seq).toBe(7)
  })

  it('drops an event whose data is not a JSON object', () => {
    for (const data of ['', 'not json', 'null', '7', '"text"', '[1, 2]', '{"cut":']) {
      expect(parseEvent('model_call', '1', data), data).toBeNull()
    }
  })

  it('has a kind for each of the eight events of the contract', () => {
    expect([...EVENT_KINDS].sort()).toEqual(
      ['fallback', 'model_call', 'repair', 'run_end', 'run_start', 'stage_end', 'stage_start', 'tool_result'],
    )
  })
})

// The KB functions put a book's title and a proposal's id in a URL. Each of the two holds a space, `/`, `?`, `#` and `%`: the
// characters that would end a path segment, start a query or a fragment, or be read as an escape. `fetch` is stubbed, so nothing
// is sent, and the stub is taken away after each test. The expected paths are written out, not built with `encodeURIComponent`,
// so that a function that stopped encoding would not agree with itself.
describe('the KB requests', () => {
  const TITLE = 'a b/c?d#e%f'
  const TITLE_IN_PATH = 'a%20b%2Fc%3Fd%23e%25f'
  const PROPOSAL = 'x/y z?#%'
  const PROPOSAL_IN_PATH = 'x%2Fy%20z%3F%23%25'
  const KB = { kb_id: 11, entries: [], profiles: [], proposal_queue: [], human_review_queue: [] }
  const DECISIONS = ['approve', 'reject'] as const
  const posted = (decision: string) => ({
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision }), // exactly this: the route takes no other key
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  // `fetch` answers `body` with `status`. The stub records the URL and the init of each call, and is a mock function, so that the
  // last test can tell whether one is left behind.
  function stubFetch(body: unknown, status = 200) {
    const stub = vi.fn((_url: string, _init?: RequestInit) =>
      Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })),
    )
    vi.stubGlobal('fetch', stub)
    return stub
  }

  // The one request that was made.
  const only = (stub: ReturnType<typeof stubFetch>) => {
    expect(stub).toHaveBeenCalledTimes(1)
    const [url, init] = stub.mock.calls[0]
    return { url, init }
  }

  it('reads the KB of a book from a path that holds the title as one segment, with a GET and no body', async () => {
    const stub = stubFetch(KB)
    expect(await fetchKB(TITLE)).toEqual(KB)
    const { url, init } = only(stub)
    expect(url).toBe(`/api/books/${TITLE_IN_PATH}/kb`)
    expect(init?.method ?? 'GET').toBe('GET')
    expect(init?.body).toBeUndefined()
  })

  it.each(DECISIONS)('decides a proposal (%s) with a POST of exactly {"decision"}, the title and the id one segment each', async (decision) => {
    const answer = { outcome: 'proposal_rejected', kb: KB }
    const stub = stubFetch(answer)
    expect(await decideProposal(TITLE, PROPOSAL, decision)).toEqual(answer)
    const { url, init } = only(stub)
    expect(url).toBe(`/api/books/${TITLE_IN_PATH}/kb/proposals/${PROPOSAL_IN_PATH}/decision`)
    expect(init).toMatchObject(posted(decision))
  })

  it.each(DECISIONS)('decides an entry (%s) with a POST of exactly {"decision"}, the title one segment', async (decision) => {
    const stub = stubFetch(KB)
    expect(await decideEntry(TITLE, 40, decision)).toEqual(KB)
    const { url, init } = only(stub)
    expect(url).toBe(`/api/books/${TITLE_IN_PATH}/kb/entries/40/decision`)
    expect(init).toMatchObject(posted(decision))
  })

  it.each(DECISIONS)('decides a profile (%s) with a POST of exactly {"decision"}, the title one segment', async (decision) => {
    const stub = stubFetch(KB)
    expect(await decideProfile(TITLE, 38, decision)).toEqual(KB)
    const { url, init } = only(stub)
    expect(url).toBe(`/api/books/${TITLE_IN_PATH}/kb/profiles/38/decision`)
    expect(init).toMatchObject(posted(decision))
  })

  it('rejects a decision the rules refused with an ApiError: status 409, and the server\'s detail as the message', async () => {
    const DETAIL = 'entry 40 cannot be approved: entry 42 (封缄) is approved for the source term 封印'
    const decisions = [
      () => decideProposal(TITLE, PROPOSAL, 'approve'),
      () => decideEntry(TITLE, 40, 'approve'),
      () => decideProfile(TITLE, 38, 'approve'),
    ]
    for (const decide of decisions) {
      stubFetch({ detail: DETAIL }, 409)
      const error = await decide().then(
        () => null,
        (e: unknown) => e,
      )
      expect(error).toBeInstanceOf(ApiError)
      expect((error as ApiError).status).toBe(409)
      expect((error as ApiError).message).toBe(DETAIL)
    }
  })

  it('tells a row that is gone from a refusal by the status: 404, and the detail as the message', async () => {
    stubFetch({ detail: 'no proposal with that id in the Editor KB of this book' }, 404)
    const error = await decideProposal(TITLE, PROPOSAL, 'reject').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(404)
    expect((error as ApiError).message).toBe('no proposal with that id in the Editor KB of this book')
  })

  it('tells a backend that did not answer by the status too: 0', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    const error = await decideEntry(TITLE, 40, 'reject').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(0)
  })

  it('leaves no stub of fetch behind', () => {
    expect(vi.isMockFunction(globalThis.fetch)).toBe(false)
  })
})

// The rename's two requests (`previewRename`, `renameRow`): one function for each, whatever the row is, so each is checked for an entry
// and for a profile. A rename's body is exactly `{"target"}`, and the title is one segment of the path, as for the decisions.
describe('the rename requests', () => {
  const TITLE = 'a b/c?d#e%f'
  const TITLE_IN_PATH = 'a%20b%2Fc%3Fd%23e%25f'
  const KB = { kb_id: 11, entries: [], profiles: [], proposal_queue: [], human_review_queue: [], proposal_counts: {} }
  const BOX = { page_index: 4, idx: 5, text_ja: 'ここら一帯を仕切るトルティヤー一家の名', before: '托尔蒂亚家族掌管', after: '托尔蒂亚一家掌管' }
  const LISTED = { page_index: 7, idx: 4, text_ja: 'トルティヤー一家', before: '托尔蒂亚一家', after: null }
  const PREVIEW = (kind: RenameKind, id: number) => ({
    kind,
    id,
    before: '托尔蒂亚家族',
    after: '托尔蒂亚一家',
    changes: [BOX],
    listed: [LISTED],
    settles: ['c6f34baec105729df518f50d8ef4e37e'],
  })
  const ROWS = [
    { kind: 'entry', id: 40, segment: 'entries' },
    { kind: 'profile', id: 38, segment: 'profiles' },
  ] as const
  const TARGET = '托尔蒂亚一家'
  const posted = (target: string) => ({
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ target }), // exactly this: the route takes no other key
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  function stubFetch(body: unknown, status = 200, statusText = '') {
    const stub = vi.fn((_url: string, _init?: RequestInit) =>
      Promise.resolve(new Response(JSON.stringify(body), { status, statusText, headers: { 'Content-Type': 'application/json' } })),
    )
    vi.stubGlobal('fetch', stub)
    return stub
  }
  const only = (stub: ReturnType<typeof stubFetch>) => {
    expect(stub).toHaveBeenCalledTimes(1)
    const [url, init] = stub.mock.calls[0]
    return { url, init }
  }
  const failure = (call: () => Promise<unknown>) => call().then(() => null, (e: unknown) => e as ApiError)

  it.each(ROWS)('previews the rename of a $kind with a POST of exactly {"target"}, the title one segment, and answers the plan', async ({ kind, id, segment }) => {
    const answer = PREVIEW(kind, id)
    const stub = stubFetch(answer)
    expect(await previewRename(TITLE, kind, id, TARGET)).toEqual(answer)
    const { url, init } = only(stub)
    expect(url).toBe(`/api/books/${TITLE_IN_PATH}/kb/${segment}/${id}/rename/preview`)
    expect(init).toMatchObject(posted(TARGET))
    expect(JSON.parse(init?.body as string)).toEqual({ target: TARGET })
  })

  it.each(ROWS)('renames a $kind with a POST of exactly {"target"}, the title one segment, and answers the write', async ({ kind, id, segment }) => {
    const answer = { ...PREVIEW(kind, id), runs: [{ page_index: 4, run_id: 906 }], kb: KB }
    const stub = stubFetch(answer)
    expect(await renameRow(TITLE, kind, id, TARGET)).toEqual(answer)
    const { url, init } = only(stub)
    expect(url).toBe(`/api/books/${TITLE_IN_PATH}/kb/${segment}/${id}/rename`)
    expect(init).toMatchObject(posted(TARGET))
    expect(JSON.parse(init?.body as string)).toEqual({ target: TARGET })
  })

  it('sends the rendering as it was typed: spaces, quotes and a newline are the server\'s to judge, so a 422 can say so', async () => {
    for (const target of ['', '  ', ' 托尔蒂亚一家 ', 'a "quoted" \\ name', 'two\nlines']) {
      const stub = stubFetch(PREVIEW('entry', 40))
      await previewRename('tojime_no_siora', 'entry', 40, target)
      expect(JSON.parse(only(stub).init?.body as string), JSON.stringify(target)).toEqual({ target })
      vi.unstubAllGlobals()
    }
  })

  it('keeps a 409 and its reason: the status, and the server\'s detail as the message, for the preview and the write', async () => {
    const DETAIL = 'a Run is still running on page 10 (Run 914) of tojime_no_siora, and a Run that finishes writes the KB: rename after it ends'
    for (const call of [() => previewRename(TITLE, 'entry', 40, TARGET), () => renameRow(TITLE, 'profile', 38, TARGET)]) {
      stubFetch({ detail: DETAIL }, 409, 'Conflict')
      const error = await failure(call)
      expect(error).toBeInstanceOf(ApiError)
      expect(error?.status).toBe(409)
      expect(error?.message).toBe(DETAIL)
      expect(error?.fromDetail).toBe(true)
    }
  })

  it('keeps a 422 that carries a sentence: the rendering the KB would not store, as the server words it', async () => {
    const DETAIL = 'invalid new rendering: it is empty (or only spaces)'
    for (const call of [() => previewRename(TITLE, 'entry', 40, ''), () => renameRow(TITLE, 'entry', 40, '')]) {
      stubFetch({ detail: DETAIL }, 422, 'Unprocessable Entity')
      const error = await failure(call)
      expect(error?.status).toBe(422)
      expect(error?.message).toBe(DETAIL)
      expect(error?.fromDetail).toBe(true)
    }
  })

  it('does not show a 422 whose detail is a list as text: that is a body the route did not take, a bug of the caller\'s', async () => {
    const LIST = [{ type: 'missing', loc: ['body', 'target'], msg: 'Field required', input: {} }]
    for (const call of [() => previewRename(TITLE, 'entry', 40, TARGET), () => renameRow(TITLE, 'profile', 38, TARGET)]) {
      stubFetch({ detail: LIST }, 422, 'Unprocessable Entity')
      const error = await failure(call)
      expect(error).toBeInstanceOf(ApiError)
      expect(error?.status).toBe(422)
      expect(error?.fromDetail).toBe(false)
      expect(error?.message).toBe('422 Unprocessable Entity')
      for (const piece of ['missing', 'Field required', 'target', 'body', 'loc']) expect(error?.message, piece).not.toContain(piece)
    }
  })

  it('tells a row that is gone by the status: 404, and the detail as the message', async () => {
    stubFetch({ detail: 'no entry 40 in the KB' }, 404, 'Not Found')
    const error = await failure(() => previewRename(TITLE, 'entry', 40, TARGET))
    expect(error?.status).toBe(404)
    expect(error?.message).toBe('no entry 40 in the KB')
  })

  it('tells a backend that did not answer by the status too: 0, and not a refusal of the rules', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    const error = await failure(() => renameRow(TITLE, 'entry', 40, TARGET))
    expect(error).toBeInstanceOf(ApiError)
    expect(error?.status).toBe(0)
    expect(error?.fromDetail).toBe(false)
  })

  it('does not take the dev server\'s own 5xx for the backend\'s sentence', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response('Bad Gateway', { status: 502, statusText: 'Bad Gateway' }))))
    const error = await failure(() => previewRename(TITLE, 'entry', 40, TARGET))
    expect(error?.status).toBe(502)
    expect(error?.fromDetail).toBe(false)
    expect(error?.message).toMatch(/Cannot reach the backend/)
  })

  it('leaves no stub of fetch behind', () => {
    expect(vi.isMockFunction(globalThis.fetch)).toBe(false)
  })
})
