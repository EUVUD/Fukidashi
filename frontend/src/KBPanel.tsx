import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
} from 'react'
import {
  ApiError,
  decideEntry,
  decideProfile,
  decideProposal,
  fetchKB,
  previewRename,
  renameRow,
  type HistoryLine,
  type KBDecision,
  type KBEntry,
  type KBProfile,
  type KBProposal,
  type KBView,
  type RenamedPage,
  type RenamePreview,
} from './api'
import {
  EMPTY_PROPOSAL_QUEUE,
  EMPTY_REVIEW_QUEUE,
  NO_EDITOR_KB,
  NO_ENTRIES,
  NO_PROFILES,
  NO_RIVAL,
  RENAME_LABEL,
  TAKE_LABEL,
  canRename,
  describeEntry,
  describeHistoryLine,
  describeOutcome,
  describePayload,
  describeProfile,
  describeRival,
  describeRowDecision,
  describeWaiting,
  entryControlName,
  formatAt,
  isConflict,
  isRefusal,
  itemKey,
  offerForProposal,
  offerForRow,
  profileControlName,
  proposalControlName,
  proposalLabel,
  provenanceLabel,
  reasonText,
  renameControlName,
  rivalsOf,
  statusLabel,
  summaryLine,
  takeControlName,
  takeOffer,
  verbOf,
  type ItemRef,
  type Offer,
  type PayloadWords,
  type Words,
} from './kbView'
import { Link } from './Link'
import {
  CANCEL_LABEL,
  CONFIRM_LABEL,
  PREVIEW_LABEL,
  STALE_NOTE,
  describeLeft,
  describePreview,
  describeRename,
  previewHint,
  renameCardName,
  type LeftWords,
  type ListedBox,
  type RenameRow,
} from './renameView'
import { routePath } from './route'

interface Props {
  book: string
  navigate: (path: string) => void
  runsEnded: number // how many Runs of the editor have ended: each may have written the KB (`full` does, at finalize)
  onRenamed: (book: string, runs: readonly RenamedPage[]) => void // a rename wrote Human edits on these pages: the open one reads again
}

// The one rename that is open: where its card sits, what it renames, what the field holds and the preview made for it.
interface RenameOpen {
  host: string // `itemKey` of the item whose card shows it: the row itself, or the conflict that offered to take its rendering
  row: RenameRow
  fixed: boolean // "Take this rendering": the rendering is the proposal's, so there is no field to type in
  draft: string // the field's text; for a fixed one, the proposal's rendering
  preview: { target: string; plan: RenamePreview } | null // the last answer, with the rendering it was made for
  stale: boolean // a preview was dropped because the KB was read again or changed: the card says so
}

// A preview was made from the KB as it was: once the KB is read again, or a decision changed it, it may no longer be what the write
// would do. It goes, and the field keeps its text.
const dropPreview = (open: RenameOpen | null): RenameOpen | null =>
  open === null || open.preview === null ? open : { ...open, preview: null, stale: true }

// What every row needs of the panel: the book, the way to a page, whether a request is in flight, the reason a request about the
// row was refused (and the element that shows it, where focus goes when the button is gone), the way to decide, which takes the
// button that asked for it, and the rename that is open with the ways to work it.
interface Shared {
  book: string
  navigate: (path: string) => void
  disabled: boolean
  errorOf: (ref: ItemRef) => string | null
  refusalRef: RefObject<HTMLParagraphElement | null>
  decide: (ref: ItemRef, decision: KBDecision, label: Words, control: HTMLElement) => void
  rename: RenameOpen | null
  left: { key: string; words: LeftWords } | null // the boxes the last rename left to check by hand, shown under its row
  openRename: (host: ItemRef, row: RenameRow, draft: string, fixed: boolean, control: HTMLElement) => void
  setDraft: (draft: string) => void
  preview: (control: HTMLElement) => void
  confirm: (control: HTMLElement) => void
  cancelRename: () => void
}

// Focus fell to the top of the page: nothing has it, or the body does. That is where the browser puts it when the focused
// button is disabled, as every control is while a request is in flight.
const focusIsLost = () => document.activeElement === null || document.activeElement === document.body

// The one panel of a book's Editor KB (OUTLINE §0, Proposal Queue, Human Review Queue and Editor KB; SCOPE.md #5, #6): a
// disclosure, closed until the reader opens it, whose line carries the counts. It lists what waits for a person, then the entries
// and the profiles, and lets a person approve or reject through #82's routes, and rename through #88's, after a preview. What it
// says and what it offers is kbView.ts's and renameView.ts's; this only places it. The parent keys it by book, so the KB it
// loaded and the reader's choice to open it stay while the reader moves between the book's pages.
//
// A request of the person's (a decision, a rename's preview, a rename's write) is one request at a time: `deciding` is set while
// one is in flight and every control is disabled, which is also why focus has to be put back (see the two effects below).
export function KBPanel({ book, navigate, runsEnded, onRenamed }: Props) {
  const ids = useId()
  const [kb, setKb] = useState<KBView | null>(null)
  const [busy, setBusy] = useState<'loading' | 'deciding' | null>('loading') // one request at a time
  const [loadError, setLoadError] = useState<string | null>(null) // the last load failed; the KB it had stays
  const [panelError, setPanelError] = useState<{ what: string; message: string } | null>(null) // a request failed, not by the rules
  const [refused, setRefused] = useState<{ key: string; message: string } | null>(null) // the rules refused it (409), or a 422
  const [said, setSaid] = useState<Words | null>(null) // what the last decision or rename did
  const [rename, setRename] = useState<RenameOpen | null>(null) // one card at a time
  const [left, setLeft] = useState<{ key: string; words: LeftWords } | null>(null) // what the last rename left to check by hand
  const loading = useRef<AbortController | null>(null)
  const deciding = useRef(false)
  const reloadAfter = useRef(false) // a load follows the decision's answer: it was asked for in the meantime, or the answer was a 404
  const gone = useRef(false) // the reader left the book: a decision still on its way starts nothing more
  const refocus = useRef<HTMLElement | null>(null) // the button of a request that failed: focus goes back once nothing is in flight
  const opener = useRef<HTMLElement | null>(null) // the button that opened the card: Cancel and Esc give focus back to it
  const status = useRef<HTMLParagraphElement>(null)
  const errorLine = useRef<HTMLParagraphElement>(null) // "The decision failed: …"
  const refusal = useRef<HTMLParagraphElement>(null) // the refusal beside an item: only one is ever shown, the last request's

  // What the last request left on screen (what it did, why it was refused, why it failed, the boxes it left to check) is over.
  const forget = useCallback(() => {
    setPanelError(null)
    setRefused(null)
    setSaid(null)
    setLeft(null)
  }, [])

  // Reads the KB. A load that is still on its way is dropped for it, and so is a decision's answer being older than a load's. An
  // open preview goes with it: it was made from the KB as it was.
  const load = useCallback(() => {
    if (deciding.current) {
      reloadAfter.current = true
      return
    }
    setRename(dropPreview)
    loading.current?.abort()
    const ctrl = new AbortController()
    loading.current = ctrl
    setBusy('loading')
    setLoadError(null)
    fetchKB(book, ctrl.signal).then(
      (next) => {
        if (ctrl.signal.aborted) return
        setKb(next)
        setBusy(null)
      },
      (e: Error) => {
        if (ctrl.signal.aborted) return
        setLoadError(e.message)
        setBusy(null)
      },
    )
  }, [book])

  // When the book is shown, and again whenever a Run of the editor ends (a `full` Run writes the KB at finalize, before its
  // `run_end`). Read while the panel is closed too: its line is what tells the reader that something waits for them. A Run's
  // end also makes what an earlier decision left on screen stale, as Refresh does. A decision still in flight keeps its own
  // result: `load` holds the read back until its answer is in, and the read clears nothing.
  useEffect(() => {
    gone.current = false
    if (!deciding.current) forget()
    load()
    return () => {
      gone.current = true
      loading.current?.abort()
    }
  }, [load, forget, runsEnded])

  // The reader asked for it.
  const refresh = () => {
    forget()
    load()
  }

  // A decision that was made leaves the button that made it gone from the list (an approved proposal leaves its queue), and
  // focus with it: the line that says what happened takes it, so a keyboard goes on from there, not from the top of the page.
  useEffect(() => {
    if (said !== null && focusIsLost()) status.current?.focus({ preventScroll: true })
  }, [said])

  // A decision that failed leaves the same hole: the button was disabled while the request was in flight, so focus fell to the
  // top of the page. Once nothing is in flight, focus goes back to the button that made the decision; when that is gone from
  // the panel (a 404: the row was deleted, and the read that follows takes it from the list), to the refusal beside the item,
  // or to the line that says the decision failed, or to the status line. Focus the reader has moved elsewhere in the meantime
  // is left where it is.
  useEffect(() => {
    const control = refocus.current
    if (control === null || busy !== null) return
    refocus.current = null
    if (!focusIsLost()) return
    // The button is where the reader was: nothing to scroll to. The lines may be off screen, and are what the reader must see.
    if (control.isConnected) control.focus({ preventScroll: true })
    else (refusal.current ?? errorLine.current ?? status.current)?.focus()
  }, [busy])

  // One request of the person's at a time. `key` is the item a refusal is shown beside (`itemKey`), `control` the button that asked:
  // where focus goes back to if the request fails, and `what` says what failed, if it did not fail by the rules. `send` makes the
  // request and puts its answer on screen.
  const run = async (key: string, control: HTMLElement, what: string, send: () => Promise<void>) => {
    if (deciding.current) return
    loading.current?.abort() // a read that began before this request would answer with a KB older than the one it will
    deciding.current = true
    setBusy('deciding')
    forget()
    try {
      await send()
    } catch (e) {
      refocus.current = control
      // 409: the rules refused it, and the KB is as it was; the reason goes beside the item. So does a 422 that carries the
      // server's own sentence (a rendering the KB would not store). Anything else goes in the panel.
      if (e instanceof ApiError && isRefusal(e.status, e.fromDetail)) setRefused({ key, message: e.message })
      else setPanelError({ what, message: (e as Error).message })
      // 404: the item is gone (or its KB is not there): read the KB again, so that the row leaves without Refresh. The error
      // stays, because `load` clears nothing of what a request left.
      if (e instanceof ApiError && e.status === 404) reloadAfter.current = true
    } finally {
      deciding.current = false
      setBusy(null)
      if (reloadAfter.current && !gone.current) {
        reloadAfter.current = false
        load()
      }
    }
  }

  const decide = (ref: ItemRef, decision: KBDecision, label: Words, control: HTMLElement) =>
    run(itemKey(ref), control, 'The decision failed', async () => {
      let next: KBView
      let words: Words
      if (ref.kind === 'proposal') {
        const answer = await decideProposal(book, ref.id, decision)
        next = answer.kb
        words = describeOutcome(answer.outcome, label)
      } else {
        next = await (ref.kind === 'entry' ? decideEntry : decideProfile)(book, ref.id, decision)
        words = describeRowDecision(ref.kind, ref.id, label, decision)
      }
      setKb(next)
      setRename(dropPreview) // the KB changed under it
      setLoadError(null) // the answer is as fresh as a load's: a read that failed before it is over, as a load that succeeds says
      setSaid(words)
    })

  // What renaming the row to `target` would do. The card keeps the answer with the rendering it was made for: the confirm writes that.
  const requestPreview = (key: string, row: RenameRow, target: string, control: HTMLElement) =>
    run(key, control, 'The preview failed', async () => {
      const plan = await previewRename(book, row.kind, row.id, target)
      setRename((open) =>
        open !== null && open.host === key && open.row.kind === row.kind && open.row.id === row.id
          ? { ...open, preview: { target, plan }, stale: false }
          : open,
      )
    })

  // The rename itself. The announcement and the boxes left to check are made from its answer, not from the preview's: the server
  // plans again, so the book may have changed in between. The Human edits it wrote go to the page that is open.
  const requestRename = (key: string, row: RenameRow, target: string, control: HTMLElement) =>
    run(key, control, 'The rename failed', async () => {
      const out = await renameRow(book, row.kind, row.id, target)
      const boxes = describeLeft(out)
      setKb(out.kb)
      setLoadError(null)
      setRename(null)
      setLeft(boxes === null ? null : { key: itemKey({ kind: row.kind, id: row.id }), words: boxes })
      setSaid(describeRename(out))
      onRenamed(book, out.runs)
    })

  // Opens the card on an item, and closes any other: only one is open at a time. "Take this rendering" has its rendering already,
  // so it previews at once.
  const openRename = (host: ItemRef, row: RenameRow, draft: string, fixed: boolean, control: HTMLElement) => {
    if (deciding.current) return
    opener.current = control
    setRename({ host: itemKey(host), row, fixed, draft, preview: null, stale: false })
    if (fixed) void requestPreview(itemKey(host), row, draft, control)
  }

  // Closing the card ends what it asked: a refusal it got goes with it, and the text of the field. Focus goes back to the button that
  // opened it, before the card goes: the card's own button has it now.
  const cancelRename = () => {
    if (deciding.current) return
    const host = rename?.host
    const control = opener.current
    opener.current = null
    setRename(null)
    setRefused((last) => (last !== null && last.key === host ? null : last))
    if (control?.isConnected) control.focus({ preventScroll: true })
  }

  const shared: Shared = {
    book,
    navigate,
    disabled: busy !== null,
    errorOf: (ref) => (refused?.key === itemKey(ref) ? refused.message : null),
    refusalRef: refusal,
    decide,
    rename,
    left,
    openRename,
    setDraft: (draft) => setRename((open) => open && { ...open, draft }),
    preview: (control) => {
      if (rename !== null) void requestPreview(rename.host, rename.row, rename.draft, control)
    },
    confirm: (control) => {
      if (rename?.preview) void requestRename(rename.host, rename.row, rename.preview.target, control)
    },
    cancelRename,
  }

  return (
    <section className="kb-panel" aria-label="Knowledge Base">
      <details>
        <summary>{summaryLine(kb, loadError !== null)}</summary>
        <div className="kb-body">
          <div className="kb-toolbar">
            <span className="muted">
              {kb?.kb_id == null ? `Editor KB of ${book}` : `Editor KB ${kb.kb_id} of ${book}`}
            </span>
            <button type="button" className="secondary" onClick={refresh} disabled={busy !== null}>
              {busy === 'loading' ? 'Loading…' : 'Refresh'}
            </button>
          </div>
          {/* Always there, so that a screen reader announces what a decision did when it arrives. */}
          <p ref={status} className="kb-status" role="status" tabIndex={-1}>
            {said && <Text words={said} />}
          </p>
          {loadError && <p className="error" role="alert">The KB could not be loaded: {loadError}</p>}
          {panelError && (
            <p ref={errorLine} className="error" role="alert" tabIndex={-1}>
              {panelError.what}: {panelError.message}
            </p>
          )}
          {kb === null && busy === 'loading' && <p className="muted">Loading the KB…</p>}
          {kb !== null && kb.kb_id === null && <p className="muted">{NO_EDITOR_KB}</p>}
          {kb !== null && kb.kb_id !== null && (
            <>
              <Section
                id={`${ids}-review`}
                title="Human Review Queue"
                count={kb.human_review_queue.length}
                hint="New names and conflicts wait here for a person."
              >
                {kb.human_review_queue.length === 0 ? (
                  <p className="muted">{EMPTY_REVIEW_QUEUE}</p>
                ) : (
                  <ul className="kb-list">
                    {kb.human_review_queue.map((p) => (
                      <ProposalItem key={p.proposal_id} p={p} kb={kb} shared={shared} />
                    ))}
                  </ul>
                )}
              </Section>
              <Section
                id={`${ids}-proposals`}
                title="Proposal Queue"
                count={kb.proposal_queue.length}
                hint="Terms the Curator gave no decision for."
              >
                {kb.proposal_queue.length === 0 ? (
                  <p className="muted">{EMPTY_PROPOSAL_QUEUE}</p>
                ) : (
                  <ul className="kb-list">
                    {kb.proposal_queue.map((p) => (
                      <ProposalItem key={p.proposal_id} p={p} kb={kb} shared={shared} />
                    ))}
                  </ul>
                )}
              </Section>
              <Section
                id={`${ids}-entries`}
                title="Entries"
                count={kb.entries.length}
                hint="Every status: a rejected entry stays listed, so that it can be approved again."
              >
                {kb.entries.length === 0 ? (
                  <p className="muted">{NO_ENTRIES}</p>
                ) : (
                  <ul className="kb-list">
                    {kb.entries.map((e) => (
                      <EntryItem key={e.id} e={e} kb={kb} shared={shared} />
                    ))}
                  </ul>
                )}
              </Section>
              <Section
                id={`${ids}-profiles`}
                title="Profiles"
                count={kb.profiles.length}
                hint="Characters. A new name's profile waits for a person until they approve it."
              >
                {kb.profiles.length === 0 ? (
                  <p className="muted">{NO_PROFILES}</p>
                ) : (
                  <ul className="kb-list">
                    {kb.profiles.map((r) => (
                      <ProfileItem key={r.id} r={r} kb={kb} shared={shared} />
                    ))}
                  </ul>
                )}
              </Section>
            </>
          )}
        </div>
      </details>
    </section>
  )
}

// ---- Pieces --------------------------------------------------------------------------------------------------------------

// Text that quotes the book: each piece in Japanese or Simplified Chinese carries its language.
function Text({ words }: { words: Words }) {
  return (
    <>
      {words.map((piece, i) =>
        typeof piece === 'string' ? (
          piece
        ) : (
          <span key={i} lang={piece.lang}>
            {piece.text}
          </span>
        ),
      )}
    </>
  )
}

function Section({ id, title, count, hint, children }: { id: string; title: string; count: number; hint: string; children: ReactNode }) {
  return (
    <section className="kb-section" aria-labelledby={id}>
      <h3 id={id}>
        {title} ({count})
      </h3>
      <p className="muted">{hint}</p>
      {children}
    </section>
  )
}

function PageLink({ shared, page, chip = false }: { shared: Shared; page: number; chip?: boolean }) {
  return (
    <Link
      to={routePath(shared.book, page)}
      navigate={shared.navigate}
      className={chip ? 'chip' : undefined}
      title={`Page ${page} of ${shared.book}`}
    >
      {chip ? page : `page ${page}`}
    </Link>
  )
}

function Pages({ shared, pages }: { shared: Shared; pages: readonly number[] }) {
  return (
    <div className="kb-pages">
      <span>Evidence pages:</span>
      {pages.length === 0 ? (
        <span className="muted">none</span>
      ) : (
        <span className="chips">
          {pages.map((page) => (
            <PageLink key={page} shared={shared} page={page} chip />
          ))}
        </span>
      )}
    </div>
  )
}

function Extras({ words }: { words: PayloadWords }) {
  return (
    <>
      {words.extras.map((extra, i) => (
        <p key={i} className="kb-extra">
          <Text words={extra} />
        </p>
      ))}
    </>
  )
}

// The buttons of an item: the decisions offered, with the rename's button after them (`after`), or the conflict's "Take this
// rendering" before them (`before`), and the note that says why something is not offered.
function Decisions({
  offer,
  nameOf,
  onDecide,
  disabled,
  before,
  after,
}: {
  offer: Offer
  nameOf: (decision: KBDecision) => string
  onDecide: (decision: KBDecision, control: HTMLElement) => void
  disabled: boolean
  before?: ReactNode
  after?: ReactNode
}) {
  if (offer.decisions.length === 0 && offer.note === null && !before && !after) return null
  return (
    <div className="kb-actions">
      {before}
      {offer.decisions.map((decision) => (
        <button
          key={decision}
          type="button"
          className={decision === 'reject' ? 'secondary' : undefined}
          aria-label={nameOf(decision)}
          disabled={disabled}
          onClick={(event) => onDecide(decision, event.currentTarget)}
        >
          {verbOf(decision)}
        </button>
      ))}
      {after}
      {offer.note && <span className="muted kb-note">{offer.note}</span>}
    </div>
  )
}

// The reason the rules refused a decision, beside its item. It can take focus (`focusRef`, which only the one being shown holds):
// when the decision's button is gone from the panel, the reader's keyboard goes on from here.
function Refusal({ message, focusRef }: { message: string | null; focusRef: RefObject<HTMLParagraphElement | null> }) {
  return message === null ? null : (
    <p ref={focusRef} className="error kb-refusal" role="alert" tabIndex={-1}>
      {message}
    </p>
  )
}

function History({ lines }: { lines: readonly HistoryLine[] }) {
  return (
    <details className="kb-history">
      <summary>History ({lines.length})</summary>
      <ol>
        {lines.map((line, i) => (
          <li key={i}>
            <span className="muted kb-at">{formatAt(line.at)}</span> <Text words={describeHistoryLine(line)} />
          </li>
        ))}
      </ol>
    </details>
  )
}

// A status as a tag: the colours only echo the words.
const tone = (status: string) => (status === 'approved' ? ' ok' : status === 'pending_human' ? ' wait' : '')

function ProposalItem({ p, kb, shared }: { p: KBProposal; kb: KBView; shared: Shared }) {
  const words = describePayload(p.kind, p.payload)
  const waits = describeWaiting(p)
  const ref: ItemRef = { kind: 'proposal', id: p.proposal_id }
  const rivals = rivalsOf(p, kb)
  const take = takeOffer(p, kb)
  const open = shared.rename?.host === itemKey(ref) ? shared.rename : null
  return (
    <li className="kb-item waiting">
      <p className="kb-head">
        <span className="tag wait">{waits.label}</span>
        <span className="tag">{words.label}</span>
        <strong className="kb-pair">
          <Text words={words.pair} />
        </strong>
      </p>
      <Extras words={words} />
      <p className="kb-extra muted">{waits.text}</p>
      {isConflict(p) &&
        (rivals.length > 0 ? (
          <ul className="kb-rivals" aria-label="Rows of the KB it collides with">
            {rivals.map((r) => (
              <li key={`${r.kind}:${r.id}`}>
                <Text words={describeRival(r)} />
              </li>
            ))}
          </ul>
        ) : (
          <p className="kb-extra muted">{NO_RIVAL}</p>
        ))}
      <p className="kb-extra">
        Proposed on <PageLink shared={shared} page={p.page_index} />. Reason: {reasonText(p)}
      </p>
      <Decisions
        offer={offerForProposal(p, kb)}
        nameOf={(decision) => proposalControlName(decision, p)}
        onDecide={(decision, control) => shared.decide(ref, decision, proposalLabel(p), control)}
        disabled={shared.disabled}
        before={
          take && (
            <button
              type="button"
              aria-label={takeControlName(p, take)}
              disabled={shared.disabled}
              onClick={(event) =>
                shared.openRename(ref, { kind: 'entry', id: take.entry.id }, take.target, true, event.currentTarget)
              }
            >
              {TAKE_LABEL}
            </button>
          )
        }
      />
      {open && <RenameCard open={open} kb={kb} shared={shared} />}
      <Refusal message={shared.errorOf(ref)} focusRef={shared.refusalRef} />
    </li>
  )
}

// The button that opens a rename's form on a row, offered on a readable one (`canRename`).
function RenameButton({ name, current, row, host, shared }: { name: string; current: string; row: RenameRow; host: ItemRef; shared: Shared }) {
  return (
    <button
      type="button"
      className="secondary"
      aria-label={name}
      disabled={shared.disabled}
      onClick={(event) => shared.openRename(host, row, current, false, event.currentTarget)}
    >
      {RENAME_LABEL}
    </button>
  )
}

function EntryItem({ e, kb, shared }: { e: KBEntry; kb: KBView; shared: Shared }) {
  const words = describeEntry(e)
  const ref: ItemRef = { kind: 'entry', id: e.id }
  const open = shared.rename?.host === itemKey(ref) ? shared.rename : null
  const left = shared.left?.key === itemKey(ref) ? shared.left.words : null
  return (
    <li className={`kb-item ${e.status}`}>
      <p className="kb-head">
        <span className="tag">Entry {e.id}</span>
        <strong className="kb-pair">
          <Text words={words.pair} />
        </strong>
        <span className={`tag${tone(e.status)}`}>{statusLabel(e.status)}</span>
        <span className="tag">provenance: {provenanceLabel(e.provenance)}</span>
      </p>
      <Extras words={words} />
      <Pages shared={shared} pages={e.evidence_pages} />
      <History lines={e.history} />
      <Decisions
        offer={offerForRow(e.status)}
        nameOf={(decision) => entryControlName(decision, e)}
        onDecide={(decision, control) => shared.decide(ref, decision, words.pair, control)}
        disabled={shared.disabled}
        after={
          canRename(e.status) && (
            <RenameButton
              name={renameControlName('entry', e.id, words.pair)}
              current={e.target_term}
              row={{ kind: 'entry', id: e.id }}
              host={ref}
              shared={shared}
            />
          )
        }
      />
      {open && <RenameCard open={open} kb={kb} shared={shared} />}
      {left && <Left words={left} shared={shared} />}
      <Refusal message={shared.errorOf(ref)} focusRef={shared.refusalRef} />
    </li>
  )
}

function ProfileItem({ r, kb, shared }: { r: KBProfile; kb: KBView; shared: Shared }) {
  const words = describeProfile(r)
  const ref: ItemRef = { kind: 'profile', id: r.id }
  const open = shared.rename?.host === itemKey(ref) ? shared.rename : null
  const left = shared.left?.key === itemKey(ref) ? shared.left.words : null
  return (
    <li className={`kb-item ${r.status}`}>
      <p className="kb-head">
        <span className="tag">Profile {r.id}</span>
        <strong className="kb-pair">
          <Text words={words.pair} />
        </strong>
        <span className={`tag${tone(r.status)}`}>{statusLabel(r.status)}</span>
        <span className="tag">provenance: {provenanceLabel(r.provenance)}</span>
      </p>
      <Extras words={words} />
      <Pages shared={shared} pages={r.evidence_pages} />
      <History lines={r.history} />
      <Decisions
        offer={offerForRow(r.status)}
        nameOf={(decision) => profileControlName(decision, r)}
        onDecide={(decision, control) => shared.decide(ref, decision, words.pair, control)}
        disabled={shared.disabled}
        after={
          canRename(r.status) && (
            <RenameButton
              name={renameControlName('profile', r.id, words.pair)}
              current={r.name_tgt}
              row={{ kind: 'profile', id: r.id }}
              host={ref}
              shared={shared}
            />
          )
        }
      />
      {open && <RenameCard open={open} kb={kb} shared={shared} />}
      {left && <Left words={left} shared={shared} />}
      <Refusal message={shared.errorOf(ref)} focusRef={shared.refusalRef} />
    </li>
  )
}

// ---- A rename's card: its form, and the preview that comes before the write ---------------------------------------------------

// Boxes that a rename lists and does not change, each with why and, when it has one, the translation it holds now. A person edits
// them by hand, and the page link is the way there.
function ListedBoxes({ boxes, shared }: { boxes: readonly ListedBox[]; shared: Shared }) {
  return (
    <ul className="kb-boxes">
      {boxes.map(({ page, box, why }) => (
        <li key={`${page}:${box.idx}`}>
          <p>
            <PageLink shared={shared} page={page} /> <span className="tag">box {box.idx}</span> <Text words={box.ja} />
          </p>
          <p className="muted">
            <Text words={why} />
            {box.before !== null && (
              <>
                {' '}
                <Text words={box.before} />
              </>
            )}
          </p>
        </li>
      ))}
    </ul>
  )
}

// What the last rename left to check by hand, under its row, until the next request or a reload.
function Left({ words, shared }: { words: LeftWords; shared: Shared }) {
  return (
    <div className="kb-left" role="group" aria-label="Boxes left to check by hand">
      <p className="kb-extra">{words.heading}</p>
      <ListedBoxes boxes={words.boxes} shared={shared} />
    </div>
  )
}

// The form (a field with the rendering, Preview and Cancel) and, once a preview has answered, what the rename would do with the
// Confirm that writes it. Enter in the field previews; Esc cancels. Nothing is written before the Confirm, and it writes the
// rendering the preview was made for, not what the field holds by then.
function RenameCard({ open, kb, shared }: { open: RenameOpen; kb: KBView; shared: Shared }) {
  const ids = useId()
  const field = useRef<HTMLInputElement>(null)
  const group = useRef<HTMLDivElement>(null)
  const previewButton = useRef<HTMLButtonElement>(null)
  const { preview, row } = open
  const words = preview === null ? null : describePreview(preview.plan, preview.target, kb)
  const hint = preview === null ? null : previewHint(open.draft, preview.target, !open.fixed)

  // Opened: the reader types the new rendering next, over the one that is there.
  useEffect(() => {
    field.current?.focus()
    field.current?.select()
  }, [])
  // A preview answered, and the button that asked lost focus while the request was in flight. The preview takes it, so that a keyboard
  // goes on from what it says, and a second Enter does not confirm what was not read.
  useEffect(() => {
    if (preview !== null && focusIsLost()) group.current?.focus()
  }, [preview])

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (previewButton.current) shared.preview(previewButton.current)
  }
  // An input method's own Esc is not ours.
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key !== 'Escape' || event.nativeEvent.isComposing || shared.disabled) return
    event.preventDefault()
    shared.cancelRename()
  }

  return (
    <div className="kb-rename" role="group" aria-label={renameCardName(row.kind, row.id)} onKeyDown={onKeyDown}>
      <form onSubmit={onSubmit}>
        {open.fixed ? (
          <p className="kb-extra">
            Rendering to take: <span lang="zh-Hans">{open.draft}</span>
          </p>
        ) : (
          <label className="kb-field">
            <span>New rendering</span>
            <input
              ref={field}
              type="text"
              lang="zh-Hans"
              value={open.draft}
              disabled={shared.disabled}
              onChange={(event) => shared.setDraft(event.target.value)}
            />
          </label>
        )}
        <div className="kb-actions">
          <button ref={previewButton} type="submit" disabled={shared.disabled}>
            {PREVIEW_LABEL}
          </button>
          <button type="button" className="secondary" disabled={shared.disabled} onClick={shared.cancelRename}>
            {CANCEL_LABEL}
          </button>
          {!open.fixed && <span className="muted kb-note">Enter previews · Esc cancels</span>}
        </div>
      </form>
      {open.stale && (
        <p className="kb-extra muted" role="status">
          {STALE_NOTE}
        </p>
      )}
      {preview !== null && words !== null && (
        <div ref={group} className="kb-preview" role="group" aria-labelledby={`${ids}-title`} tabIndex={-1}>
          <h4 id={`${ids}-title`}>
            <Text words={words.title} />
          </h4>
          {words.stored && (
            <p className="kb-extra">
              <Text words={words.stored} />
            </p>
          )}
          {words.approves && <p className="kb-extra">{words.approves}</p>}
          <p className="kb-extra">{words.summary}</p>
          {words.pages.length > 0 && (
            <ul className="kb-boxes" aria-label="Boxes that would change">
              {words.pages.map(({ page, boxes }) => (
                <li key={page}>
                  <PageLink shared={shared} page={page} />
                  <ul>
                    {boxes.map((box) => (
                      <li key={box.idx}>
                        <p>
                          <span className="tag">box {box.idx}</span> <Text words={box.ja} />
                        </p>
                        <p>
                          <span className="muted">was</span> {box.before && <Text words={box.before} />}
                        </p>
                        <p>
                          <span className="muted">now</span> {box.after && <Text words={box.after} />}
                        </p>
                      </li>
                    ))}
                  </ul>
                </li>
              ))}
            </ul>
          )}
          {words.listed.length > 0 && (
            <>
              <p className="kb-extra">Listed, not changed (a person edits these by hand):</p>
              <ListedBoxes boxes={words.listed} shared={shared} />
            </>
          )}
          {words.settles.length > 0 && (
            <>
              <p className="kb-extra">It also settles:</p>
              <ul className="kb-settles">
                {words.settles.map((settled, i) => (
                  <li key={i}>
                    <Text words={settled} />
                  </li>
                ))}
              </ul>
            </>
          )}
          {hint && <p className="kb-extra muted">{hint}</p>}
          <div className="kb-actions">
            <button
              type="button"
              aria-label={words.confirm}
              disabled={shared.disabled}
              onClick={(event) => shared.confirm(event.currentTarget)}
            >
              {CONFIRM_LABEL}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
