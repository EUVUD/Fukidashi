import { useCallback, useEffect, useId, useRef, useState, type ReactNode } from 'react'
import {
  ApiError,
  decideEntry,
  decideProfile,
  decideProposal,
  fetchKB,
  type HistoryLine,
  type KBDecision,
  type KBEntry,
  type KBProfile,
  type KBProposal,
  type KBView,
} from './api'
import {
  EMPTY_PROPOSAL_QUEUE,
  EMPTY_REVIEW_QUEUE,
  NO_EDITOR_KB,
  NO_ENTRIES,
  NO_PROFILES,
  NO_RIVAL,
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
  itemKey,
  offerForProposal,
  offerForRow,
  profileControlName,
  proposalControlName,
  proposalWords,
  provenanceLabel,
  reasonText,
  rivalsOf,
  statusLabel,
  summaryLine,
  verbOf,
  type ItemRef,
  type Offer,
  type PayloadWords,
  type Words,
} from './kbView'
import { Link } from './Link'
import { routePath } from './route'

interface Props {
  book: string
  navigate: (path: string) => void
  runsEnded: number // how many Runs of the editor have ended: each may have written the KB (`full` does, at finalize)
}

// What every row needs of the panel: the book, the way to a page, whether a request is in flight, the reason a decision on
// the row was refused and the way to decide.
interface Shared {
  book: string
  navigate: (path: string) => void
  disabled: boolean
  errorOf: (ref: ItemRef) => string | null
  decide: (ref: ItemRef, decision: KBDecision, label: Words) => void
}

// The one panel of a book's Editor KB (OUTLINE §0, Proposal Queue, Human Review Queue and Editor KB; SCOPE.md #5): a disclosure,
// closed until the reader opens it, whose line carries the counts. It lists what waits for a person, then the entries and the
// profiles, and lets a person approve or reject through #82's routes. What it says and what it offers is kbView.ts's; this
// only places it. The parent keys it by book, so the KB it loaded and the reader's choice to open it stay while the reader
// moves between the book's pages.
export function KBPanel({ book, navigate, runsEnded }: Props) {
  const ids = useId()
  const [kb, setKb] = useState<KBView | null>(null)
  const [busy, setBusy] = useState<'loading' | 'deciding' | null>('loading') // one request at a time
  const [loadError, setLoadError] = useState<string | null>(null) // the last load failed; the KB it had stays
  const [panelError, setPanelError] = useState<string | null>(null) // a decision failed for a reason other than the rules'
  const [refused, setRefused] = useState<{ key: string; message: string } | null>(null) // the rules refused it (409)
  const [said, setSaid] = useState<Words | null>(null) // what the last decision did
  const loading = useRef<AbortController | null>(null)
  const deciding = useRef(false)
  const reloadAfter = useRef(false) // a load was asked for while a decision was in flight: it follows the answer
  const gone = useRef(false) // the reader left the book: a decision still on its way starts nothing more
  const status = useRef<HTMLParagraphElement>(null)

  // Reads the KB. A load that is still on its way is dropped for it, and so is a decision's answer being older than a load's.
  const load = useCallback(() => {
    if (deciding.current) {
      reloadAfter.current = true
      return
    }
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
  // `run_end`). Read while the panel is closed too: its line is what tells the reader that something waits for them.
  useEffect(() => {
    gone.current = false
    load()
    return () => {
      gone.current = true
      loading.current?.abort()
    }
  }, [load, runsEnded])

  // The reader asked for it: what the last decision left on screen (what it did, why it was refused, why it failed) is over.
  const refresh = () => {
    setPanelError(null)
    setRefused(null)
    setSaid(null)
    load()
  }

  // A decision that was made leaves the button that made it gone from the list (an approved proposal leaves its queue), and
  // focus with it: the line that says what happened takes it, so a keyboard goes on from there, not from the top of the page.
  useEffect(() => {
    if (said !== null && (document.activeElement === null || document.activeElement === document.body)) {
      status.current?.focus({ preventScroll: true })
    }
  }, [said])

  const decide = async (ref: ItemRef, decision: KBDecision, label: Words) => {
    if (deciding.current) return
    loading.current?.abort() // a read that began before this decision would answer with a KB older than the one it will
    deciding.current = true
    setBusy('deciding')
    setPanelError(null)
    setRefused(null)
    setSaid(null)
    try {
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
      setSaid(words)
    } catch (e) {
      // 409: the rules refused it, and the KB is as it was; the reason goes beside the item. Anything else goes in the panel.
      if (e instanceof ApiError && e.status === 409) setRefused({ key: itemKey(ref), message: e.message })
      else setPanelError((e as Error).message)
    } finally {
      deciding.current = false
      setBusy(null)
      if (reloadAfter.current && !gone.current) {
        reloadAfter.current = false
        load()
      }
    }
  }

  const shared: Shared = {
    book,
    navigate,
    disabled: busy !== null,
    errorOf: (ref) => (refused?.key === itemKey(ref) ? refused.message : null),
    decide,
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
          {panelError && <p className="error" role="alert">The decision failed: {panelError}</p>}
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
                      <EntryItem key={e.id} e={e} shared={shared} />
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
                      <ProfileItem key={r.id} r={r} shared={shared} />
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

function Decisions({
  offer,
  nameOf,
  onDecide,
  disabled,
}: {
  offer: Offer
  nameOf: (decision: KBDecision) => string
  onDecide: (decision: KBDecision) => void
  disabled: boolean
}) {
  if (offer.decisions.length === 0 && offer.note === null) return null
  return (
    <div className="kb-actions">
      {offer.decisions.map((decision) => (
        <button
          key={decision}
          type="button"
          className={decision === 'reject' ? 'secondary' : undefined}
          aria-label={nameOf(decision)}
          disabled={disabled}
          onClick={() => onDecide(decision)}
        >
          {verbOf(decision)}
        </button>
      ))}
      {offer.note && <span className="muted kb-note">{offer.note}</span>}
    </div>
  )
}

function Refusal({ message }: { message: string | null }) {
  return message === null ? null : (
    <p className="error kb-refusal" role="alert">
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
        offer={offerForProposal(p)}
        nameOf={(decision) => proposalControlName(decision, p)}
        onDecide={(decision) => shared.decide(ref, decision, proposalWords(p))}
        disabled={shared.disabled}
      />
      <Refusal message={shared.errorOf(ref)} />
    </li>
  )
}

function EntryItem({ e, shared }: { e: KBEntry; shared: Shared }) {
  const words = describeEntry(e)
  const ref: ItemRef = { kind: 'entry', id: e.id }
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
        onDecide={(decision) => shared.decide(ref, decision, words.pair)}
        disabled={shared.disabled}
      />
      <Refusal message={shared.errorOf(ref)} />
    </li>
  )
}

function ProfileItem({ r, shared }: { r: KBProfile; shared: Shared }) {
  const words = describeProfile(r)
  const ref: ItemRef = { kind: 'profile', id: r.id }
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
        onDecide={(decision) => shared.decide(ref, decision, words.pair)}
        disabled={shared.disabled}
      />
      <Refusal message={shared.errorOf(ref)} />
    </li>
  )
}
