// What a book's Editor KB says in words, and what a person may do with each row: the summary line, the words for a proposal,
// the decisions offered, the rows a conflict collides with, the outcome of a decision and the lines of a row's history
// (OUTLINE §0, Proposal Queue, Human Review Queue and Editor KB; §8, "A person's decisions on KB items"). Pure logic with no
// React and no DOM. Only types come from './api', so the module loads anywhere and Vitest checks it directly (OUTLINE §8,
// "Frontend tests"). What the KB holds, and what its rules refuse, is the backend's word (backend/fukidashi/kb/curator.py);
// the wording is here, and so is what is not offered because the server always refuses it.
import type { HistoryLine, KBDecision, KBEntry, KBProfile, KBProposal, KBView, ReviewProposal } from './api'

// ---- Words: text with its language -------------------------------------------------------------------------------------
// A line that quotes the book holds Japanese (a source term, a name, an alias) and Simplified Chinese (a rendering). Each such
// piece says which, so that the component can mark it with `lang` and the tests can check it.

export type Lang = 'ja' | 'zh-Hans'
export type Piece = string | { text: string; lang: Lang }
export type Words = readonly Piece[]

const ja = (text: string): Piece => ({ text, lang: 'ja' })
const zh = (text: string): Piece => ({ text, lang: 'zh-Hans' })

/** The words as one plain string, for a label, a test or a screen reader's name. */
export const plain = (words: Words): string => words.map((piece) => (typeof piece === 'string' ? piece : piece.text)).join('')

/** `封印 → 封缄`: a Japanese source and its Simplified Chinese rendering. */
const pair = (source: string, target: string): Words => [ja(source), ' → ', zh(target)]

/** The pieces with a separator between them. */
function joined(pieces: readonly Piece[], separator = ', '): Piece[] {
  return pieces.flatMap((piece, i) => (i === 0 ? [piece] : [separator, piece]))
}

// ---- States and counting words -------------------------------------------------------------------------------------------

/** The states in which the agents can find a row (OUTLINE §3.3, "Readable states"). A row in any other state is not there for
 *  the KB's rules, so it never collides with a proposal. */
export const READABLE: readonly string[] = ['approved', 'pending_human']
export const isReadable = (status: string): boolean => READABLE.includes(status)

const STATUS_WORDS = new Map([
  ['approved', 'approved'],
  ['pending_human', 'waiting for a person'],
  ['rejected', 'rejected'],
  ['proposed', 'undecided'],
])
/** A status in words; one this panel does not know is shown as it is. */
export const statusLabel = (status: string): string => STATUS_WORDS.get(status) ?? status

/** Who made a row: `agent`, or a person once they decided it (`human`). */
export const provenanceLabel = (provenance: string): string => (provenance === 'human' ? 'person' : provenance)

const counted = (n: number, one: string, many = `${one}s`): string => `${n} ${n === 1 ? one : many}`

// ---- The summary line ------------------------------------------------------------------------------------------------------

/** The disclosure's one line. `kb` is null until the first load has answered; `failed` says that the last load did not (the KB
 *  it had, if any, is still what the line counts). A book with no Editor KB says so. */
export function summaryLine(kb: KBView | null, failed = false): string {
  const head = 'Knowledge Base'
  if (kb === null) return `${head} · ${failed ? 'could not be loaded' : 'loading…'}`
  const tail = failed ? ['could not refresh'] : []
  if (kb.kb_id === null) return [head, 'no Editor KB yet', ...tail].join(' · ')
  return [
    head,
    `${kb.human_review_queue.length} waiting for a person`,
    `${kb.proposal_queue.length} undecided`,
    counted(kb.entries.length, 'entry', 'entries'),
    counted(kb.profiles.length, 'profile'),
    ...tail,
  ].join(' · ')
}

/** What the body says for a book with no Editor KB. */
export const NO_EDITOR_KB =
  'This book has no Editor KB yet. One forms when a page of the book is translated with full. Until then there is nothing to decide.'
export const EMPTY_REVIEW_QUEUE = 'Nothing waits for a person.'
export const EMPTY_PROPOSAL_QUEUE =
  'Nothing is undecided. This queue usually is empty: a term stays here only when the Curator gave no decision for it and nothing in the KB holds its source.'
export const NO_ENTRIES = 'The KB has no entries yet.'
export const NO_PROFILES = 'The KB has no profiles yet.'

// ---- The words for a payload ----------------------------------------------------------------------------------------------

const str = (value: unknown): string => (typeof value === 'string' ? value : '')
const strings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []

export interface PayloadWords {
  label: string // 'Term' or 'Character'; a kind this panel does not know is shown as it is
  pair: Words // the source and its rendering: 封印 → 封缄
  extras: Words[] // one line each: a term's category; a character's other names and notes
}

/** What a proposal proposes (a term: source → target and its category; a character: name → rendering, its other names and its
 *  notes). A field the payload lacks is left out; a kind this panel does not know shows the payload as it came. */
export function describePayload(kind: string, payload: Record<string, unknown>): PayloadWords {
  if (kind === 'term') {
    const category = str(payload.category)
    return {
      label: 'Term',
      pair: pair(str(payload.source_term), str(payload.target_term)),
      extras: category ? [[`category: ${category}`]] : [],
    }
  }
  if (kind === 'character') {
    const aliases = strings(payload.aliases)
    const notes = str(payload.notes)
    const extras: Words[] = []
    if (aliases.length > 0) extras.push(['also called ', ...joined(aliases.map(ja))])
    if (notes) extras.push([`notes: ${notes}`])
    return { label: 'Character', pair: pair(str(payload.name_src), str(payload.name_tgt)), extras }
  }
  return { label: kind, pair: [JSON.stringify(payload)], extras: [] }
}

/** An entry or a profile in the same words as a proposal, so that a row reads like the proposal that made it. */
export const describeEntry = (e: KBEntry): PayloadWords =>
  describePayload('term', { source_term: e.source_term, target_term: e.target_term, category: e.category })
export const describeProfile = (r: KBProfile): PayloadWords =>
  describePayload('character', { name_src: r.name_src, name_tgt: r.name_tgt, aliases: r.aliases, notes: r.notes })

export const proposalWords = (p: KBProposal): Words => describePayload(p.kind, p.payload).pair

/** `proposalWords` as one string, with the page: what names a proposal's controls. */
export const proposalName = (p: KBProposal): string => `${plain(proposalWords(p))}, proposed on page ${p.page_index}`

/** `proposalWords` with the page, still as words: what names a proposal in a sentence, `封印 → 封缄 (proposed on page 19)`. Two
 *  proposals can have one pair on different pages, and a decision's announcement must not read alike for both. The pair keeps its
 *  languages. */
export const proposalLabel = (p: KBProposal): Words => [...proposalWords(p), ` (proposed on page ${p.page_index})`]

/** The proposal's reason, or the words for there being none: the Proposal Queue's terms have none while undecided. */
export const reasonText = (p: KBProposal): string => p.reason ?? 'None yet.'

// ---- What a proposal waits for ----------------------------------------------------------------------------------------------

/** A proposal of the Human Review Queue has `waits_for`; one of the Proposal Queue does not. */
export const isReview = (p: KBProposal): p is ReviewProposal => 'waits_for' in p

/** A proposal that nothing was written for: the KB already held something that collides with it. */
export const isConflict = (p: KBProposal): p is ReviewProposal => isReview(p) && p.waits_for === 'conflict'

/** What the proposal is waiting for, as a short label and a sentence. */
export function describeWaiting(p: KBProposal): { label: string; text: string } {
  if (!isReview(p)) return { label: 'Undecided', text: 'Nobody decided it yet: the Curator gave no decision for it.' }
  if (p.waits_for === 'new_name') {
    const profile = p.profile_id === null ? '' : ` (profile ${p.profile_id})`
    return { label: 'New name', text: `A profile was made for it${profile} and waits for a person to confirm the name.` }
  }
  if (p.waits_for === 'conflict') {
    return { label: 'Conflict', text: 'Nothing was written for it: the KB already holds something that collides with it.' }
  }
  return { label: p.waits_for, text: 'This panel does not know what it waits for.' }
}

// ---- The decisions offered ---------------------------------------------------------------------------------------------------

/** What a person may do with an item, and, when something is left out for a reason, the line that says why. The server still
 *  decides: this only avoids offering what it always refuses. */
export interface Offer {
  decisions: readonly KBDecision[]
  note: string | null
}

export const RENAME_NOTE = 'Approving is not offered: taking this rendering would be a rename, which is not built yet.'
export const UNDECIDED_NAME_NOTE = 'Approving is not offered: only an undecided term can be approved.'

const BOTH: Offer = { decisions: ['approve', 'reject'], note: null }

/** An undecided term and a new name: both decisions. A conflict: only Reject, with the reason. A name that waits undecided
 *  (a state the rules never leave) and a `waits_for` this panel does not know are not in the table: the first can only be
 *  rejected, as the server has it, and for the second nothing is offered. */
export function offerForProposal(p: KBProposal): Offer {
  if (!isReview(p)) return p.kind === 'term' ? BOTH : { decisions: ['reject'], note: UNDECIDED_NAME_NOTE }
  if (p.waits_for === 'new_name') return BOTH
  if (p.waits_for === 'conflict') return { decisions: ['reject'], note: RENAME_NOTE }
  return { decisions: [], note: null }
}

/** An entry or a profile: Approve unless it is approved, Reject unless it is rejected. */
export function offerForRow(status: string): Offer {
  const decisions: KBDecision[] = []
  if (status !== 'approved') decisions.push('approve')
  if (status !== 'rejected') decisions.push('reject')
  return { decisions, note: null }
}

const VERBS: Record<KBDecision, string> = { approve: 'Approve', reject: 'Reject' }
export const verbOf = (decision: KBDecision): string => VERBS[decision]

/** The name of a decision control: it says what it decides, so that a screen reader's list of buttons is not a list of
 *  "Approve" and "Reject". */
export const proposalControlName = (decision: KBDecision, p: KBProposal): string => `${verbOf(decision)} ${proposalName(p)}`
export const entryControlName = (decision: KBDecision, e: KBEntry): string =>
  `${verbOf(decision)} entry ${e.id}: ${plain(describeEntry(e).pair)}`
export const profileControlName = (decision: KBDecision, r: KBProfile): string =>
  `${verbOf(decision)} profile ${r.id}: ${plain(describeProfile(r).pair)}`

/** Where a decision's refusal is shown: beside its item. */
export type ItemRef = { kind: 'proposal'; id: string } | { kind: 'entry'; id: number } | { kind: 'profile'; id: number }
export const itemKey = (ref: ItemRef): string => `${ref.kind}:${ref.id}`

// ---- The rows a conflict collides with ---------------------------------------------------------------------------------------

/** A readable row of the KB that a conflict collides with now. */
export interface Rival {
  kind: 'entry' | 'profile'
  id: number
  status: string
  source: string // an entry's source term; a profile's name
  target: string
  alias: string | null // set when a profile collides through one of its aliases, not through its name
}

const entryRival = (e: KBEntry): Rival => ({
  kind: 'entry', id: e.id, status: e.status, source: e.source_term, target: e.target_term, alias: null,
})
const profileRival = (r: KBProfile, alias: string | null): Rival => ({
  kind: 'profile', id: r.id, status: r.status, source: r.name_src, target: r.name_tgt, alias,
})

/** The readable rows of the KB that a conflict collides with, as the KB is now (it may have changed since the proposal was
 *  queued). A term collides with the entry that has its source, and across kinds with a profile whose name or alias is that
 *  source; a character with the profile that has its name, and across kinds with the entry whose source is that name. Rows
 *  that are not readable (`rejected`, `proposed`) never collide. Nothing but a conflict has rivals: a new name's own profile
 *  would be one, and it is not. Same-kind rows come first. */
export function rivalsOf(p: KBProposal, kb: KBView): Rival[] {
  if (!isConflict(p)) return []
  const term = p.kind === 'term'
  if (!term && p.kind !== 'character') return []
  const source = str(term ? p.payload.source_term : p.payload.name_src)
  const entries = kb.entries.filter((e) => isReadable(e.status) && e.source_term === source).map(entryRival)
  const profiles = kb.profiles.flatMap((r): Rival[] => {
    if (!isReadable(r.status)) return []
    if (r.name_src === source) return [profileRival(r, null)]
    return term && r.aliases.includes(source) ? [profileRival(r, source)] : []
  })
  return term ? [...entries, ...profiles] : [...profiles, ...entries]
}

/** `The KB holds 封印 → 封印 (entry 40, approved)`; through an alias, `… (profile 38, approved), which is also called ゆき`. */
export function describeRival(r: Rival): Words {
  return [
    'The KB holds ',
    ...pair(r.source, r.target),
    ` (${r.kind} ${r.id}, ${statusLabel(r.status)})`,
    ...(r.alias === null ? [] : [', which is also called ', ja(r.alias)]),
  ]
}

/** What a conflict says when no readable row has its source or name. It says only what the panel knows: `rivalsOf` looks for the
 *  same source or name, and the server's rules reach further (a character is matched by containment and by a shared alias), so the
 *  row it collided with may be one this panel does not look for. The reason says what it was. */
export const NO_RIVAL =
  'No readable entry or profile with this source or name was found. Its reason says what it collided with.'

// ---- The outcome of a decision ---------------------------------------------------------------------------------------------

/** What a person's decision on a proposal did, in words (`label`: the proposal as `proposalLabel` has it, with its page). The
 *  proposals route sends seven outcomes. Only `entry_created`, `entry_confirmed` and `profile_approved` are worded as
 *  approved: when the rules hold an approved proposal back (`held_as_conflict`, `held_as_conflict_across_kinds`) nothing was
 *  added and it waits for a person again. An outcome this panel does not know is shown as it is, and is not worded as approved
 *  either. */
export function describeOutcome(outcome: string, label: Words): Words {
  switch (outcome) {
    case 'entry_created':
      return [...label, ' was approved by a person and added as a new entry.']
    case 'entry_confirmed':
      return [...label, ' was approved by a person. It confirms an entry the KB already holds.']
    case 'held_as_conflict':
      return [...label, ' was not added: the KB holds another rendering of this term, so it waits for a person again.']
    case 'held_as_conflict_across_kinds':
      return [...label, ' was not added: this term is the name or an alias of a character in the KB, so it waits for a person again.']
    case 'proposal_rejected':
      return [...label, ' was rejected by a person.']
    case 'profile_approved':
      return [...label, ' was approved by a person. The new name is now an approved profile.']
    case 'profile_rejected':
      return [...label, ' was rejected by a person. The new name is a rejected profile.']
    default:
      return [...label, `: the server answered "${outcome}".`]
  }
}

/** What a person's decision on an entry or a profile did. The two routes answer the KB and no outcome: a 200 says that the
 *  decision was made. */
export function describeRowDecision(kind: 'entry' | 'profile', id: number, label: Words, decision: KBDecision): Words {
  const what = kind === 'entry' ? 'Entry' : 'Profile'
  return [`${what} ${id} (`, ...label, `) was ${decision === 'approve' ? 'approved' : 'rejected'} by a person.`]
}

// ---- A row's history -------------------------------------------------------------------------------------------------------------

const EVENTS = new Map([
  ['created', 'Created'],
  ['confirmed', 'Confirmed'],
  ['merged', 'Merged'],
  ['approved', 'Approved'],
  ['rejected', 'Rejected'],
])

/** The backend's UTC time (`2026-11-14T21:33:27+00:00`) as `2026-11-14 21:33:27 UTC`; anything else as it is. */
export function formatAt(at: string): string {
  const m = /^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:\+00:00|Z)$/.exec(at)
  return m ? `${m[1]} ${m[2]} UTC` : at
}

/** One line of a row's history: what happened, on which page or by a person, and the status it replaced. A page-bound line
 *  (an agent's `created`, `confirmed`, `merged`) says its page. A person's line says so, and the page of the proposal it came
 *  from, or that it came through one. An event this panel does not know is shown as it is. */
export function describeHistoryLine(line: HistoryLine): Words {
  const event = str(line.event)
  const page = typeof line.page === 'number' ? line.page : null
  const person = line.by === 'human'
  const words: Piece[] = [EVENTS.get(event) ?? `Event "${event}"`]
  if (person) {
    words.push(' by a person')
    if (page !== null) words.push(`, from the proposal made on page ${page}`)
    else if (str(line.proposal_id)) words.push(', through its proposal')
  } else if (page !== null) {
    words.push(` on page ${page}`)
  }
  if (event === 'merged') {
    const aliases = strings(line.aliases)
    words.push(aliases.length > 0 ? ': new aliases ' : ': no new alias', ...joined(aliases.map(ja)))
  }
  if (typeof line.from === 'string') words.push(` (was ${statusLabel(line.from)})`)
  return words
}
