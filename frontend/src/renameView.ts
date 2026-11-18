// What a rename says in words: the preview that comes before it, the announcement that follows it, and the boxes it leaves to
// check by hand (OUTLINE §0, Rename; §8, "Rename → whole-book sync"; SCOPE.md #6). Pure logic with no React and no DOM, like
// kbView.ts, whose words it builds on; only types come from './api', so Vitest checks it directly (OUTLINE §8, "Frontend
// tests"). Which boxes a rename changes, and what it refuses, is the backend's word (backend/fukidashi/kb/curator.py and
// kb/rename.py): this only words what the backend answered, and the announcement is made from the write's answer, never from
// the preview's, because the write plans again and the book may have changed in between.
import type { KBView, RenameBox, RenameKind, RenameOut, RenamePreview } from './api'
import { counted, ja, proposalLabel, zh, type Words } from './kbView'

/** The row a rename changes. */
export interface RenameRow {
  kind: RenameKind
  id: number
}

const KIND_WORDS: Record<RenameKind, string> = { entry: 'Entry', profile: 'Profile' }
/** `Entry 40`, `Profile 38`. */
export const rowLabel = (kind: RenameKind, id: number): string => `${KIND_WORDS[kind]} ${id}`

/** What the preview's buttons say; the confirm's accessible name starts with its label. */
export const PREVIEW_LABEL = 'Preview'
export const CANCEL_LABEL = 'Cancel'
export const CONFIRM_LABEL = 'Confirm rename'

/** What the card says when a preview was dropped: the KB was read again, or a decision changed it, so what the preview said may no
 *  longer be what the write would do. The text in the field stays. */
export const STALE_NOTE =
  'The preview was dropped because the KB was read again or changed: it may no longer be what the rename would do. Preview again to see it.'

/** The name of a rename's card, for a screen reader: `Rename entry 40`. */
export const renameCardName = (kind: RenameKind, id: number): string => `Rename ${rowLabel(kind, id).toLowerCase()}`

/** The row's source (an entry's source term, a profile's name) and status in the KB shown, or null when it is not there. */
function rowIn(kb: KBView, kind: RenameKind, id: number): { source: string; status: string } | null {
  if (kind === 'entry') {
    const e = kb.entries.find((row) => row.id === id)
    return e === undefined ? null : { source: e.source_term, status: e.status }
  }
  const r = kb.profiles.find((row) => row.id === id)
  return r === undefined ? null : { source: r.name_src, status: r.status }
}

/** `Entry 40 (封印)`: the row, with its source in Japanese when the KB shown has the row. */
function named(kb: KBView, kind: RenameKind, id: number): Words {
  const found = rowIn(kb, kind, id)
  return found === null ? [rowLabel(kind, id)] : [`${rowLabel(kind, id)} (`, ja(found.source), ')']
}

// ---- The preview --------------------------------------------------------------------------------------------------------------

/** A box of the book as a preview shows it: its Japanese and its translations, each with its language, so that the panel marks
 *  them like the rest of what it quotes. `before` is the translation the page displays now, null for a box with none; `after` is
 *  the same with the old rendering replaced, null for a box that is only listed. */
export interface BoxWords {
  idx: number // the region id on its page
  ja: Words
  before: Words | null
  after: Words | null
}

const boxWords = (box: RenameBox): BoxWords => ({
  idx: box.idx,
  ja: [ja(box.text_ja)],
  before: box.before === null ? null : [zh(box.before)],
  after: box.after === null ? null : [zh(box.after)],
})

/** The boxes of one page: `changes` come in page order, then `idx`, and are grouped as they come. */
export interface PreviewPage {
  page: number
  boxes: BoxWords[]
}

export function groupByPage(boxes: readonly RenameBox[]): PreviewPage[] {
  const byPage = new Map<number, BoxWords[]>()
  for (const box of boxes) {
    const group = byPage.get(box.page_index)
    if (group === undefined) byPage.set(box.page_index, [boxWords(box)])
    else group.push(boxWords(box))
  }
  return [...byPage].map(([page, group]) => ({ page, boxes: group }))
}

/** A box that is listed and not changed, and why: its Japanese holds the source, but its translation does not use the old
 *  rendering, or its page has no translation at all. A person edits it by hand. */
export interface ListedBox {
  page: number
  box: BoxWords
  why: Words
}

export function describeListed(boxes: readonly RenameBox[], before: string): ListedBox[] {
  return boxes.map((box) => ({
    page: box.page_index,
    box: boxWords(box),
    why: box.before === null ? ['Its page has no translation yet.'] : ['Its translation does not use ', zh(before), '.'],
  }))
}

/** What a preview says, ready to place. `typed` is the rendering the preview was made for, as it was sent: `plan.after` is how
 *  the KB stores it, and the two can differ. `kb` is the KB the panel shows, which names the row and the proposals it settles. */
export interface PreviewWords {
  title: Words // `Rename entry 40 (封印): 托尔蒂亚家族 → 托尔蒂亚一家`
  stored: Words | null // set when the KB stores the rendering otherwise than it was typed
  approves: string | null // set when the row waits for a person: its rename approves it
  summary: string // how many boxes change and how many are only listed; with none to change, that only the KB changes
  pages: PreviewPage[] // `changes`, by page
  listed: ListedBox[]
  settles: Words[] // the proposals the write settles with the row, each named with its pair and page
  confirm: string // the accessible name of the confirm button: what it renames, and how many boxes change
}

function summarise(changes: number, pages: number, listed: number): string {
  const first =
    changes === 0
      ? 'No box would change: only the KB changes.'
      : `${counted(changes, 'box', 'boxes')} would change, on ${counted(pages, 'page')}.`
  const second = listed === 0 ? '' : ` ${counted(listed, 'box', 'boxes')} ${listed === 1 ? 'is' : 'are'} listed to check by hand, not changed.`
  return first + second
}

/** A settled proposal is named as the queues name it. An id the KB shown does not have (it was settled or decided in the
 *  meantime) is named by its id. */
function settledWords(ids: readonly string[], kb: KBView): Words[] {
  const queued = [...kb.human_review_queue, ...kb.proposal_queue]
  return ids.map((id) => {
    const p = queued.find((q) => q.proposal_id === id)
    return p === undefined ? [`proposal ${id.slice(0, 8)}`] : proposalLabel(p)
  })
}

export function describePreview(plan: RenamePreview, typed: string, kb: KBView): PreviewWords {
  const found = rowIn(kb, plan.kind, plan.id)
  const label = rowLabel(plan.kind, plan.id)
  const pages = groupByPage(plan.changes)
  return {
    title: [renameCardName(plan.kind, plan.id), ...(found === null ? [] : [' (', ja(found.source), ')']), ': ', zh(plan.before), ' → ', zh(plan.after)],
    stored:
      typed === plan.after
        ? null
        : ['You typed ', zh(typed), '. The KB stores it as ', zh(plan.after), ' (trimmed, and normalised to NFKC).'],
    approves: found?.status === 'pending_human' ? `${label} waits for a person: renaming it also approves it.` : null,
    summary: summarise(plan.changes.length, pages.length, plan.listed.length),
    pages,
    listed: describeListed(plan.listed, plan.before),
    settles: settledWords(plan.settles, kb),
    confirm: `${CONFIRM_LABEL}: ${label.toLowerCase()} to ${plan.after}, changing ${counted(plan.changes.length, 'box', 'boxes')}`,
  }
}

/** Said under a preview when the field holds something else than the preview was made for: the confirm writes the rendering of
 *  the preview, not the field's, so the person is told which one it is. Null when they agree, and for a rendering that is not
 *  typed (a conflict's own). */
export function previewHint(draft: string, previewed: string, typed: boolean): string | null {
  if (!typed || draft === previewed) return null
  return 'The field has changed since this preview. Confirm renames to the rendering above, not to the field; Preview again to see what the field would do.'
}

// ---- The announcement ---------------------------------------------------------------------------------------------------------

/** What a rename did, from the write's answer: what was renamed, how many boxes changed on how many pages, how many are left to
 *  check by hand, and how many conflicts were settled. Each count says one and many right, and zero is many. An entry settles
 *  the term conflicts that now agree with it; a profile settles only the new-name proposal it was made from, which is no
 *  conflict, so that is worded as what it is. */
export function describeRename(out: RenameOut): Words {
  const pages = new Set(out.changes.map((c) => c.page_index)).size
  const left = out.listed.length
  const settled = out.settles.length
  const settledSentence =
    out.kind === 'profile' && settled > 0
      ? `${counted(settled, 'new-name proposal')} ${settled === 1 ? 'was' : 'were'} settled.`
      : `${counted(settled, 'conflict')} ${settled === 1 ? 'was' : 'were'} settled.`
  return [
    ...named(out.kb, out.kind, out.id),
    ' was renamed by a person: ',
    zh(out.before),
    ' → ',
    zh(out.after),
    `. ${counted(out.changes.length, 'box', 'boxes')} changed, on ${counted(pages, 'page')}. `,
    `${counted(left, 'box', 'boxes')} ${left === 1 ? 'is' : 'are'} left to check by hand. `,
    settledSentence,
  ]
}

/** The boxes a rename left to check by hand, shown under the row until the next request: null when there are none. */
export interface LeftWords {
  heading: string
  boxes: ListedBox[]
}

export function describeLeft(out: RenameOut): LeftWords | null {
  if (out.listed.length === 0) return null
  return { heading: `Left to check by hand (${out.listed.length}):`, boxes: describeListed(out.listed, out.before) }
}
