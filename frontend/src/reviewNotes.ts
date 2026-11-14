// What the page says about its review, in words and by region: the Translator's Flags and the Reviewer's Issues of the run
// whose stages produced the displayed translation (OUTLINE §0, Flag and Verdict / Issue). Pure logic with no React and no DOM.
// Only types come from './api', so the module loads anywhere and Vitest checks it directly (OUTLINE §8, "Frontend tests").
// The notes and what is current are the backend's word (backend/fukidashi/review_notes.py); the wording is here.
import type { PageReview } from './api'

// ---- A note: a Flag or an Issue, with its words -----------------------------------------------------------------------

export type NoteKind = 'flag' | 'issue'

/** What a note was about. `current`: the translation on show. `earlier`: a translation the Orchestrator replaced (the
 *  revision that followed a send-back). `edited`: the text before a person's edit, which holds for every note of an edited
 *  region, whichever stage made it. */
export type NoteAbout = 'current' | 'earlier' | 'edited'

export interface Note {
  kind: NoteKind
  region: number
  label: string // its kind and round: "Flag · first pass", "Review 2 · note on approval"
  text: string // the Flag's reason or the Issue's comment, in Simplified Chinese
  suggestion: string | null // Issues only
  about: NoteAbout
}

/** The notes of one region: in the order the run made them, with the counts the marker and its tooltip need. */
export interface RegionNotes {
  notes: Note[]
  flags: number
  issues: number
  current: number // how many are about the translation on show
}

/** "Flag · first pass" for the first pass, "Flag · revision 2" for the second revision. */
export const flagLabel = (pass: number): string => (pass === 0 ? 'Flag · first pass' : `Flag · revision ${pass}`)

/** "Review 1 · revise" for an Issue of a review that sent the page back, "Review 2 · note on approval" for one that came with
 *  an approval. A review with no verdict has no Issues, but the label still says so. */
export function issueLabel(review: number, verdict: string | null): string {
  const what = verdict === 'revise' ? 'revise' : verdict === 'approve' ? 'note on approval' : (verdict ?? 'no verdict')
  return `Review ${review} · ${what}`
}

/** What a note was about: an edited region says so whatever the note's own `current` is, because the text it was about is gone. */
export function noteAbout(current: boolean, region: number, edited: readonly number[]): NoteAbout {
  if (edited.includes(region)) return 'edited'
  return current ? 'current' : 'earlier'
}

/** The words for a note that is not about the translation on show; null for one that is. */
export function aboutText(about: NoteAbout): string | null {
  if (about === 'edited') return "About the text before a person's edit"
  if (about === 'earlier') return 'About an earlier translation'
  return null
}

// A run's stages are translate, review, revise, review, revise, ...: the pass of a Flag and the number of a review say
// where in that order the note was made, so the two lists can be read as one story.
const flagPlace = (pass: number) => 2 * pass // the first pass is stage 0, the nth revision stage 2n
const issuePlace = (review: number) => 2 * review - 1 // the nth review is stage 2n - 1

/** Every note of the review, in the order the run made them: a Flag of the first pass, the first review's Issues, a Flag of
 *  the first revision, the second review's Issues, and so on. Within one stage they keep the order the run stored them in. */
export function notesOf(review: PageReview): Note[] {
  const placed: { place: number; note: Note }[] = [
    ...review.flags.map((flag) => ({
      place: flagPlace(flag.pass),
      note: {
        kind: 'flag' as const,
        region: flag.region,
        label: flagLabel(flag.pass),
        text: flag.reason,
        suggestion: null,
        about: noteAbout(flag.current, flag.region, review.edited),
      },
    })),
    ...review.issues.map((issue) => ({
      place: issuePlace(issue.review),
      note: {
        kind: 'issue' as const,
        region: issue.region,
        label: issueLabel(issue.review, issue.verdict),
        text: issue.comment,
        suggestion: issue.suggestion,
        about: noteAbout(issue.current, issue.region, review.edited),
      },
    })),
  ]
  return placed.sort((a, b) => a.place - b.place).map(({ note }) => note) // a stable sort: one stage's notes keep their order
}

/** The notes by the region they are about. A region with none is not in the map. */
export function groupNotes(review: PageReview | null): Map<number, RegionNotes> {
  const groups = new Map<number, RegionNotes>()
  if (review === null) return groups
  for (const note of notesOf(review)) {
    let group = groups.get(note.region)
    if (group === undefined) {
      group = { notes: [], flags: 0, issues: 0, current: 0 }
      groups.set(note.region, group)
    }
    group.notes.push(note)
    if (note.kind === 'flag') group.flags++
    else group.issues++
    if (note.about === 'current') group.current++
  }
  return groups
}

// ---- Counting words -----------------------------------------------------------------------------------------------

const counted = (n: number, one: string): string => `${n} ${one}${n === 1 ? '' : 's'}`

/** "3", "3 and 8", "3, 8 and 9". */
const andList = (items: readonly number[]): string =>
  items.length <= 1 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`

/** What a marker says on hover and to a screen reader: "1 Flag, 2 Issues", and whether any of them is about the translation on show. */
export function describeRegion(group: RegionNotes): string {
  const parts = [group.flags > 0 ? counted(group.flags, 'Flag') : null, group.issues > 0 ? counted(group.issues, 'Issue') : null]
  const what = parts.filter((part) => part !== null).join(', ')
  return group.current === 0 ? `${what}, none about the translation on show` : what
}

// ---- The review line ----------------------------------------------------------------------------------------------

/** How the review ended, from the run's `final_stop` and its number of send-backs. Only `approved` is ever worded as approved:
 *  a page that stopped at the send-back limit, or whose review or revision was unusable, was not.
 *  The review that judged the stored translation is number `sendBacks + 1`, except after an unusable revision: then the review
 *  that sent the page back is the last one (number `sendBacks`) and nothing judged what the revision would have been. */
export function describeEnd(finalStop: string, sendBacks: number): string {
  switch (finalStop) {
    case 'approved':
      return `Approved on review ${sendBacks + 1}, ${sendBacks === 0 ? 'with no send-backs' : `after ${counted(sendBacks, 'send-back')}`}.`
    case 'send_back_limit':
      return `Stopped at the send-back limit after ${counted(sendBacks, 'send-back')}: the last revision was not reviewed.`
    case 'review_unusable':
      return `Review ${sendBacks + 1} gave no verdict, so the translation stands.`
    case 'revision_unusable':
      return `Review ${sendBacks} sent the page back, but its revision could not be used: the reviewed translation stands.`
    default:
      return `The review ended with "${finalStop}".`
  }
}

/** "4 Issues (3 about this translation), 1 Flag (none about this translation)." */
export function describeCounts(review: PageReview): string {
  const part = (one: string, total: number, current: number): string => {
    if (total === 0) return `no ${one}s`
    if (current === total) return counted(total, one)
    return `${counted(total, one)} (${current === 0 ? 'none' : current} about this translation)`
  }
  const issues = part('Issue', review.issues.length, review.issues.filter((issue) => issue.current).length)
  const flags = part('Flag', review.flags.length, review.flags.filter((flag) => flag.current).length)
  return `${issues}, ${flags}.`
}

/** "A person edited region 8." for the regions edited since the run; null when none was. */
export function describeEdited(edited: readonly number[]): string | null {
  return edited.length === 0 ? null : `A person edited ${edited.length === 1 ? 'region' : 'regions'} ${andList(edited)}.`
}

/** The run the review is of: "Review of Run 448 (full)". It is not always the run on show: a person's edit sits on top of it. */
export const reviewSubject = (review: PageReview): string => `Review of Run ${review.run_id} (${review.config})`

/** The review line: how the review ended, how many Issues and Flags it left, and the regions a person edited since. */
export function reviewLine(review: PageReview): string {
  return [describeEnd(review.final_stop, review.send_backs), describeCounts(review), describeEdited(review.edited)]
    .filter((part) => part !== null)
    .join(' ')
}
