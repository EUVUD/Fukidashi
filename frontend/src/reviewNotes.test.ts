import { describe, expect, it } from 'vitest'
import type { PageReview, ReviewFlag, ReviewIssue } from './api'
import {
  aboutText,
  describeCounts,
  describeEdited,
  describeEnd,
  describeRegion,
  flagLabel,
  groupNotes,
  issueLabel,
  noteAbout,
  notesOf,
  reviewLine,
  reviewSubject,
} from './reviewNotes'

// ---- Fixtures: a review as the backend sends it (backend/fukidashi/review_notes.py) --------------------------------------

const flag = (region: number, pass: number, current: boolean, reason = `reason of ${region}`): ReviewFlag => ({
  region,
  reason,
  pass,
  current,
})

const issue = (
  region: number,
  review: number,
  verdict: string | null,
  current: boolean,
  suggestion: string | null = null,
  comment = `comment on ${region}`,
): ReviewIssue => ({ region, comment, suggestion, verdict, review, current })

const review = (over: Partial<PageReview> = {}): PageReview => ({
  run_id: 448,
  config: 'full',
  send_backs: 1,
  final_stop: 'approved',
  edited: [],
  flags: [],
  issues: [],
  ...over,
})

/** `tojime_no_siora/018` (Run 448): a flag on the first pass, a review that sent the page back, a revision that flagged
 *  nothing, and a second review that approved with notes on three regions. */
const sentBackOnce = (over: Partial<PageReview> = {}): PageReview =>
  review({
    flags: [flag(9, 0, false)],
    issues: [
      issue(9, 1, 'revise', false, '不过……'),
      issue(3, 2, 'approve', true),
      issue(6, 2, 'approve', true),
      issue(8, 2, 'approve', true),
    ],
    ...over,
  })

const STOPS = ['approved', 'review_unusable', 'revision_unusable', 'send_back_limit']

// ---- How the review ended ---------------------------------------------------------------------------------------------

describe('how the review ended', () => {
  it('says on which review a page was approved and after how many send-backs', () => {
    expect(describeEnd('approved', 0)).toBe('Approved on review 1, with no send-backs.')
    expect(describeEnd('approved', 1)).toBe('Approved on review 2, after 1 send-back.')
    expect(describeEnd('approved', 2)).toBe('Approved on review 3, after 2 send-backs.') // the Orchestrator never gets here
  })

  it('says that the send-back limit was reached and that the last revision was not reviewed', () => {
    expect(describeEnd('send_back_limit', 2)).toBe(
      'Stopped at the send-back limit after 2 send-backs: the last revision was not reviewed.',
    )
  })

  it('says that a review gave no verdict and the translation stands, for the first review and a later one', () => {
    expect(describeEnd('review_unusable', 0)).toBe('Review 1 gave no verdict, so the translation stands.')
    expect(describeEnd('review_unusable', 1)).toBe('Review 2 gave no verdict, so the translation stands.')
  })

  it('names the review that sent the page back when its revision was unusable, and that the reviewed translation stands', () => {
    expect(describeEnd('revision_unusable', 1)).toBe(
      'Review 1 sent the page back, but its revision could not be used: the reviewed translation stands.',
    )
    expect(describeEnd('revision_unusable', 2)).toBe(
      'Review 2 sent the page back, but its revision could not be used: the reviewed translation stands.',
    )
  })

  it('words nothing but `approved` as approved, whatever the number of send-backs', () => {
    for (const stop of STOPS.filter((s) => s !== 'approved')) {
      for (const sendBacks of [0, 1, 2]) {
        expect(describeEnd(stop, sendBacks), `${stop} after ${sendBacks}`).not.toMatch(/approv/i)
      }
    }
    expect(describeEnd('approved', 1)).toMatch(/^Approved/)
    expect(describeEnd('a_stop_from_the_future', 1)).not.toMatch(/approv/i) // an unknown stop is not approved either
    expect(describeEnd('a_stop_from_the_future', 1)).toContain('a_stop_from_the_future')
  })

  it('gives each stop its own words', () => {
    const lines = STOPS.map((stop) => describeEnd(stop, 1))
    expect(new Set(lines).size).toBe(STOPS.length)
  })
})

describe('the review line', () => {
  it('puts the ending first and then the counts, and says which notes are about the translation on show', () => {
    expect(reviewLine(sentBackOnce())).toBe(
      'Approved on review 2, after 1 send-back. 4 Issues (3 about this translation), 1 Flag (none about this translation).',
    )
  })

  it('leaves out "about this translation" when every note is', () => {
    const all = review({ send_backs: 0, flags: [flag(2, 0, true), flag(4, 0, true)], issues: [issue(2, 1, 'approve', true)] })
    expect(describeCounts(all)).toBe('1 Issue, 2 Flags.')
  })

  it('counts none, and says so, rather than 0', () => {
    expect(describeCounts(review())).toBe('no Issues, no Flags.')
    expect(describeCounts(review({ issues: [issue(1, 1, 'revise', false)] }))).toBe(
      '1 Issue (none about this translation), no Flags.',
    )
  })

  it('names the regions a person edited, and only when there are some', () => {
    expect(describeEdited([])).toBeNull()
    expect(describeEdited([8])).toBe('A person edited region 8.')
    expect(describeEdited([3, 8])).toBe('A person edited regions 3 and 8.')
    expect(describeEdited([3, 8, 9])).toBe('A person edited regions 3, 8 and 9.')
    expect(reviewLine(sentBackOnce({ edited: [8] }))).toMatch(/ A person edited region 8\.$/)
    expect(reviewLine(sentBackOnce())).not.toMatch(/edited/)
  })

  it('says for every stop how the review ended, with the counts after it', () => {
    for (const stop of STOPS) {
      const line = reviewLine(review({ final_stop: stop, send_backs: 1 }))
      expect(line.startsWith(describeEnd(stop, 1))).toBe(true)
      expect(line.endsWith('no Issues, no Flags.')).toBe(true)
    }
  })

  it('names the run the review is of, which is not always the run on show', () => {
    expect(reviewSubject(review({ run_id: 506, config: 'full_no_kb' }))).toBe('Review of Run 506 (full_no_kb)')
  })
})

// ---- A note's words -----------------------------------------------------------------------------------------------------

describe("a note's label", () => {
  it('says that a Flag is of the first pass or of which revision', () => {
    expect(flagLabel(0)).toBe('Flag · first pass')
    expect(flagLabel(1)).toBe('Flag · revision 1')
    expect(flagLabel(2)).toBe('Flag · revision 2')
  })

  it('says which review an Issue is from, and whether it sent the page back or came with an approval', () => {
    expect(issueLabel(1, 'revise')).toBe('Review 1 · revise')
    expect(issueLabel(2, 'approve')).toBe('Review 2 · note on approval')
    expect(issueLabel(2, 'revise')).toBe('Review 2 · revise')
    expect(issueLabel(1, null)).toBe('Review 1 · no verdict')
  })

  it('is on every note, from the kind and the round the backend gave', () => {
    const notes = notesOf(
      review({ flags: [flag(9, 0, false), flag(2, 1, true)], issues: [issue(9, 1, 'revise', false, 'x'), issue(2, 2, 'approve', true)] }),
    )
    expect(notes.map((n) => n.label)).toEqual(['Flag · first pass', 'Review 1 · revise', 'Flag · revision 1', 'Review 2 · note on approval'])
  })
})

describe('a note that is not about the translation on show', () => {
  it('says it was about an earlier translation, and a note that is current says nothing', () => {
    expect(noteAbout(true, 3, [])).toBe('current')
    expect(noteAbout(false, 9, [])).toBe('earlier')
    expect(aboutText('current')).toBeNull()
    expect(aboutText('earlier')).toMatch(/earlier translation/)
  })

  it('says, on an edited region, that it was about the text before the edit', () => {
    expect(noteAbout(false, 8, [8])).toBe('edited')
    expect(noteAbout(true, 8, [8])).toBe('edited') // the region's text is a person's now, whatever the stage said
    expect(aboutText('edited')).toMatch(/before a person's edit/)
    expect(aboutText('edited')).not.toBe(aboutText('earlier'))
  })

  it('is told apart from a current one on the notes themselves', () => {
    const notes = notesOf(sentBackOnce())
    expect(notes.map((n) => [n.region, n.about])).toEqual([
      [9, 'earlier'], // the first pass's Flag
      [9, 'earlier'], // review 1's Issue, which sent the page back
      [3, 'current'],
      [6, 'current'],
      [8, 'current'],
    ])
  })

  it('marks every note of an edited region as about the text before the edit, whichever stage made it', () => {
    const edited = sentBackOnce({ edited: [8, 9], flags: [flag(9, 0, false), flag(8, 1, false)], issues: [issue(9, 1, 'revise', false), issue(8, 2, 'approve', false), issue(3, 2, 'approve', true)] })
    expect(notesOf(edited).map((n) => [n.region, n.about])).toEqual([
      [9, 'edited'],
      [9, 'edited'],
      [8, 'edited'],
      [8, 'edited'],
      [3, 'current'],
    ])
  })

  it('keeps what the note says, and its suggestion, whether or not it is current', () => {
    const [first] = notesOf(sentBackOnce())
    const [, sentBack] = notesOf(sentBackOnce())
    expect(first).toMatchObject({ kind: 'flag', text: 'reason of 9', suggestion: null })
    expect(sentBack).toMatchObject({ kind: 'issue', text: 'comment on 9', suggestion: '不过……' })
  })
})

// ---- By region -----------------------------------------------------------------------------------------------------------

describe('grouping the notes by region', () => {
  // The lists are not in region order, and each region has notes in both and in several stages.
  const mixed = review({
    flags: [flag(9, 0, false), flag(3, 1, true)],
    issues: [issue(3, 1, 'revise', false, 'a'), issue(9, 1, 'revise', false, 'b'), issue(3, 2, 'approve', true), issue(5, 2, 'approve', true)],
  })

  it('gives each region the notes about it and no others, whatever the order of the notes', () => {
    const groups = groupNotes(mixed)
    expect([...groups.keys()].sort((a, b) => a - b)).toEqual([3, 5, 9])
    for (const [region, group] of groups) {
      expect(group.notes.every((n) => n.region === region), `region ${region}`).toBe(true)
    }
    expect(groups.get(9)?.notes.map((n) => n.label)).toEqual(['Flag · first pass', 'Review 1 · revise'])
    expect(groups.get(5)?.notes.map((n) => n.label)).toEqual(['Review 2 · note on approval'])
  })

  it("reads a region's notes in the order the run made them, whichever list they came from", () => {
    expect(groupNotes(mixed).get(3)?.notes.map((n) => n.label)).toEqual([
      'Review 1 · revise',
      'Flag · revision 1',
      'Review 2 · note on approval',
    ])
  })

  it('counts a region\'s Flags, Issues and the notes about the translation on show', () => {
    const groups = groupNotes(mixed)
    expect(groups.get(3)).toMatchObject({ flags: 1, issues: 2, current: 2 })
    expect(groups.get(9)).toMatchObject({ flags: 1, issues: 1, current: 0 })
    expect(groups.get(5)).toMatchObject({ flags: 0, issues: 1, current: 1 })
  })

  it('keeps the notes of one stage in the order the run stored them', () => {
    const one = review({ issues: [issue(7, 1, 'approve', true, null, 'first'), issue(2, 1, 'approve', true, null, 'second'), issue(7, 1, 'approve', true, null, 'third')] })
    expect(groupNotes(one).get(7)?.notes.map((n) => n.text)).toEqual(['first', 'third'])
  })

  it('has no group for a page with no review, or a review with no notes', () => {
    expect(groupNotes(null).size).toBe(0)
    expect(groupNotes(review()).size).toBe(0)
  })

  it('does not change the review it was given', () => {
    const given = sentBackOnce()
    const before = JSON.stringify(given)
    groupNotes(given)
    notesOf(given)
    reviewLine(given)
    expect(JSON.stringify(given)).toBe(before)
  })
})

describe('what a marker says', () => {
  it('counts the Flags and the Issues of the region', () => {
    expect(describeRegion({ notes: [], flags: 1, issues: 2, current: 3 })).toBe('1 Flag, 2 Issues')
    expect(describeRegion({ notes: [], flags: 0, issues: 1, current: 1 })).toBe('1 Issue')
    expect(describeRegion({ notes: [], flags: 2, issues: 0, current: 2 })).toBe('2 Flags')
  })

  it('says when none of them is about the translation on show', () => {
    expect(describeRegion({ notes: [], flags: 1, issues: 1, current: 0 })).toBe('1 Flag, 1 Issue, none about the translation on show')
  })
})
