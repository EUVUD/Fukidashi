import { describe, expect, it } from 'vitest'
import type { KBEntry, KBProfile, KBProposal, KBView, RenameBox, RenameOut, RenamePreview, ReviewProposal } from './api'
import { plain, proposalLabel } from './kbView'
import {
  CANCEL_LABEL,
  CONFIRM_LABEL,
  PREVIEW_LABEL,
  STALE_NOTE,
  describeLeft,
  describeListed,
  describePreview,
  describeRename,
  groupByPage,
  previewHint,
  renameCardName,
  rowLabel,
} from './renameView'

// ---- Fixtures: what the backend answers (backend/fukidashi/api/kb.py), over `tojime_no_siora`'s Editor KB as task 042's check built it ----

const AT = '2026-11-18T07:00:00+00:00'

const entry = (id: number, source: string, target: string, status = 'approved'): KBEntry => ({
  id,
  source_term: source,
  target_term: target,
  category: '',
  status,
  provenance: 'agent',
  evidence_pages: [4],
  history: [{ at: AT, event: 'created', page: 4 }],
})

const profile = (id: number, name: string, rendering: string, status = 'approved', aliases: string[] = []): KBProfile => ({
  id,
  name_src: name,
  name_tgt: rendering,
  aliases,
  notes: '',
  status,
  provenance: 'agent',
  evidence_pages: [1],
  history: [{ at: AT, event: 'created', page: 1 }],
})

const conflict = (id: string, source: string, target: string, page: number): ReviewProposal => ({
  proposal_id: id,
  kind: 'term',
  payload: { source_term: source, target_term: target, category: '' },
  proposer: 'translator',
  page_index: page,
  run_id: 437,
  reason: 'conflicts with entry 40',
  waits_for: 'conflict',
  profile_id: null,
})

const newName = (id: string, name: string, rendering: string, page: number): ReviewProposal => ({
  proposal_id: id,
  kind: 'character',
  payload: { name_src: name, name_tgt: rendering, aliases: [], notes: '' },
  proposer: 'translator',
  page_index: page,
  run_id: 431,
  reason: 'a new name waits for a human',
  waits_for: 'new_name',
  profile_id: 38,
})

const kbOf = (over: Partial<KBView> = {}): KBView => ({
  kb_id: 11,
  entries: [entry(40, 'トルティヤー一家', '托尔蒂亚家族'), entry(41, '結界', '结界', 'pending_human')],
  profiles: [profile(38, 'シオラ', '希奥拉', 'pending_human', ['綴じ眼のシオラ'])],
  proposal_queue: [],
  human_review_queue: [conflict('c6f34baec105729df518f50d8ef4e37e', 'トルティヤー一家', '托尔蒂亚一家', 7), newName('6a45decc', 'シオラ', '希奥拉', 1)],
  proposal_counts: { proposed: 0, approved: 0, rejected: 0, pending_human: 2 },
  ...over,
})

const box = (page_index: number, idx: number, text_ja: string, before: string | null, after: string | null): RenameBox => ({
  page_index,
  idx,
  text_ja,
  before,
  after,
})

const CHANGED = box(4, 5, 'ここら一帯を仕切るトルティヤー一家の名が聞こえたので', '我听说这一带都由托尔蒂亚家族掌管……', '我听说这一带都由托尔蒂亚一家掌管……')
const LISTED = box(7, 4, 'トルティヤー一家に枪を売るのか', '居然想让靠枪战起家的托尔蒂亚一家买你的枪法？', null)
const UNTRANSLATED = box(9, 1, 'トルティヤー一家だ', null, null)

const plan = (over: Partial<RenamePreview> = {}): RenamePreview => ({
  kind: 'entry',
  id: 40,
  before: '托尔蒂亚家族',
  after: '托尔蒂亚一家',
  changes: [],
  listed: [],
  settles: [],
  ...over,
})

const out = (over: Partial<RenameOut> = {}): RenameOut => ({ ...plan(), runs: [], kb: kbOf(), ...over })

const text = (words: readonly (string | { text: string; lang: string })[]) => plain(words as never)

// ---- The preview ---------------------------------------------------------------------------------------------------------------

describe('the boxes of a preview, grouped by page', () => {
  it('puts the boxes of one page together, pages and boxes in the order the server sent them', () => {
    const boxes = [box(4, 2, 'あ', 'a', 'b'), box(4, 5, 'い', 'c', 'd'), box(7, 1, 'う', 'e', 'f'), box(12, 0, 'え', 'g', 'h')]
    expect(groupByPage(boxes).map((g) => [g.page, g.boxes.map((b) => b.idx)])).toEqual([[4, [2, 5]], [7, [1]], [12, [0]]])
  })

  it('gives no group for no box', () => {
    expect(groupByPage([])).toEqual([])
  })

  it('words each box with its Japanese and both translations, each in its language', () => {
    const [group] = groupByPage([CHANGED])
    expect(group.boxes[0]).toEqual({
      idx: 5,
      ja: [{ text: 'ここら一帯を仕切るトルティヤー一家の名が聞こえたので', lang: 'ja' }],
      before: [{ text: '我听说这一带都由托尔蒂亚家族掌管……', lang: 'zh-Hans' }],
      after: [{ text: '我听说这一带都由托尔蒂亚一家掌管……', lang: 'zh-Hans' }],
    })
  })

  it('leaves out what a box does not have: no translation, and no new one for a box that is only listed', () => {
    const [group] = groupByPage([UNTRANSLATED, LISTED])
    expect(group.boxes.map((b) => [b.idx, b.before === null, b.after === null])).toEqual([[1, true, true]])
    const [listed] = groupByPage([LISTED])
    expect([listed.boxes[0].before, listed.boxes[0].after]).toEqual([[{ text: LISTED.before, lang: 'zh-Hans' }], null])
  })
})

describe('the boxes that are listed, and why', () => {
  it('says that a translation does not use the old rendering, naming it in Simplified Chinese', () => {
    const [listed] = describeListed([LISTED], '托尔蒂亚家族')
    expect(text(listed.why)).toBe('Its translation does not use 托尔蒂亚家族.')
    expect(listed.why).toContainEqual({ text: '托尔蒂亚家族', lang: 'zh-Hans' })
    expect([listed.page, listed.box.idx]).toEqual([7, 4])
    expect(listed.box.ja).toEqual([{ text: LISTED.text_ja, lang: 'ja' }])
    expect(listed.box.before).toEqual([{ text: LISTED.before, lang: 'zh-Hans' }])
    expect(listed.box.after).toBeNull()
  })

  it('says that a page has no translation when the box has none, and does not name the old rendering', () => {
    const [listed] = describeListed([UNTRANSLATED], '托尔蒂亚家族')
    expect(text(listed.why)).toBe('Its page has no translation yet.')
  })

  it('tells the two reasons apart, in a list that holds both', () => {
    const reasons = describeListed([LISTED, UNTRANSLATED], '托尔蒂亚家族').map((l) => text(l.why))
    expect(new Set(reasons).size).toBe(2)
  })

  it('takes an empty translation, which is not none, for one that does not use the old rendering', () => {
    expect(text(describeListed([box(3, 0, 'あ', '', null)], '甲')[0].why)).toBe('Its translation does not use 甲.')
  })
})

describe('what a preview says', () => {
  it('words the rendering before → after, the row named by its source in Japanese', () => {
    const words = describePreview(plan(), '托尔蒂亚一家', kbOf())
    expect(text(words.title)).toBe('Rename entry 40 (トルティヤー一家): 托尔蒂亚家族 → 托尔蒂亚一家')
    expect(words.title).toContainEqual({ text: 'トルティヤー一家', lang: 'ja' })
    expect(words.title).toContainEqual({ text: '托尔蒂亚家族', lang: 'zh-Hans' })
    expect(words.title).toContainEqual({ text: '托尔蒂亚一家', lang: 'zh-Hans' })
  })

  it('words a profile\'s rename the same way, with the name of the profile', () => {
    const words = describePreview(plan({ kind: 'profile', id: 38, before: '希奥拉', after: '希欧拉' }), '希欧拉', kbOf())
    expect(text(words.title)).toBe('Rename profile 38 (シオラ): 希奥拉 → 希欧拉')
  })

  it('names the row by its kind and id when the KB shown does not have it', () => {
    const words = describePreview(plan({ id: 99 }), '托尔蒂亚一家', kbOf())
    expect(text(words.title)).toBe('Rename entry 99: 托尔蒂亚家族 → 托尔蒂亚一家')
  })

  it('says nothing about the stored form when it is what was typed', () => {
    expect(describePreview(plan(), '托尔蒂亚一家', kbOf()).stored).toBeNull()
  })

  it('says how the KB stores the rendering when that is not what was typed: spaces at the ends, and the normal form', () => {
    const padded = describePreview(plan(), '  托尔蒂亚一家 ', kbOf()).stored
    expect(padded).not.toBeNull()
    expect(text(padded!)).toBe('You typed   托尔蒂亚一家 . The KB stores it as 托尔蒂亚一家 (trimmed, and normalised to NFKC).')
    expect(padded).toContainEqual({ text: '托尔蒂亚一家', lang: 'zh-Hans' })
    const wide = describePreview(plan({ after: 'ABC' }), 'ＡＢＣ', kbOf()).stored
    expect(text(wide!)).toContain('The KB stores it as ABC')
  })

  it('says that renaming a row that waits for a person approves it, and nothing of the kind for an approved row', () => {
    expect(describePreview(plan(), '托尔蒂亚一家', kbOf()).approves).toBeNull()
    const waiting = describePreview(plan({ id: 41, before: '结界', after: '屏障' }), '屏障', kbOf())
    expect(waiting.approves).toBe('Entry 41 waits for a person: renaming it also approves it.')
    const name = describePreview(plan({ kind: 'profile', id: 38, before: '希奥拉', after: '希欧拉' }), '希欧拉', kbOf())
    expect(name.approves).toBe('Profile 38 waits for a person: renaming it also approves it.')
  })

  it('groups the changes by page, each box with its Japanese and its before and after', () => {
    const second = box(4, 8, 'あ', '托尔蒂亚家族来了', '托尔蒂亚一家来了')
    const words = describePreview(plan({ changes: [CHANGED, second, box(7, 4, 'い', '托尔蒂亚家族', '托尔蒂亚一家')] }), '托尔蒂亚一家', kbOf())
    expect(words.pages.map((g) => [g.page, g.boxes.map((b) => b.idx)])).toEqual([[4, [5, 8]], [7, [4]]])
    expect(words.pages[0].boxes[0]).toEqual(groupByPage([CHANGED])[0].boxes[0])
  })

  it('lists the boxes that are only listed, each with its reason', () => {
    const words = describePreview(plan({ listed: [LISTED, UNTRANSLATED] }), '托尔蒂亚一家', kbOf())
    expect(words.listed.map((l) => [l.page, text(l.why)])).toEqual([
      [7, 'Its translation does not use 托尔蒂亚家族.'],
      [9, 'Its page has no translation yet.'],
    ])
  })

  it('names the proposals it settles with their pair and page, as the queues do', () => {
    const kb = kbOf()
    const words = describePreview(plan({ settles: ['c6f34baec105729df518f50d8ef4e37e'] }), '托尔蒂亚一家', kb)
    expect(words.settles).toHaveLength(1)
    expect(text(words.settles[0])).toBe('トルティヤー一家 → 托尔蒂亚一家 (proposed on page 7)')
    expect(words.settles[0]).toEqual(proposalLabel(kb.human_review_queue[0]))
    expect(words.settles[0]).toContainEqual({ text: 'トルティヤー一家', lang: 'ja' })
    expect(words.settles[0]).toContainEqual({ text: '托尔蒂亚一家', lang: 'zh-Hans' })
  })

  it('names a profile\'s new-name proposal, and a proposal of the Proposal Queue, the same way', () => {
    const undecided: KBProposal = {
      proposal_id: 'u1',
      kind: 'term',
      payload: { source_term: '呪文', target_term: '咒语', category: '' },
      proposer: 'translator',
      page_index: 3,
      run_id: 5,
      reason: null,
    }
    const kb = kbOf({ proposal_queue: [undecided] })
    const words = describePreview(plan({ kind: 'profile', id: 38, before: '希奥拉', after: '希欧拉', settles: ['6a45decc', 'u1'] }), '希欧拉', kb)
    expect(words.settles.map(text)).toEqual(['シオラ → 希奥拉 (proposed on page 1)', '呪文 → 咒语 (proposed on page 3)'])
  })

  it('names a proposal the KB shown does not have by the start of its id', () => {
    const words = describePreview(plan({ settles: ['0123456789abcdef'] }), '托尔蒂亚一家', kbOf())
    expect(words.settles.map(text)).toEqual(['proposal 01234567'])
  })

  it('settles nothing by default', () => {
    expect(describePreview(plan(), '托尔蒂亚一家', kbOf()).settles).toEqual([])
  })
})

describe('what a preview says about how many boxes', () => {
  const summary = (changes: RenameBox[], listed: RenameBox[]) => describePreview(plan({ changes, listed }), '托尔蒂亚一家', kbOf()).summary
  const many = (page: number, count: number) => Array.from({ length: count }, (_, i) => box(page, i, 'あ', '甲', '乙'))

  it('says that only the KB changes when no box does', () => {
    expect(summary([], [])).toBe('No box would change: only the KB changes.')
    expect(summary([], [])).toMatch(/only the KB changes/)
  })

  it('counts the boxes and the pages, one and many right', () => {
    expect(summary(many(4, 1), [])).toBe('1 box would change, on 1 page.')
    expect(summary([...many(4, 2), ...many(7, 1)], [])).toBe('3 boxes would change, on 2 pages.')
  })

  it('adds the boxes that are only listed, and says that they are not changed', () => {
    expect(summary(many(4, 1), [LISTED])).toBe('1 box would change, on 1 page. 1 box is listed to check by hand, not changed.')
    expect(summary(many(4, 2), [LISTED, UNTRANSLATED])).toBe('2 boxes would change, on 1 page. 2 boxes are listed to check by hand, not changed.')
  })

  it('says that only the KB changes even when some boxes are listed', () => {
    expect(summary([], [LISTED])).toBe('No box would change: only the KB changes. 1 box is listed to check by hand, not changed.')
  })
})

describe('the confirm button of a preview', () => {
  const confirm = (changes: RenameBox[], over: Partial<RenamePreview> = {}) =>
    describePreview(plan({ changes, ...over }), '托尔蒂亚一家', kbOf()).confirm

  it('has a name that says what it renames and how many boxes change', () => {
    expect(confirm([CHANGED])).toBe('Confirm rename: entry 40 to 托尔蒂亚一家, changing 1 box')
    expect(confirm([CHANGED, CHANGED, CHANGED])).toBe('Confirm rename: entry 40 to 托尔蒂亚一家, changing 3 boxes')
    expect(confirm([])).toBe('Confirm rename: entry 40 to 托尔蒂亚一家, changing 0 boxes')
    expect(confirm([CHANGED], { kind: 'profile', id: 38, after: '希欧拉' })).toBe('Confirm rename: profile 38 to 希欧拉, changing 1 box')
  })

  it('starts with what the button says, and names the stored rendering, not what was typed', () => {
    const name = describePreview(plan(), '  托尔蒂亚一家', kbOf()).confirm
    expect(name.startsWith(CONFIRM_LABEL)).toBe(true)
    expect(name).toContain('to 托尔蒂亚一家,')
  })

  it('differs between two rows, and between two renderings', () => {
    expect(new Set([confirm([]), confirm([], { id: 41 }), confirm([], { after: '托尔蒂亚世家' })]).size).toBe(3)
  })
})

describe('the words on the card\'s buttons', () => {
  it('are the ones the task names, and differ', () => {
    expect([PREVIEW_LABEL, CANCEL_LABEL, CONFIRM_LABEL]).toEqual(['Preview', 'Cancel', 'Confirm rename'])
  })

  it('say that a dropped preview may no longer be what the rename would do, and what to do', () => {
    expect(STALE_NOTE).toMatch(/dropped/)
    expect(STALE_NOTE).toMatch(/no longer be what the rename would do/)
    expect(STALE_NOTE).toMatch(/Preview again/)
  })

  it('say that the KB was read again or changed, for a preview is dropped by a Refresh as well as by a decision', () => {
    expect(STALE_NOTE).toMatch(/read again or changed/)
    expect(STALE_NOTE).not.toMatch(/changed since/) // a Refresh need not have found anything changed
  })

  it('label a row by its kind and id', () => {
    expect(rowLabel('entry', 40)).toBe('Entry 40')
    expect(rowLabel('profile', 38)).toBe('Profile 38')
  })

  it('name the card as the title of a preview does, in the middle of a sentence', () => {
    expect(renameCardName('entry', 40)).toBe('Rename entry 40')
    expect(renameCardName('profile', 38)).toBe('Rename profile 38')
    expect(describePreview(plan(), '托尔蒂亚一家', kbOf()).title).toContainEqual(renameCardName('entry', 40))
  })
})

describe('what the card says when the field and the preview disagree', () => {
  it('says nothing when the field holds what the preview was made for, or when there is no field', () => {
    expect(previewHint('托尔蒂亚一家', '托尔蒂亚一家', true)).toBeNull()
    expect(previewHint('anything', '托尔蒂亚一家', false)).toBeNull()
  })

  it('says which of the two the confirm writes, when they differ', () => {
    const hint = previewHint('托尔蒂亚世家', '托尔蒂亚一家', true)
    expect(hint).toMatch(/Confirm renames to the rendering above, not to the field/)
    expect(hint).toMatch(/Preview again/)
  })
})

// ---- The announcement: made from the write's answer ---------------------------------------------------------------------------------

describe('the announcement of a rename', () => {
  const said = (over: Partial<RenameOut> = {}) => text(describeRename(out(over)))
  const many = (page: number, count: number) => Array.from({ length: count }, (_, i) => box(page, i, 'あ', '甲', '乙'))

  it('says what was renamed, by a person, from which rendering to which', () => {
    expect(said()).toMatch(/^Entry 40 \(トルティヤー一家\) was renamed by a person: 托尔蒂亚家族 → 托尔蒂亚一家\. /)
    expect(text(describeRename(out({ kind: 'profile', id: 38, before: '希奥拉', after: '希欧拉' })))).toMatch(
      /^Profile 38 \(シオラ\) was renamed by a person: 希奥拉 → 希欧拉\. /,
    )
  })

  it('counts a single box, a single page, a single box left and a single conflict settled in the singular', () => {
    expect(said({ changes: [CHANGED], listed: [LISTED], settles: ['c1'], runs: [{ page_index: 4, run_id: 906 }] })).toBe(
      'Entry 40 (トルティヤー一家) was renamed by a person: 托尔蒂亚家族 → 托尔蒂亚一家. 1 box changed, on 1 page. 1 box is left to check by hand. 1 conflict was settled.',
    )
  })

  it('counts zero as many: 0 boxes, 0 pages, 0 conflicts', () => {
    expect(said()).toBe(
      'Entry 40 (トルティヤー一家) was renamed by a person: 托尔蒂亚家族 → 托尔蒂亚一家. 0 boxes changed, on 0 pages. 0 boxes are left to check by hand. 0 conflicts were settled.',
    )
  })

  it('counts several in the plural, the pages by the distinct pages the changes are on', () => {
    expect(said({ changes: [...many(4, 2), ...many(7, 1), ...many(9, 2)], listed: [LISTED, UNTRANSLATED], settles: ['c1', 'c2'] })).toBe(
      'Entry 40 (トルティヤー一家) was renamed by a person: 托尔蒂亚家族 → 托尔蒂亚一家. 5 boxes changed, on 3 pages. 2 boxes are left to check by hand. 2 conflicts were settled.',
    )
  })

  it('is made from the write\'s answer: its changes, its list and its settled conflicts, whatever a preview said', () => {
    const preview = plan({ changes: many(4, 3), listed: [LISTED], settles: ['c1'] })
    const written = out({ ...preview, changes: many(4, 1), listed: [], settles: [] }) // the book changed between the two
    expect(text(describeRename(written))).toMatch(/1 box changed, on 1 page\. 0 boxes are left to check by hand\. 0 conflicts were settled\.$/)
    expect(text(describeRename(written))).not.toMatch(/3 boxes/)
  })

  it('names a profile\'s settled new-name proposal as what it is, not as a conflict', () => {
    const profileOut = { kind: 'profile' as const, id: 38, before: '希奥拉', after: '希欧拉' }
    expect(text(describeRename(out({ ...profileOut, settles: ['6a45decc'] })))).toMatch(/1 new-name proposal was settled\.$/)
    expect(text(describeRename(out({ ...profileOut, settles: [] })))).toMatch(/0 conflicts were settled\.$/)
  })

  it('names the row by its kind and id when the KB it answers does not have it', () => {
    expect(said({ id: 99 })).toMatch(/^Entry 99 was renamed by a person: /)
  })

  it('keeps the two renderings in Simplified Chinese and the source in Japanese, so that the panel can mark them', () => {
    const words = describeRename(out())
    expect(words).toContainEqual({ text: 'トルティヤー一家', lang: 'ja' })
    expect(words).toContainEqual({ text: '托尔蒂亚家族', lang: 'zh-Hans' })
    expect(words).toContainEqual({ text: '托尔蒂亚一家', lang: 'zh-Hans' })
  })

  it('does not word a rename as approved or rejected: it says renamed', () => {
    expect(said()).not.toMatch(/approv|reject/i)
  })
})

describe('the boxes a rename left to check by hand', () => {
  it('has none to show when the write listed none', () => {
    expect(describeLeft(out())).toBeNull()
    expect(describeLeft(out({ changes: [CHANGED] }))).toBeNull()
  })

  it('counts them, and gives each with the reason, from the write\'s answer', () => {
    const left = describeLeft(out({ listed: [LISTED, UNTRANSLATED] }))
    expect(left?.heading).toBe('Left to check by hand (2):')
    expect(left?.boxes.map((l) => [l.page, l.box.idx, text(l.why)])).toEqual([
      [7, 4, 'Its translation does not use 托尔蒂亚家族.'],
      [9, 1, 'Its page has no translation yet.'],
    ])
  })

  it('counts one box as one', () => {
    expect(describeLeft(out({ listed: [LISTED] }))?.heading).toBe('Left to check by hand (1):')
  })
})
