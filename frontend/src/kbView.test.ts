import { describe, expect, it } from 'vitest'
import type { HistoryLine, KBDecision, KBEntry, KBProfile, KBProposal, KBView, ReviewProposal } from './api'
import {
  CONFLICT_NOTE,
  EMPTY_PROPOSAL_QUEUE,
  NO_EDITOR_KB,
  NO_RIVAL,
  RENAME_LABEL,
  TAKE_LABEL,
  TAKE_NOTE,
  UNDECIDED_NAME_NOTE,
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
  isReadable,
  isRefusal,
  itemKey,
  offerForProposal,
  offerForRow,
  plain,
  profileControlName,
  proposalControlName,
  proposalLabel,
  proposalName,
  proposalWords,
  provenanceLabel,
  reasonText,
  renameControlName,
  rivalsOf,
  statusLabel,
  summaryLine,
  takeControlName,
  takeOffer,
} from './kbView'

// ---- Fixtures: an Editor KB as the backend sends it (backend/fukidashi/api/kb.py) ------------------------------------------

const AT = '2026-11-14T21:33:27+00:00'
const line = (event: string, extra: Record<string, unknown> = {}): HistoryLine => ({ at: AT, event, ...extra })

const entry = (id: number, source: string, target: string, status = 'approved', over: Partial<KBEntry> = {}): KBEntry => ({
  id,
  source_term: source,
  target_term: target,
  category: '',
  status,
  provenance: 'agent',
  evidence_pages: [18],
  history: [line('created', { page: 18, run_id: 448, proposal_id: `p${id}`, reason: 'a recurring spell' })],
  ...over,
})

const profile = (id: number, name: string, rendering: string, status = 'approved', over: Partial<KBProfile> = {}): KBProfile => ({
  id,
  name_src: name,
  name_tgt: rendering,
  aliases: [],
  notes: '',
  status,
  provenance: 'agent',
  evidence_pages: [18],
  history: [line('created', { page: 18, run_id: 448, proposal_id: `n${id}`, reason: 'a new name waits for a human' })],
  ...over,
})

const termProposal = (id: string, source: string, target: string, over: Partial<KBProposal> = {}): KBProposal => ({
  proposal_id: id,
  kind: 'term',
  payload: { source_term: source, target_term: target, category: '' },
  proposer: 'translator',
  page_index: 19,
  run_id: 448,
  reason: null,
  ...over,
})

const nameProposal = (id: string, name: string, rendering: string, over: Partial<KBProposal> = {}): KBProposal => ({
  proposal_id: id,
  kind: 'character',
  payload: { name_src: name, name_tgt: rendering, aliases: [], notes: '' },
  proposer: 'translator',
  page_index: 19,
  run_id: 448,
  reason: null,
  ...over,
})

const waiting = (p: KBProposal, waits_for: string, profile_id: number | null = null): ReviewProposal => ({
  ...p,
  reason: waits_for === 'new_name' ? 'a new name waits for a human' : 'conflicts with entry 40, which is 封印',
  waits_for,
  profile_id,
})

const kbOf = (over: Partial<KBView> = {}): KBView => ({
  kb_id: 11,
  entries: [],
  profiles: [],
  proposal_queue: [],
  human_review_queue: [],
  proposal_counts: { proposed: 0, approved: 0, rejected: 0, pending_human: 0 },
  ...over,
})

/** `tojime_no_siora`'s Editor KB as task 040's check builds it, after one rejection: the rows a conflict can collide with,
 *  among rows that cannot be collided with (rejected, undecided) and one that waits for a person. */
const kb = kbOf({
  entries: [
    entry(40, '封印', '封印', 'approved', { category: 'magic' }),
    entry(41, '魔法', '魔法', 'rejected'),
    entry(43, '王都', '王都', 'proposed'),
    entry(44, '結界', '结界', 'pending_human'),
  ],
  profiles: [
    profile(38, 'ユキ', '小雪', 'approved', { aliases: ['ゆき'], notes: 'soft-spoken' }),
    profile(39, '雪乃', '雪乃', 'rejected', { aliases: ['シノ'] }),
  ],
})

// ---- The summary line -------------------------------------------------------------------------------------------------------

describe('the summary line', () => {
  it('gives the counts: waiting for a person, undecided, entries and profiles', () => {
    const many = kbOf({
      human_review_queue: [1, 2, 3].map((i) => waiting(termProposal(`c${i}`, '封印', '封缄'), 'conflict')),
      entries: Array.from({ length: 12 }, (_, i) => entry(i + 1, `語${i}`, `词${i}`)),
      profiles: Array.from({ length: 4 }, (_, i) => profile(i + 1, `名${i}`, `名${i}`)),
    })
    expect(summaryLine(many)).toBe('Knowledge Base · 3 waiting for a person · 0 undecided · 12 entries · 4 profiles')
  })

  it('counts the undecided from the Proposal Queue and says one entry and one profile in the singular', () => {
    const one = kbOf({ proposal_queue: [termProposal('u1', '魔法', '魔法')], entries: [entry(1, '魔法', '魔法')], profiles: [profile(2, 'ユキ', '小雪')] })
    expect(summaryLine(one)).toBe('Knowledge Base · 0 waiting for a person · 1 undecided · 1 entry · 1 profile')
  })

  it('counts every row it lists, a rejected one too', () => {
    expect(summaryLine(kb)).toBe('Knowledge Base · 0 waiting for a person · 0 undecided · 4 entries · 2 profiles')
  })

  it('says that the book has no Editor KB, and says nothing else about its content', () => {
    const line = summaryLine(kbOf({ kb_id: null }))
    expect(line).toBe('Knowledge Base · no Editor KB yet')
    expect(line).not.toMatch(/\d/)
  })

  it('says that it is loading, and that loading failed', () => {
    expect(summaryLine(null)).toBe('Knowledge Base · loading…')
    expect(summaryLine(null, true)).toBe('Knowledge Base · could not be loaded')
  })

  it('keeps what it had when a refresh failed, and says so', () => {
    expect(summaryLine(kb, true)).toBe(`${summaryLine(kb)} · could not refresh`)
    expect(summaryLine(kbOf({ kb_id: null }), true)).toBe('Knowledge Base · no Editor KB yet · could not refresh')
  })

  it('tells the body of a book with no Editor KB that one forms when a page is translated with full', () => {
    expect(NO_EDITOR_KB).toMatch(/no Editor KB yet/)
    expect(NO_EDITOR_KB).toMatch(/translated with full/)
  })

  it('says in one line that the Proposal Queue usually is empty', () => {
    expect(EMPTY_PROPOSAL_QUEUE).toMatch(/usually is empty/)
    expect(EMPTY_PROPOSAL_QUEUE).not.toContain('\n')
  })
})

// ---- The words for a payload --------------------------------------------------------------------------------------------------

describe('the words for a payload', () => {
  it('words a term as source → target with its category, the source in Japanese and the rendering in Simplified Chinese', () => {
    const words = describePayload('term', { source_term: '封印', target_term: '封印', category: 'magic' })
    expect(words.label).toBe('Term')
    expect(plain(words.pair)).toBe('封印 → 封印')
    expect(words.pair).toEqual([{ text: '封印', lang: 'ja' }, ' → ', { text: '封印', lang: 'zh-Hans' }])
    expect(words.extras.map(plain)).toEqual(['category: magic'])
  })

  it('leaves out a term\'s empty category', () => {
    expect(describePayload('term', { source_term: '魔法', target_term: '魔法', category: '' }).extras).toEqual([])
  })

  it('words a character as name → rendering with its other names (Japanese) and its notes', () => {
    const words = describePayload('character', { name_src: 'ユキ', name_tgt: '小雪', aliases: ['ゆき', '雪'], notes: 'soft-spoken' })
    expect(words.label).toBe('Character')
    expect(plain(words.pair)).toBe('ユキ → 小雪')
    expect(words.extras.map(plain)).toEqual(['also called ゆき, 雪', 'notes: soft-spoken'])
    expect(words.extras[0]).toContainEqual({ text: 'ゆき', lang: 'ja' })
    expect(words.extras[0]).toContainEqual({ text: '雪', lang: 'ja' })
  })

  it('leaves out a character\'s empty aliases and notes', () => {
    expect(describePayload('character', { name_src: 'ユキ', name_tgt: '小雪', aliases: [], notes: '' }).extras).toEqual([])
  })

  it('shows a kind it does not know, with its payload as it came', () => {
    const words = describePayload('place', { where: '王都' })
    expect(words.label).toBe('place')
    expect(plain(words.pair)).toBe('{"where":"王都"}')
  })

  it('does not fail on a payload that lacks a field or has the wrong type', () => {
    expect(plain(describePayload('term', {}).pair)).toBe(' → ')
    expect(describePayload('character', { name_src: 'ユキ', name_tgt: '小雪', aliases: 'ゆき', notes: 7 }).extras).toEqual([])
    expect(plain(describePayload('character', { name_src: 'ユキ', name_tgt: '小雪', aliases: ['ゆき', 3, null] }).extras[0])).toBe('also called ゆき')
  })

  it('words an entry and a profile as the proposal that made them', () => {
    expect(plain(describeEntry(entry(40, '封印', '封印', 'approved', { category: 'magic' })).pair)).toBe('封印 → 封印')
    expect(describeEntry(entry(40, '封印', '封印', 'approved', { category: 'magic' })).extras.map(plain)).toEqual(['category: magic'])
    expect(describeProfile(kb.profiles[0]).extras.map(plain)).toEqual(['also called ゆき', 'notes: soft-spoken'])
  })

  it('names a proposal by its pair and its page', () => {
    const p = termProposal('u1', '魔法', '魔法', { page_index: 18 })
    expect(plain(proposalWords(p))).toBe('魔法 → 魔法')
    expect(proposalName(p)).toBe('魔法 → 魔法, proposed on page 18')
  })

  it('names a proposal in a sentence by its pair and its page, the pair keeping its languages', () => {
    const p = termProposal('u1', '封印', '封缄', { page_index: 19 })
    expect(plain(proposalLabel(p))).toBe('封印 → 封缄 (proposed on page 19)')
    expect(proposalLabel(p).slice(0, 3)).toEqual(proposalWords(p))
    expect(proposalLabel(p)).toContainEqual({ text: '封印', lang: 'ja' })
    expect(proposalLabel(p)).toContainEqual({ text: '封缄', lang: 'zh-Hans' })
  })

  it('names a character the same way', () => {
    const p = nameProposal('n1', 'ユキ', '小雪', { page_index: 20 })
    expect(plain(proposalLabel(p))).toBe('ユキ → 小雪 (proposed on page 20)')
  })

  it('tells two proposals of one pair apart by their pages', () => {
    const [a, b] = [19, 17].map((page_index) => termProposal(`p${page_index}`, '封印', '封缄', { page_index }))
    expect(plain(proposalLabel(a))).not.toBe(plain(proposalLabel(b)))
  })

  it('gives a reason as it is, and words the lack of one', () => {
    expect(reasonText(termProposal('u1', '魔法', '魔法', { reason: 'conflicts with entry 40, which is 封印' }))).toBe('conflicts with entry 40, which is 封印')
    expect(reasonText(termProposal('u1', '魔法', '魔法'))).toBe('None yet.')
  })
})

// ---- What a proposal waits for ----------------------------------------------------------------------------------------------

describe('what a proposal waits for', () => {
  it('words the three states of the two queues apart', () => {
    const undecided = describeWaiting(termProposal('u1', '魔法', '魔法'))
    const newName = describeWaiting(waiting(nameProposal('n1', 'ユキ', '小雪'), 'new_name', 38))
    const conflict = describeWaiting(waiting(termProposal('c1', '封印', '封缄'), 'conflict'))
    expect([undecided.label, newName.label, conflict.label]).toEqual(['Undecided', 'New name', 'Conflict'])
    expect(new Set([undecided.text, newName.text, conflict.text]).size).toBe(3)
  })

  it('names the profile a new name made', () => {
    expect(describeWaiting(waiting(nameProposal('n1', 'ユキ', '小雪'), 'new_name', 38)).text).toBe(
      'A profile was made for it (profile 38) and waits for a person to confirm the name.',
    )
    expect(describeWaiting(waiting(nameProposal('n1', 'ユキ', '小雪'), 'new_name')).text).toBe(
      'A profile was made for it and waits for a person to confirm the name.',
    )
  })

  it('shows what it does not know as it is', () => {
    expect(describeWaiting(waiting(termProposal('x1', '魔法', '魔法'), 'a_wait_from_the_future')).label).toBe('a_wait_from_the_future')
  })

  it('knows a conflict from a new name, an undecided proposal and a wait it does not know', () => {
    expect(isConflict(waiting(termProposal('c1', '封印', '封缄'), 'conflict'))).toBe(true)
    expect(isConflict(waiting(nameProposal('c2', 'ユキ', '雪'), 'conflict'))).toBe(true)
    expect(isConflict(waiting(nameProposal('n1', 'ユキ', '小雪'), 'new_name', 38))).toBe(false)
    expect(isConflict(termProposal('u1', '魔法', '魔法'))).toBe(false)
    expect(isConflict(waiting(termProposal('x1', '封印', '封缄'), 'a_wait_from_the_future'))).toBe(false)
  })
})

// ---- The decisions offered: each row of Scope's table ----------------------------------------------------------------------

describe('the decisions offered', () => {
  it('offers an undecided term Approve and Reject', () => {
    expect(offerForProposal(termProposal('u1', '魔法', '魔法'), kb)).toEqual({ decisions: ['approve', 'reject'], note: null })
  })

  it('offers a new name Approve and Reject', () => {
    expect(offerForProposal(waiting(nameProposal('n1', 'ユキ', '小雪'), 'new_name', 38), kb)).toEqual({
      decisions: ['approve', 'reject'],
      note: null,
    })
  })

  it('offers a conflict Reject only, with one line that says why, and that the rendering can be taken when it can', () => {
    const taken = offerForProposal(waiting(termProposal('c1', '封印', '封缄'), 'conflict'), kb) // entry 40 holds 封印
    expect(taken).toEqual({ decisions: ['reject'], note: TAKE_NOTE })
    expect(TAKE_NOTE).toMatch(/Take this rendering to rename that entry to it, or reject the proposal/)
    const kept = offerForProposal(waiting(termProposal('c2', 'ユキ', '雪'), 'conflict'), kb) // a profile's name, not an entry's source
    expect(kept).toEqual({ decisions: ['reject'], note: CONFLICT_NOTE })
    expect(CONFLICT_NOTE).toMatch(/Reject the proposal to close it/)
  })

  it('no longer says that there is no rename to take: the two notes name what to do, and differ', () => {
    for (const note of [TAKE_NOTE, CONFLICT_NOTE]) {
      expect(note).not.toMatch(/not\s+built|\byet\b/)
      expect(note).toMatch(/^Approving is not offered: the KB keeps /)
    }
    expect(TAKE_NOTE).not.toBe(CONFLICT_NOTE)
  })

  it('offers every conflict Reject only, whatever kind of proposal it is', () => {
    for (const p of [termProposal('c1', '封印', '封缄'), nameProposal('c2', 'ユキ', '雪')]) {
      expect(offerForProposal(waiting(p, 'conflict'), kb).decisions, p.kind).toEqual(['reject'])
    }
  })

  it('gives a character conflict, and a term whose source is a character\'s name or alias, the note that only rejects', () => {
    for (const p of [nameProposal('c2', 'ユキ', '雪'), termProposal('c3', 'ユキ', '雪'), termProposal('c4', 'ゆき', '雪')]) {
      expect(offerForProposal(waiting(p, 'conflict'), kb).note, plain(proposalWords(p))).toBe(CONFLICT_NOTE)
    }
  })

  it('offers a name that waits undecided Reject only, as the server never approves one', () => {
    const offer = offerForProposal(nameProposal('n1', 'ユキ', '小雪'), kb)
    expect(offer).toEqual({ decisions: ['reject'], note: UNDECIDED_NAME_NOTE })
  })

  it('offers nothing for a wait it does not know', () => {
    expect(offerForProposal(waiting(termProposal('x1', '魔法', '魔法'), 'a_wait_from_the_future'), kb)).toEqual({ decisions: [], note: null })
  })

  it('offers an approved entry or profile Reject only', () => {
    expect(offerForRow('approved')).toEqual({ decisions: ['reject'], note: null })
  })

  it('offers a rejected entry or profile Approve only, so that it can be approved again', () => {
    expect(offerForRow('rejected')).toEqual({ decisions: ['approve'], note: null })
  })

  it('offers an entry or profile that is neither approved nor rejected both', () => {
    for (const status of ['pending_human', 'proposed', 'a_status_from_the_future']) {
      expect(offerForRow(status).decisions, status).toEqual(['approve', 'reject'])
    }
  })

  it('never offers a decision that changes nothing: Approve on an approved row, Reject on a rejected one', () => {
    expect(offerForRow('approved').decisions).not.toContain('approve')
    expect(offerForRow('rejected').decisions).not.toContain('reject')
  })
})

describe('the names of the decision controls', () => {
  const p = termProposal('u1', '魔法', '魔法', { page_index: 18 })

  it('name the proposal, with its pair and its page', () => {
    expect(proposalControlName('approve', p)).toBe('Approve 魔法 → 魔法, proposed on page 18')
    expect(proposalControlName('reject', p)).toBe('Reject 魔法 → 魔法, proposed on page 18')
  })

  it('tell two proposals of one pair apart by their pages', () => {
    const later = termProposal('u2', '魔法', '魔法', { page_index: 21 })
    expect(proposalControlName('reject', later)).not.toBe(proposalControlName('reject', p))
  })

  it('name the entry or the profile by its kind and id, and its pair', () => {
    expect(entryControlName('reject', kb.entries[0])).toBe('Reject entry 40: 封印 → 封印')
    expect(entryControlName('approve', kb.entries[1])).toBe('Approve entry 41: 魔法 → 魔法')
    expect(profileControlName('approve', kb.profiles[1])).toBe('Approve profile 39: 雪乃 → 雪乃')
  })

  it('differ between the two decisions of one item', () => {
    const names = (['approve', 'reject'] as KBDecision[]).map((d) => entryControlName(d, kb.entries[0]))
    expect(new Set(names).size).toBe(2)
  })

  it('make a key for each kind of item that no other kind shares', () => {
    expect(itemKey({ kind: 'proposal', id: '12bd' })).toBe('proposal:12bd')
    expect(itemKey({ kind: 'entry', id: 40 })).toBe('entry:40')
    expect(itemKey({ kind: 'profile', id: 40 })).toBe('profile:40')
  })
})

// ---- The rows a conflict collides with: both kinds of conflict for each kind of proposal ---------------------------------------

describe('the rows a conflict collides with', () => {
  it('finds, for a term, the readable entry with the same source: the example Scope words', () => {
    const rivals = rivalsOf(waiting(termProposal('c1', '封印', '封缄'), 'conflict'), kb)
    expect(rivals).toEqual([{ kind: 'entry', id: 40, status: 'approved', source: '封印', target: '封印', alias: null }])
    expect(plain(describeRival(rivals[0]))).toBe('The KB holds 封印 → 封印 (entry 40, approved)')
  })

  it('finds, for a term across kinds, the readable profile whose name is that source', () => {
    const rivals = rivalsOf(waiting(termProposal('c2', 'ユキ', '雪'), 'conflict'), kb)
    expect(rivals).toEqual([{ kind: 'profile', id: 38, status: 'approved', source: 'ユキ', target: '小雪', alias: null }])
    expect(plain(describeRival(rivals[0]))).toBe('The KB holds ユキ → 小雪 (profile 38, approved)')
  })

  it('finds, for a term across kinds, the readable profile whose alias is that source, and says it is an alias', () => {
    const rivals = rivalsOf(waiting(termProposal('c3', 'ゆき', '雪'), 'conflict'), kb)
    expect(rivals).toEqual([{ kind: 'profile', id: 38, status: 'approved', source: 'ユキ', target: '小雪', alias: 'ゆき' }])
    expect(plain(describeRival(rivals[0]))).toBe('The KB holds ユキ → 小雪 (profile 38, approved), which is also called ゆき')
  })

  it('finds, for a character, the readable profile with the same name', () => {
    const rivals = rivalsOf(waiting(nameProposal('c4', 'ユキ', '雪'), 'conflict'), kb)
    expect(rivals.map((r) => [r.kind, r.id])).toEqual([['profile', 38]])
    expect(plain(describeRival(rivals[0]))).toBe('The KB holds ユキ → 小雪 (profile 38, approved)')
  })

  it('finds, for a character across kinds, the readable entry whose source is that name', () => {
    const rivals = rivalsOf(waiting(nameProposal('c5', '封印', '封印'), 'conflict'), kb)
    expect(rivals.map((r) => [r.kind, r.id])).toEqual([['entry', 40]])
    expect(plain(describeRival(rivals[0]))).toBe('The KB holds 封印 → 封印 (entry 40, approved)')
  })

  it('looks only for the same name: an alias, or a name that contains one, finds nothing (rule 6 matches those, on the server)', () => {
    expect(rivalsOf(waiting(nameProposal('c6', 'ゆき', '雪'), 'conflict'), kb)).toEqual([]) // an alias of profile 38
    expect(rivalsOf(waiting(nameProposal('c6b', 'ユキちゃん', '雪酱'), 'conflict'), kb)).toEqual([]) // contains the name of profile 38
  })

  it('finds a row that waits for a person: it is readable too, and the line says so', () => {
    const rivals = rivalsOf(waiting(termProposal('c7', '結界', '屏障'), 'conflict'), kb)
    expect(rivals.map((r) => [r.kind, r.id, r.status])).toEqual([['entry', 44, 'pending_human']])
    expect(plain(describeRival(rivals[0]))).toBe('The KB holds 結界 → 结界 (entry 44, waiting for a person)')
  })

  it('puts the row of the same kind first, then the one across kinds, whichever kind the proposal is', () => {
    const both = kbOf({ entries: [entry(50, 'ユキ', '雪')], profiles: [profile(38, 'ユキ', '小雪')] })
    expect(rivalsOf(waiting(termProposal('c8', 'ユキ', '雪人'), 'conflict'), both).map((r) => r.kind)).toEqual(['entry', 'profile'])
    expect(rivalsOf(waiting(nameProposal('c9', 'ユキ', '雪人'), 'conflict'), both).map((r) => r.kind)).toEqual(['profile', 'entry'])
  })

  it('does not find a rejected row: a rejected entry, a rejected profile or its alias', () => {
    expect(rivalsOf(waiting(termProposal('r1', '魔法', '魔術'), 'conflict'), kb)).toEqual([]) // entry 41 is rejected
    expect(rivalsOf(waiting(termProposal('r2', '雪乃', '雪野'), 'conflict'), kb)).toEqual([]) // profile 39 is rejected
    expect(rivalsOf(waiting(termProposal('r3', 'シノ', '诗乃'), 'conflict'), kb)).toEqual([]) // an alias of profile 39
    expect(rivalsOf(waiting(nameProposal('r4', '雪乃', '雪野'), 'conflict'), kb)).toEqual([])
    expect(rivalsOf(waiting(nameProposal('r5', '魔法', '魔術'), 'conflict'), kb)).toEqual([])
  })

  it('does not find an undecided row either: the agents cannot find it, so the rules do not see it', () => {
    expect(rivalsOf(waiting(termProposal('r6', '王都', '王城'), 'conflict'), kb)).toEqual([]) // entry 43 is `proposed`
  })

  it('finds nothing when no row has the source or name: no row at all, and says only that, not why', () => {
    expect(rivalsOf(waiting(termProposal('n1', '呪文', '咒语'), 'conflict'), kb)).toEqual([])
    expect(rivalsOf(waiting(termProposal('n1', '封印', '封缄'), 'conflict'), kbOf())).toEqual([])
    expect(NO_RIVAL).toMatch(/^No readable entry or profile with this source or name was found\./)
    expect(NO_RIVAL).toMatch(/reason says what it collided with/)
    expect(NO_RIVAL).not.toMatch(/chang/) // rule 6 can make a conflict that no row of the same name explains: nothing changed
  })

  it('has rivals for a conflict only: not for a new name, whose own profile is readable, nor for an undecided term', () => {
    expect(rivalsOf(waiting(nameProposal('n2', 'ユキ', '小雪'), 'new_name', 38), kb)).toEqual([])
    expect(rivalsOf(termProposal('u1', '封印', '封缄'), kb)).toEqual([])
    expect(rivalsOf(waiting(termProposal('x1', '封印', '封缄'), 'a_wait_from_the_future'), kb)).toEqual([])
  })

  it('has none for a kind it does not know', () => {
    const unknown: ReviewProposal = { ...waiting(termProposal('k1', '封印', '封缄'), 'conflict'), kind: 'place', payload: { source_term: '封印' } }
    expect(rivalsOf(unknown, kb)).toEqual([])
  })

  it('marks the source in Japanese and the rendering in Simplified Chinese', () => {
    const [rival] = rivalsOf(waiting(termProposal('c1', '封印', '封缄'), 'conflict'), kb)
    const words = describeRival(rival)
    expect(words).toContainEqual({ text: '封印', lang: 'ja' })
    expect(words).toContainEqual({ text: '封印', lang: 'zh-Hans' })
    const [alias] = rivalsOf(waiting(termProposal('c3', 'ゆき', '雪'), 'conflict'), kb)
    expect(describeRival(alias)).toContainEqual({ text: 'ゆき', lang: 'ja' })
  })
})

// ---- The outcome of a decision ----------------------------------------------------------------------------------------------

describe('the outcome of a decision', () => {
  const proposal = termProposal('u1', '魔法', '魔法', { page_index: 18 })
  const label = proposalLabel(proposal)
  const say = (outcome: string) => plain(describeOutcome(outcome, label))
  const APPROVED = ['entry_created', 'entry_confirmed', 'profile_approved']
  const NOT_APPROVED = ['held_as_conflict', 'held_as_conflict_across_kinds', 'proposal_rejected', 'profile_rejected']
  const SEVEN = [...APPROVED, ...NOT_APPROVED]

  it('words an undecided term that was approved as a new entry', () => {
    expect(say('entry_created')).toBe('魔法 → 魔法 (proposed on page 18) was approved by a person and added as a new entry.')
  })

  it('words one that confirmed the entry the KB already holds', () => {
    expect(say('entry_confirmed')).toBe(
      '魔法 → 魔法 (proposed on page 18) was approved by a person. It confirms an entry the KB already holds.',
    )
  })

  it('words a new name that was approved', () => {
    expect(say('profile_approved')).toBe(
      '魔法 → 魔法 (proposed on page 18) was approved by a person. The new name is now an approved profile.',
    )
  })

  it('names the page the proposal was made on, in each of the seven outcomes and in one it does not know', () => {
    for (const outcome of [...SEVEN, 'a_future_outcome']) expect(say(outcome), outcome).toContain('proposed on page 18')
  })

  it('reads differently for two proposals of one pair made on different pages, in each outcome', () => {
    const later = proposalLabel(termProposal('c1', '封印', '封缄', { page_index: 19 }))
    const earlier = proposalLabel(termProposal('u2', '封印', '封缄', { page_index: 17 }))
    for (const outcome of [...SEVEN, 'a_future_outcome']) {
      const [a, b] = [later, earlier].map((words) => plain(describeOutcome(outcome, words)))
      expect(a, outcome).toContain('page 19')
      expect(b, outcome).toContain('page 17')
      expect(a, outcome).not.toBe(b)
    }
  })

  it('words only entry_created, entry_confirmed and profile_approved as approved', () => {
    for (const outcome of APPROVED) expect(say(outcome), outcome).toMatch(/was approved by a person/)
    for (const outcome of NOT_APPROVED) expect(say(outcome), outcome).not.toMatch(/approv/i)
  })

  it('says of a held proposal that it was not added and waits for a person again, for both kinds of conflict', () => {
    for (const outcome of ['held_as_conflict', 'held_as_conflict_across_kinds']) {
      expect(say(outcome), outcome).toMatch(/was not added/)
      expect(say(outcome), outcome).toMatch(/waits for a person again/)
    }
  })

  it('tells the two held outcomes apart: another rendering, or a name', () => {
    expect(say('held_as_conflict')).toMatch(/another rendering/)
    expect(say('held_as_conflict_across_kinds')).toMatch(/name or an alias of a character/)
  })

  it('words the two rejections', () => {
    expect(say('proposal_rejected')).toBe('魔法 → 魔法 (proposed on page 18) was rejected by a person.')
    expect(say('profile_rejected')).toBe(
      '魔法 → 魔法 (proposed on page 18) was rejected by a person. The new name is a rejected profile.',
    )
  })

  it('gives each of the seven outcomes its own words', () => {
    expect(new Set(SEVEN.map(say)).size).toBe(7)
  })

  it('shows an outcome it does not know as it is, and does not word it as approved', () => {
    const words = say('a_future_outcome')
    expect(words).toContain('a_future_outcome')
    expect(words).not.toMatch(/approv/i)
    // `entry_approved` is the entries route's outcome, which the proposals route never sends: it is shown as it is, not worded
    expect(say('entry_approved')).toBe('魔法 → 魔法 (proposed on page 18): the server answered "entry_approved".')
    expect(say('entry_approved')).not.toMatch(/was approved/)
  })

  it('keeps the proposal in the words as it came, with its languages marked', () => {
    for (const outcome of [...SEVEN, 'a_future_outcome']) {
      expect(describeOutcome(outcome, label).slice(0, label.length), outcome).toEqual(label)
      expect(describeOutcome(outcome, label), outcome).toContainEqual({ text: '魔法', lang: 'ja' })
      expect(describeOutcome(outcome, label), outcome).toContainEqual({ text: '魔法', lang: 'zh-Hans' })
    }
  })

  it('words an entry or a profile that was decided, which the routes answer with no outcome', () => {
    expect(plain(describeRowDecision('entry', 40, proposalWords(termProposal('u', '封印', '封印')), 'reject'))).toBe(
      'Entry 40 (封印 → 封印) was rejected by a person.',
    )
    expect(plain(describeRowDecision('profile', 38, proposalWords(nameProposal('n', 'ユキ', '小雪')), 'approve'))).toBe(
      'Profile 38 (ユキ → 小雪) was approved by a person.',
    )
  })
})

// ---- A row's history ----------------------------------------------------------------------------------------------------------

describe('the history of a row', () => {
  const say = (event: string, extra: Record<string, unknown> = {}) => plain(describeHistoryLine(line(event, extra)))

  it('words an agent\'s line by what happened and on which page', () => {
    expect(say('created', { page: 18, run_id: 448 })).toBe('Created on page 18')
    expect(say('confirmed', { page: 19, run_id: 449 })).toBe('Confirmed on page 19')
  })

  it('words a merge with the aliases it added, in Japanese, or says that it added none', () => {
    expect(say('merged', { page: 20, aliases: ['ゆき', '雪'] })).toBe('Merged on page 20: new aliases ゆき, 雪')
    expect(describeHistoryLine(line('merged', { page: 20, aliases: ['ゆき'] }))).toContainEqual({ text: 'ゆき', lang: 'ja' })
    expect(say('merged', { page: 20, aliases: [] })).toBe('Merged on page 20: no new alias')
  })

  it('words a person\'s decision by the person and the status it replaced', () => {
    expect(say('rejected', { by: 'human', from: 'approved' })).toBe('Rejected by a person (was approved)')
    expect(say('approved', { by: 'human', from: 'rejected' })).toBe('Approved by a person (was rejected)')
    expect(say('approved', { by: 'human', from: 'pending_human' })).toBe('Approved by a person (was waiting for a person)')
  })

  it('says that a decision came through a proposal', () => {
    expect(say('approved', { by: 'human', from: 'pending_human', proposal_id: '9b2de254' })).toBe(
      'Approved by a person, through its proposal (was waiting for a person)',
    )
  })

  it('words an entry a person made by approving a proposal, and the page the proposal came from', () => {
    expect(say('created', { by: 'human', page: 18, run_id: 448, proposal_id: '12bd', reason: 'approved by a person' })).toBe(
      'Created by a person, from the proposal made on page 18',
    )
    expect(say('confirmed', { by: 'human', page: 18 })).toBe('Confirmed by a person, from the proposal made on page 18')
  })

  it('words a rename by the person and the two renderings, in Simplified Chinese, and the status it replaced', () => {
    const renamed = { by: 'human', from: 'approved', before: '托尔蒂亚家族', after: '托尔蒂亚一家' }
    expect(say('renamed', renamed)).toBe('Renamed by a person: 托尔蒂亚家族 → 托尔蒂亚一家 (was approved)')
    expect(say('renamed', { ...renamed, from: 'pending_human' })).toBe(
      'Renamed by a person: 托尔蒂亚家族 → 托尔蒂亚一家 (was waiting for a person)',
    )
    const words = describeHistoryLine(line('renamed', renamed))
    expect(words).toContainEqual({ text: '托尔蒂亚家族', lang: 'zh-Hans' })
    expect(words).toContainEqual({ text: '托尔蒂亚一家', lang: 'zh-Hans' })
    expect(words.filter((piece) => typeof piece !== 'string' && piece.lang === 'ja')).toEqual([])
  })

  it('words the line a rename leaves on the conflict it settled: a confirmation by a person', () => {
    expect(say('confirmed', { by: 'human', page: 7, run_id: 437, proposal_id: 'c6f34bae' })).toBe(
      'Confirmed by a person, from the proposal made on page 7',
    )
  })

  it('leaves out the renderings of a rename line that lacks them or holds something else', () => {
    expect(say('renamed', { by: 'human', from: 'approved' })).toBe('Renamed by a person (was approved)')
    expect(say('renamed', { by: 'human', before: '托尔蒂亚家族' })).toBe('Renamed by a person')
    expect(say('renamed', { by: 'human', before: 3, after: null })).toBe('Renamed by a person')
    expect(say('renamed')).toBe('Renamed')
  })

  it('shows an event it does not know as it is', () => {
    expect(say('frobnicated', { by: 'human', from: 'approved' })).toBe('Event "frobnicated" by a person (was approved)')
    expect(say('frobnicated')).toBe('Event "frobnicated"')
  })

  it('shows a status it does not know as it is', () => {
    expect(say('rejected', { by: 'human', from: 'a_status_from_the_future' })).toBe('Rejected by a person (was a_status_from_the_future)')
  })

  it('does not take a page or aliases of the wrong type for one', () => {
    expect(say('created', { page: '18' })).toBe('Created')
    expect(say('merged', { page: 20, aliases: 'ゆき' })).toBe('Merged on page 20: no new alias')
  })

  it('gives the time in UTC, and anything it cannot read as it is', () => {
    expect(formatAt('2026-11-14T21:33:27+00:00')).toBe('2026-11-14 21:33:27 UTC')
    expect(formatAt('2026-11-14T21:33:27Z')).toBe('2026-11-14 21:33:27 UTC')
    expect(formatAt('2026-11-14T21:33:27.123456+00:00')).toBe('2026-11-14 21:33:27 UTC')
    expect(formatAt('yesterday')).toBe('yesterday')
    expect(formatAt('2026-11-14T21:33:27+09:00')).toBe('2026-11-14T21:33:27+09:00') // not UTC: not relabelled
  })
})

// ---- States ----------------------------------------------------------------------------------------------------------------------

describe('states and provenance', () => {
  it('says which states are readable: the two the agents can find', () => {
    expect(['approved', 'pending_human', 'rejected', 'proposed'].filter(isReadable)).toEqual(['approved', 'pending_human'])
  })

  it('words each state, and shows one it does not know as it is, even one that is a property of every object', () => {
    expect(['approved', 'pending_human', 'rejected', 'proposed'].map(statusLabel)).toEqual([
      'approved',
      'waiting for a person',
      'rejected',
      'undecided',
    ])
    expect(statusLabel('a_status_from_the_future')).toBe('a_status_from_the_future')
    expect(statusLabel('constructor')).toBe('constructor')
  })

  it('words a person\'s provenance as person, and an agent\'s as agent', () => {
    expect(provenanceLabel('human')).toBe('person')
    expect(provenanceLabel('agent')).toBe('agent')
    expect(provenanceLabel('a_provenance_from_the_future')).toBe('a_provenance_from_the_future')
  })
})

// ---- Rename: where it is offered ---------------------------------------------------------------------------------------------

describe('the rename offered', () => {
  it('is offered on a readable entry or profile: an approved one, and one that waits for a person', () => {
    expect(canRename('approved')).toBe(true)
    expect(canRename('pending_human')).toBe(true)
  })

  it('is not offered on a rejected or an undecided row, which the server refuses, nor on a status it does not know', () => {
    for (const status of ['rejected', 'proposed', 'a_status_from_the_future', '', 'constructor']) {
      expect(canRename(status), status).toBe(false)
    }
  })

  it('goes with the states the rules read, and no other', () => {
    for (const status of ['approved', 'pending_human', 'rejected', 'proposed']) expect(canRename(status), status).toBe(isReadable(status))
  })

  it('names the row it renames, with its pair now, and starts with what the button says', () => {
    expect(renameControlName('entry', 40, describeEntry(kb.entries[0]).pair)).toBe('Rename entry 40: 封印 → 封印')
    expect(renameControlName('profile', 38, describeProfile(kb.profiles[0]).pair)).toBe('Rename profile 38: ユキ → 小雪')
    expect(renameControlName('entry', 40, describeEntry(kb.entries[0]).pair).startsWith(RENAME_LABEL)).toBe(true)
  })

  it('names two rows of one pair apart by their ids', () => {
    const words = describeEntry(entry(40, '封印', '封印')).pair
    expect(renameControlName('entry', 40, words)).not.toBe(renameControlName('entry', 41, words))
    expect(renameControlName('entry', 40, words)).not.toBe(renameControlName('profile', 40, words))
  })
})

// ---- "Take this rendering": zero, one and two entry rivals, and a profile rival ------------------------------------------------

describe('"Take this rendering"', () => {
  const conflict = (source: string, target: string, over: Partial<KBProposal> = {}) =>
    waiting(termProposal('c1', source, target, over), 'conflict')

  it('is offered when exactly one readable entry has the source: that entry, renamed to the proposal\'s rendering', () => {
    const offer = takeOffer(conflict('封印', '封缄'), kb)
    expect(offer).toEqual({ entry: kb.entries[0], target: '封缄' })
    expect(offer?.entry.id).toBe(40)
  })

  it('is not offered when no entry has the source', () => {
    expect(takeOffer(conflict('呪文', '咒语'), kb)).toBeNull()
    expect(takeOffer(conflict('封印', '封缄'), kbOf())).toBeNull()
  })

  it('is not offered when two readable entries have it: there is no one entry to rename', () => {
    const two = kbOf({ entries: [entry(50, '封印', '封印'), entry(51, '封印', '封缄印', 'pending_human')] })
    expect(rivalsOf(conflict('封印', '封缄'), two).filter((r) => r.kind === 'entry')).toHaveLength(2)
    expect(takeOffer(conflict('封印', '封缄'), two)).toBeNull()
  })

  it('counts only readable entries: a rejected or an undecided entry with the source is not a rival', () => {
    expect(takeOffer(conflict('魔法', '魔術'), kb)).toBeNull() // entry 41 is rejected
    expect(takeOffer(conflict('王都', '王城'), kb)).toBeNull() // entry 43 is undecided
    const one = kbOf({ entries: [entry(50, '封印', '封印', 'rejected'), entry(51, '封印', '封印', 'approved')] })
    expect(takeOffer(conflict('封印', '封缄'), one)?.entry.id).toBe(51)
  })

  it('is offered for an entry that waits for a person: its rename approves it', () => {
    expect(takeOffer(conflict('結界', '屏障'), kb)?.entry).toEqual(kb.entries[3])
  })

  it('is not offered when a profile is the only rival: a term whose source is a character\'s name or an alias', () => {
    expect(takeOffer(conflict('ユキ', '雪'), kb)).toBeNull() // the name of profile 38
    expect(takeOffer(conflict('ゆき', '雪'), kb)).toBeNull() // its alias
  })

  it('is offered for the entry when a profile has the same source too: the entry is the one to rename', () => {
    const both = kbOf({ entries: [entry(50, 'ユキ', '雪')], profiles: [profile(38, 'ユキ', '小雪')] })
    expect(takeOffer(conflict('ユキ', '雪人'), both)?.entry.id).toBe(50)
  })

  it('is not offered for a character conflict, a new name, an undecided term or a wait it does not know', () => {
    expect(takeOffer(waiting(nameProposal('c2', '封印', '封缄'), 'conflict'), kb)).toBeNull() // an entry's source as a character
    expect(takeOffer(waiting(nameProposal('n1', 'ユキ', '小雪'), 'new_name', 38), kb)).toBeNull()
    expect(takeOffer(termProposal('u1', '封印', '封缄'), kb)).toBeNull()
    expect(takeOffer(waiting(termProposal('x1', '封印', '封缄'), 'a_wait_from_the_future'), kb)).toBeNull()
  })

  it('is not offered for a kind it does not know, or a proposal that lacks its rendering', () => {
    const unknown: ReviewProposal = { ...conflict('封印', '封缄'), kind: 'place', payload: { source_term: '封印', target_term: '封缄' } }
    expect(takeOffer(unknown, kb)).toBeNull()
    const lacking: ReviewProposal = { ...conflict('封印', '封缄'), payload: { source_term: '封印' } }
    expect(takeOffer(lacking, kb)).toBeNull()
    expect(takeOffer(conflict('封印', ''), kb)).toBeNull()
  })

  it('is not offered when the entry has the proposal\'s rendering already: nothing is left to take, and the server refuses it', () => {
    expect(takeOffer(conflict('封印', '封印'), kb)).toBeNull()
  })

  it('goes with the note: the one that says the rendering can be taken exactly when it is offered', () => {
    for (const p of [conflict('封印', '封缄'), conflict('ユキ', '雪'), conflict('呪文', '咒语')]) {
      const note = offerForProposal(p, kb).note
      expect(note === TAKE_NOTE, plain(proposalWords(p))).toBe(takeOffer(p, kb) !== null)
    }
  })

  it('names what it renames and what to, with the page it was proposed on, and starts with what the button says', () => {
    const p = conflict('封印', '封缄', { page_index: 19 })
    const take = takeOffer(p, kb)
    expect(take).not.toBeNull()
    expect(takeControlName(p, take!)).toBe('Take this rendering: rename entry 40 (封印 → 封印) to 封缄, as proposed on page 19')
    expect(takeControlName(p, take!).startsWith(TAKE_LABEL)).toBe(true)
  })
})

// ---- Where a failed request is shown -------------------------------------------------------------------------------------------

describe('where a failed request is shown', () => {
  it('shows the rules\' refusal beside the item: a 409, whatever it carries', () => {
    expect(isRefusal(409, true)).toBe(true)
    expect(isRefusal(409, false)).toBe(true)
  })

  it('shows a 422 beside the item when it carries the server\'s own sentence, a rendering the KB would not store', () => {
    expect(isRefusal(422, true)).toBe(true)
  })

  it('shows a 422 that is FastAPI\'s list in the panel: a body the route did not take is not text for a person', () => {
    expect(isRefusal(422, false)).toBe(false)
  })

  it('shows any other failure in the panel: a gone row, a backend that did not answer, a server error', () => {
    for (const status of [0, 400, 404, 500, 502, 503]) {
      expect(isRefusal(status, true), String(status)).toBe(false)
      expect(isRefusal(status, false), String(status)).toBe(false)
    }
  })
})
