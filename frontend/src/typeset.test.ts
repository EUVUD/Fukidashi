import { describe, expect, it } from 'vitest'
import {
  CLOSING_MARKS,
  LINE_HEIGHT,
  MAX_FONT_PX,
  MIN_FONT_PX,
  OPENING_MARKS,
  blockHeight,
  describeFigures,
  overflowRate,
  percent,
  pooled,
  summarizeBook,
  typeset,
  typesetPage,
  type Measure,
  type OverflowFigures,
  type Typeset,
} from './typeset'

// A fake measure: a CJK character or a full-width mark is 1 em wide and an ASCII character half an em, in whatever size.
const measure: Measure = (text, size) =>
  Array.from(text).reduce((width, ch) => width + (ch.codePointAt(0)! < 0x80 ? 0.5 : 1) * size, 0)

// The box is `w` wide; its height is, unless a test says otherwise, more than any layout here needs.
const lay = (text: string | null, w: number, h = 1000) => typeset(text, { w, h }, measure)

const HANZI6 = '一二三四五六'

describe('font size', () => {
  it('keeps the largest size when the text fits', () => {
    expect(lay('你好', 100)).toEqual({ fontSize: 24, lines: ['你好'], overflow: false })
  })

  it('shrinks to the largest size that fits', () => {
    // One line of 6 characters is 6 × size wide: 102 px at 17, 96 px at 16. Two lines would need 2.4 × size ≤ 30.
    expect(typeset(HANZI6, { w: 100, h: 30 }, measure)).toEqual({ fontSize: 16, lines: [HANZI6], overflow: false })
  })

  it.each([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14])('takes exactly %i px when the line is 6 × that wide', (size) => {
    // h = 29 lets one line through at every size (1.2 × 24 = 28.8) and never two (2 × 1.2 × 14 = 33.6).
    expect(typeset(HANZI6, { w: 6 * size, h: 29 }, measure)).toEqual({ fontSize: size, lines: [HANZI6], overflow: false })
  })

  it('has the constants of OUTLINE §8', () => {
    expect([MAX_FONT_PX, MIN_FONT_PX, LINE_HEIGHT]).toEqual([24, 14, 1.2])
  })

  it('accepts a line exactly as wide as the box, and lines exactly as high', () => {
    // 2 characters a line is 40 px at 20, and two lines are 2 × 20 × 1.2 = 48 px.
    expect(typeset('一二三四', { w: 40, h: 48 }, measure)).toEqual({ fontSize: 20, lines: ['一二', '三四'], overflow: false })
    // A hair less height and 20 px is out: 19 px gives two lines of 45.6 px.
    expect(typeset('一二三四', { w: 40, h: 47.9 }, measure)).toEqual({ fontSize: 19, lines: ['一二', '三四'], overflow: false })
  })

  it('breaks lines as the size goes down, so a smaller size can mean fewer lines', () => {
    // At 24 px a 60 px box holds 2 characters a line: 3 lines of 28.8 px do not fit in 70. At 20 px it holds 3: 2 lines do.
    expect(typeset('一二三四五六', { w: 60, h: 70 }, measure)).toEqual({ fontSize: 20, lines: ['一二三', '四五六'], overflow: false })
  })
})

describe('overflow', () => {
  it('lays the text out at the minimum size and reports an overflow when it does not fit there', () => {
    // 28 px holds 2 characters a line at 14 px: five lines of 16.8 px are 84 px, in a box 40 px high.
    expect(typeset('一二三四五六七八九十', { w: 30, h: 40 }, measure)).toEqual({
      fontSize: MIN_FONT_PX,
      lines: ['一二', '三四', '五六', '七八', '九十'],
      overflow: true,
    })
  })

  it('does not call it an overflow when the text fits at the minimum size exactly', () => {
    expect(typeset(HANZI6, { w: 84, h: 29 }, measure)).toMatchObject({ fontSize: 14, overflow: false })
    // One pixel narrower: a line holds 5 characters, and two lines are 33.6 px in 29.
    expect(typeset(HANZI6, { w: 83, h: 29 }, measure)).toEqual({
      fontSize: 14,
      lines: ['一二三四五', '六'],
      overflow: true,
    })
  })

  it('overflows when a box is too short for even one line, whatever the width', () => {
    expect(typeset('你', { w: 500, h: 10 }, measure)).toEqual({ fontSize: 14, lines: ['你'], overflow: true })
  })

  it('lays a character wider than the box out on a line of its own and calls it an overflow', () => {
    expect(typeset('一二', { w: 10, h: 500 }, measure)).toEqual({ fontSize: 14, lines: ['一', '二'], overflow: true })
  })
})

describe('line breaks', () => {
  it('breaks at \\n, even when the text would fit on one line', () => {
    expect(lay('一二\n三四', 200)).toEqual({ fontSize: 24, lines: ['一二', '三四'], overflow: false })
  })

  it('keeps an empty line for a second \\n, and reads \\r\\n as one break', () => {
    expect(lay('一\n\n二', 200)?.lines).toEqual(['一', '', '二'])
    expect(lay('一\r\n二', 200)?.lines).toEqual(['一', '二'])
  })

  it('breaks between any two characters, letters and Chinese alike', () => {
    expect(lay('第3页的故事', 72)).toEqual({ fontSize: 24, lines: ['第3页', '的故事'], overflow: false })
  })
})

describe('a line does not start with a closing mark', () => {
  const marks = [...new Set(Array.from(CLOSING_MARKS))]

  it('knows the marks of OUTLINE §8, full-width and half-width', () => {
    expect(marks).toHaveLength(27) // 18 full-width, 9 half-width
    for (const mark of Array.from('，。、．：；！？…）」』】〕〉》”’,.:;!?)]}')) expect(marks).toContain(mark)
  })

  it.each(marks)('takes the last character of the line with %s when it does not fit', (mark) => {
    // 一二 fills the 48 px box, so the mark has no room, and it cannot start a line: 二 goes down with it.
    expect(lay(`一二${mark}`, 48)).toEqual({ fontSize: 24, lines: ['一', `二${mark}`], overflow: false })
  })

  it('leaves the mark where it is when it fits', () => {
    expect(lay('一二。', 72)).toEqual({ fontSize: 24, lines: ['一二。'], overflow: false })
  })

  it('takes the last character with a chain of marks', () => {
    expect(lay('一二！？', 72)).toEqual({ fontSize: 24, lines: ['一', '二！？'], overflow: false })
    // …and a mark after a closing quote cannot start a line either, so the quote's character goes down too.
    expect(lay('一二”。', 72)).toEqual({ fontSize: 24, lines: ['一', '二”。'], overflow: false })
  })

  it('keeps the mark on the line, which then counts as too wide, when moving it would leave the line empty', () => {
    // 一 is 24 px and fits the 24 px box, but 一。 is 48: nothing smaller than 14 px helps, so it overflows.
    expect(lay('一。', 24)).toEqual({ fontSize: 14, lines: ['一。'], overflow: true })
    // The same mark shrinks the text instead when the box can hold both at a smaller size.
    expect(lay('一。', 40)).toEqual({ fontSize: 20, lines: ['一。'], overflow: false })
  })

  it('lets the text itself start with a mark: there is no break before it', () => {
    expect(lay('。一', 100)).toEqual({ fontSize: 24, lines: ['。一'], overflow: false })
  })
})

describe('a line does not end with an opening mark', () => {
  const marks = [...new Set(Array.from(OPENING_MARKS))]

  it('knows the marks of OUTLINE §8, full-width and half-width', () => {
    expect(marks).toHaveLength(12) // 9 full-width, 3 half-width
    for (const mark of Array.from('（「『【〔〈《“‘([{')) expect(marks).toContain(mark)
  })

  it.each(marks)('moves %s to the next line with the character after it', (mark) => {
    // 一 and the mark would fit the 48 px box, but 三 would not: the mark would end the line, so it goes down with 三.
    expect(lay(`一${mark}三`, 48)).toEqual({ fontSize: 24, lines: ['一', `${mark}三`], overflow: false })
  })

  it('moves a chain of opening marks together', () => {
    expect(lay('一「『三', 72)).toEqual({ fontSize: 24, lines: ['一', '「『三'], overflow: false })
  })

  it('leaves an opening mark that ends the text where it is', () => {
    expect(lay('一（', 100)).toEqual({ fontSize: 24, lines: ['一（'], overflow: false })
  })

  it('moves an opening mark with a closing mark that follows it, as the closing mark cannot start a line', () => {
    // 一「 would fit the 48 px box, and 」 would not: it cannot start a line, so 「 goes down with it.
    expect(lay('一「」', 48)).toEqual({ fontSize: 24, lines: ['一', '「」'], overflow: false })
  })
})

describe('what is never split', () => {
  it('keeps …… and —— whole', () => {
    // 一…… is 72 px: at 24 px it does not fit 60, and a split ('一…' and '…') would not be allowed to start a line either.
    expect(lay('一……', 60)).toEqual({ fontSize: 20, lines: ['一……'], overflow: false })
    // —— is not a closing mark, so only its own rule keeps it together: 一二— would fit the 72 px box.
    expect(lay('一二——', 72)).toEqual({ fontSize: 24, lines: ['一二', '——'], overflow: false })
  })

  it('does not let a lone … start a line', () => {
    expect(lay('一二…', 48)).toEqual({ fontSize: 24, lines: ['一', '二…'], overflow: false })
  })

  it('keeps a run of ASCII letters and digits on one line when it fits on a line', () => {
    // ABC is 36 px at 24 and the line has 12 px left after 一二: breaking by character would put A there.
    expect(lay('一二ABC', 60)).toEqual({ fontSize: 24, lines: ['一二', 'ABC'], overflow: false })
    expect(lay('一二A1b', 60)?.lines).toEqual(['一二', 'A1b'])
  })

  it('breaks a run that is longer than a line by character', () => {
    // 8 × 12 = 96 px does not fit 60: 5 characters fill a line.
    expect(lay('ABCDEFGH', 60)).toEqual({ fontSize: 24, lines: ['ABCDE', 'FGH'], overflow: false })
  })

  it('judges a run at each size: it can fit a line once the text is smaller', () => {
    // 96 px at 24, 60 px at 15: 'ABCDEFGH' goes on one line at 15 px.
    expect(typeset('ABCDEFGH', { w: 60, h: 18 }, measure)).toEqual({ fontSize: 15, lines: ['ABCDEFGH'], overflow: false })
  })

  it('counts a character outside the BMP as one', () => {
    // 𠮷 is two UTF-16 units: counted so, 𠮷野 would be 3 characters and 72 px, and a break could split the pair.
    expect(lay('𠮷野', 48)).toEqual({ fontSize: 24, lines: ['𠮷野'], overflow: false })
    expect(lay('𠮷𠮷𠮷', 48)).toEqual({ fontSize: 24, lines: ['𠮷𠮷', '𠮷'], overflow: false })
    // …and no break falls inside one, even when it is wider than the box: it stands on a line of its own.
    expect(lay('𠮷', 10)).toEqual({ fontSize: 14, lines: ['𠮷'], overflow: true })
    expect(lay('𠮷𠮷', 10)).toEqual({ fontSize: 14, lines: ['𠮷', '𠮷'], overflow: true })
  })
})

describe('no text', () => {
  it.each([null, undefined, '', ' ', '  \n\t ', '　'])('does not typeset %j', (text) => {
    expect(lay(text as string | null, 100)).toBeNull()
  })

  it('trims whitespace at both ends before it typesets', () => {
    expect(lay('  \n你好 \n ', 100)).toEqual({ fontSize: 24, lines: ['你好'], overflow: false })
  })
})

describe('the height of a block', () => {
  it('is its lines times the size times the line height', () => {
    const t = lay('一二\n三四', 200) as Typeset
    expect(blockHeight(t)).toBeCloseTo(2 * 24 * 1.2)
  })
})

const typesetBox = (overflow: boolean): Typeset => ({ fontSize: overflow ? 14 : 20, lines: ['一'], overflow })

describe('the overflow rate of a page', () => {
  it('counts the typeset boxes and the overflowing ones, and leaves the untranslated ones out of the rate', () => {
    const results = [typesetBox(false), typesetBox(true), null, typesetBox(false), null]
    expect(overflowRate(results)).toEqual({ typeset: 3, overflowing: 1, untranslated: 2, rate: 1 / 3 })
  })

  it('has no rate when no box was typeset, not a rate of 0', () => {
    expect(overflowRate([null, null])).toEqual({ typeset: 0, overflowing: 0, untranslated: 2, rate: null })
    expect(overflowRate([])).toEqual({ typeset: 0, overflowing: 0, untranslated: 0, rate: null })
  })

  it('has a rate of 0 when boxes were typeset and none overflows', () => {
    expect(overflowRate([typesetBox(false), null])).toEqual({ typeset: 1, overflowing: 0, untranslated: 1, rate: 0 })
  })

  it('comes from the segments of a page: a segment without a translation is untranslated', () => {
    const segments = [
      { w: 100, h: 100, text_zh_hans: '你好' },
      { w: 100, h: 100, text_zh_hans: null },
      { w: 28, h: 20, text_zh_hans: '一二三四五六七八九十' },
    ]
    const results = typesetPage(segments, measure)
    expect(results.map((r) => r?.overflow ?? null)).toEqual([false, null, true])
    expect(overflowRate(results)).toEqual({ typeset: 2, overflowing: 1, untranslated: 1, rate: 0.5 })
  })
})

describe('the overflow rate of a book', () => {
  const page = (typeset: number, overflowing: number, untranslated = 0): OverflowFigures => ({
    typeset,
    overflowing,
    untranslated,
    rate: typeset === 0 ? null : overflowing / typeset,
  })

  it('pools the boxes of its pages, so a page of one box weighs one box', () => {
    // 1 of 1 (100%) and 1 of 9 (11.1%): the mean of the two rates is 55.6%, the pooled rate is 2 of 10.
    const book = pooled([page(1, 1), page(9, 1)])
    expect(book).toEqual({ typeset: 10, overflowing: 2, untranslated: 0, rate: 0.2 })
    expect(book.rate).not.toBeCloseTo((1 + 1 / 9) / 2)
  })

  it('counts the pages with no typeset box in the untranslated boxes and leaves them out of the rate', () => {
    expect(pooled([page(4, 1, 2), page(0, 0, 3), page(6, 0)])).toEqual({
      typeset: 10,
      overflowing: 1,
      untranslated: 5,
      rate: 0.1,
    })
  })

  it('has no rate when no page has a typeset box', () => {
    expect(pooled([page(0, 0, 2), page(0, 0, 1)])).toEqual({ typeset: 0, overflowing: 0, untranslated: 3, rate: null })
    expect(pooled([])).toEqual({ typeset: 0, overflowing: 0, untranslated: 0, rate: null })
  })

  it('is summarized from the pages, which it keeps in the order given', () => {
    const summary = summarizeBook([
      { pageIndex: 2, results: [typesetBox(false), typesetBox(true), null] },
      { pageIndex: 3, results: [null, null] }, // regions, and none translated
      { pageIndex: 4, results: [] }, // no regions at all
      { pageIndex: 5, results: [typesetBox(true)] },
    ])
    expect(summary.pages.map((p) => [p.pageIndex, p.regions, p.figures.typeset, p.figures.overflowing, p.figures.untranslated])).toEqual([
      [2, 3, 2, 1, 1],
      [3, 2, 0, 0, 2],
      [4, 0, 0, 0, 0],
      [5, 1, 1, 1, 0],
    ])
    expect(summary.figures).toEqual({ typeset: 3, overflowing: 2, untranslated: 3, rate: 2 / 3 })
    expect(summary.overflowingPages).toEqual([2, 5])
    expect(summary.untranslatedPages).toEqual([3]) // page 4 has no regions, so it is not "regions but no translation"
  })

  it('has the sum of its pages for every count', () => {
    const results = [
      [typesetBox(true), typesetBox(false), null],
      [typesetBox(false)],
      [null, typesetBox(true), typesetBox(true)],
    ]
    const summary = summarizeBook(results.map((r, i) => ({ pageIndex: i + 1, results: r })))
    const sum = (key: 'typeset' | 'overflowing' | 'untranslated') => summary.pages.reduce((n, p) => n + p.figures[key], 0)
    expect(summary.figures.typeset).toBe(sum('typeset'))
    expect(summary.figures.overflowing).toBe(sum('overflowing'))
    expect(summary.figures.untranslated).toBe(sum('untranslated'))
  })
})

describe('words for the figures', () => {
  it('writes a rate as a percentage with at most one decimal', () => {
    expect([percent(0), percent(0.25), percent(1), percent(1 / 3), percent(0.125), percent(0.01)]).toEqual([
      '0%',
      '25%',
      '100%',
      '33.3%',
      '12.5%',
      '1%',
    ])
  })

  it('describes a page the way the page line does', () => {
    expect(describeFigures({ typeset: 8, overflowing: 2, untranslated: 1, rate: 0.25 })).toBe(
      '2 of 8 boxes overflow (25%) · 1 box not translated',
    )
    expect(describeFigures({ typeset: 8, overflowing: 1, untranslated: 0, rate: 0.125 })).toBe('1 of 8 boxes overflows (12.5%)')
    expect(describeFigures({ typeset: 1, overflowing: 0, untranslated: 3, rate: 0 })).toBe(
      '0 of 1 box overflow (0%) · 3 boxes not translated',
    )
  })

  it('says there is nothing to typeset, never 0%, when there is no rate', () => {
    expect(describeFigures({ typeset: 0, overflowing: 0, untranslated: 2, rate: null })).toBe(
      'no translation to typeset · 2 boxes not translated',
    )
    expect(describeFigures({ typeset: 0, overflowing: 0, untranslated: 0, rate: null })).toBe('no translation to typeset')
  })
})
