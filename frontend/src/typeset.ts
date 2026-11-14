// The typesetter: where a translation goes inside its box, and which boxes it does not fit. Pure logic with no React, no DOM,
// no canvas and no Konva: the text measure is passed in, so the module loads anywhere and Vitest checks it directly
// (OUTLINE §8, "Frontend tests"). The rules are OUTLINE §8, "Typesetting and the overflow rate". TypesetLayer.tsx draws the
// result on a canvas, and canvasText.ts is the measure the browser gives it.

// ---- The constants (OUTLINE §8, "Typesetting and the overflow rate") -------------------------------------------------

/** The sizes tried are integers in image pixels, from `MAX_FONT_PX` down to `MIN_FONT_PX`, each 1 px smaller than the one
 *  before. All 214 OpenMantra pages are 1170 px tall, so a size means the same on every page. */
export const MAX_FONT_PX = 24
export const MIN_FONT_PX = 14
/** A line takes `size × LINE_HEIGHT`. There is no padding: OpenMantra's boxes sit inside the bubbles. */
export const LINE_HEIGHT = 1.2

/** A line does not start with one of these: the full-width marks, then the half-width ones. */
export const CLOSING_MARKS = '，。、．：；！？…）」』】〕〉》”’' + ',.:;!?)]}'
/** A line does not end with one of these. */
export const OPENING_MARKS = '（「『【〔〈《“‘' + '([{'

// ---- Typesetting one box ----------------------------------------------------------------------------------------------

/** The width in pixels of `text` drawn at `fontSizePx`, in the font that will draw it (canvasText.ts gives the browser's). */
export type Measure = (text: string, fontSizePx: number) => number

/** A box's size in image pixels; a `Segment` has it too. */
export interface BoxSize {
  w: number
  h: number
}

export interface Typeset {
  fontSize: number // an integer in image pixels, `MAX_FONT_PX` down to `MIN_FONT_PX`
  lines: string[] // what is drawn, one string per line; none holds a line break
  overflow: boolean // the text does not fit at `MIN_FONT_PX`; `lines` are laid out at that size all the same
}

const CLOSING = new Set(CLOSING_MARKS)
const OPENING = new Set(OPENING_MARKS)

const isLetterOrDigit = (ch: string) => /^[A-Za-z0-9]$/.test(ch)
// A closing mark, or `……`. An atom is one or more characters that stay together, so every character must be one.
const isClosing = (atom: string) => Array.from(atom).every((ch) => CLOSING.has(ch))
const isOpening = (atom: string) => OPENING.has(atom)
const isRun = (atom: string) => atom.length > 1 && isLetterOrDigit(atom[0])

/** The characters of one paragraph, a code point being one character, with what is never split kept whole: a run of ASCII
 *  letters and digits (broken by character later, when it does not fit on a line), and `……` and `——`. */
function atomize(paragraph: string): string[] {
  const chars = Array.from(paragraph)
  const atoms: string[] = []
  for (let i = 0; i < chars.length; ) {
    const ch = chars[i]
    if (isLetterOrDigit(ch)) {
      let end = i + 1
      while (end < chars.length && isLetterOrDigit(chars[end])) end++
      atoms.push(chars.slice(i, end).join(''))
      i = end
    } else if ((ch === '…' || ch === '—') && chars[i + 1] === ch) {
      atoms.push(ch + ch)
      i += 2
    } else {
      atoms.push(ch)
      i++
    }
  }
  return atoms
}

/** A run of letters and digits stays on one line when it fits on a line, and is broken by character when it does not. */
const breakLongRuns = (atoms: readonly string[], width: number, size: number, measure: Measure): string[] =>
  atoms.flatMap((atom) => (isRun(atom) && measure(atom, size) > width ? Array.from(atom) : [atom]))

/** The atoms that no line break may separate. A closing mark stays with what comes before it and an opening mark with what
 *  comes after it, because a break between them would start a line with the one or end a line with the other. So a closing
 *  mark that does not fit at the end of a line takes the line's last character with it to the next line, and so does a
 *  chain of them. */
function bind(atoms: readonly string[]): string[] {
  const units: string[] = []
  let endsOpening = false
  for (const atom of atoms) {
    if (units.length > 0 && (endsOpening || isClosing(atom))) units[units.length - 1] += atom
    else units.push(atom)
    endsOpening = isOpening(atom)
  }
  return units
}

interface Layout {
  lines: string[]
  fits: boolean // every line is at most as wide as the box, and the lines are at most as high
}

/** The paragraphs laid out at `size`, filling each line as far as it goes. A unit wider than the box takes a line of its
 *  own and the layout does not fit: that is how a closing mark that would leave its line empty stays, and the line counts
 *  as too wide. With `stopWhenFull` it gives up as soon as the size cannot fit, with `lines` as far as they got. */
function layOut(paragraphs: readonly (readonly string[])[], box: BoxSize, size: number, measure: Measure, stopWhenFull: boolean): Layout {
  const lines: string[] = []
  let tooWide = false
  const tooTall = () => lines.length * size * LINE_HEIGHT > box.h
  for (const atoms of paragraphs) {
    let line = ''
    for (const unit of bind(breakLongRuns(atoms, box.w, size, measure))) {
      if (line !== '' && measure(line + unit, size) <= box.w) {
        line += unit
        continue
      }
      if (line !== '') {
        lines.push(line)
        if (stopWhenFull && tooTall()) return { lines, fits: false }
      }
      line = unit
      if (measure(line, size) > box.w) {
        tooWide = true
        if (stopWhenFull) return { lines, fits: false }
      }
    }
    lines.push(line) // an empty paragraph (two line breaks in a row) is an empty line
    if (stopWhenFull && tooTall()) return { lines, fits: false }
  }
  return { lines, fits: !tooWide && !tooTall() }
}

/** `text` laid out horizontally inside `box`, at the largest size from `MAX_FONT_PX` down to `MIN_FONT_PX` at which it fits.
 *  A box overflows when its text does not fit at `MIN_FONT_PX`, and it is laid out at that size all the same. Null when
 *  there is nothing to typeset: the text is null, empty or only whitespace. Leading and trailing whitespace is trimmed. */
export function typeset(text: string | null | undefined, box: BoxSize, measure: Measure): Typeset | null {
  const body = text?.replace(/\r\n?/g, '\n').trim()
  if (!body) return null
  const paragraphs = body.split('\n').map(atomize)
  for (let size = MAX_FONT_PX; size > MIN_FONT_PX; size--) {
    const { lines, fits } = layOut(paragraphs, box, size, measure, true)
    if (fits) return { fontSize: size, lines, overflow: false }
  }
  const { lines, fits } = layOut(paragraphs, box, MIN_FONT_PX, measure, false)
  return { fontSize: MIN_FONT_PX, lines, overflow: !fits }
}

/** The height the lines take: what a box has to hold, and what the block's vertical centring uses. */
export const blockHeight = (t: Typeset): number => t.lines.length * t.fontSize * LINE_HEIGHT

/** A page's boxes typeset, one result per segment in order; null for a segment with no translation to typeset. */
export const typesetPage = (
  segments: readonly (BoxSize & { text_zh_hans: string | null })[],
  measure: Measure,
): (Typeset | null)[] => segments.map((s) => typeset(s.text_zh_hans, s, measure))

// ---- The overflow rate (OUTLINE §0, Overflow rate; §4.4: a product metric, not part of the §4.3 comparisons) ------------

export interface OverflowFigures {
  typeset: number // the boxes with a translation, typeset
  overflowing: number // of those, the boxes whose text does not fit at `MIN_FONT_PX`
  untranslated: number // the boxes with nothing to typeset: counted apart, never in the rate
  rate: number | null // overflowing / typeset, a fraction; null, not 0, when no box was typeset
}

const figuresOf = (typeset: number, overflowing: number, untranslated: number): OverflowFigures => ({
  typeset,
  overflowing,
  untranslated,
  rate: typeset === 0 ? null : overflowing / typeset,
})

/** The figures of a list of typeset results, a page's say: null entries are untranslated boxes. */
export function overflowRate(results: readonly (Typeset | null)[]): OverflowFigures {
  let typeset = 0
  let overflowing = 0
  let untranslated = 0
  for (const result of results) {
    if (result === null) {
      untranslated++
    } else {
      typeset++
      if (result.overflow) overflowing++
    }
  }
  return figuresOf(typeset, overflowing, untranslated)
}

/** The figures of several lists together, a book's say. The boxes are pooled and the rate comes from the pooled counts:
 *  it is never the mean of the lists' rates, which would weigh a page of one box like a page of twenty. */
export const pooled = (parts: readonly OverflowFigures[]): OverflowFigures =>
  figuresOf(
    parts.reduce((sum, p) => sum + p.typeset, 0),
    parts.reduce((sum, p) => sum + p.overflowing, 0),
    parts.reduce((sum, p) => sum + p.untranslated, 0),
  )

export interface PageFigures {
  pageIndex: number
  regions: number // the boxes on the page, translated or not
  figures: OverflowFigures
}

export interface BookFigures {
  pages: PageFigures[] // every page measured, in the order given
  figures: OverflowFigures // pooled over the pages
  untranslatedPages: number[] // pages with regions and no translation at all
  overflowingPages: number[] // pages with at least one overflowing box
}

/** A book's figures from its pages' typeset results (a page with no regions has an empty list). */
export function summarizeBook(pages: readonly { pageIndex: number; results: readonly (Typeset | null)[] }[]): BookFigures {
  const measured = pages.map(({ pageIndex, results }) => ({ pageIndex, regions: results.length, figures: overflowRate(results) }))
  return {
    pages: measured,
    figures: pooled(measured.map((p) => p.figures)),
    untranslatedPages: measured.filter((p) => p.regions > 0 && p.figures.typeset === 0).map((p) => p.pageIndex),
    overflowingPages: measured.filter((p) => p.figures.overflowing > 0).map((p) => p.pageIndex),
  }
}

// ---- Words for the figures --------------------------------------------------------------------------------------------

const plural = (n: number, one: string, many: string) => (n === 1 ? one : many)

/** A fraction as a percentage with at most one decimal: 0.25 is "25%", 1 / 3 is "33.3%". */
export const percent = (fraction: number): string => `${Number((fraction * 100).toFixed(1))}%`

/** The figures in one line: "2 of 8 boxes overflow (25%) · 1 box not translated". With no typeset box there is no rate, and
 *  the line says so instead of saying 0%. */
export function describeFigures(f: OverflowFigures): string {
  const parts =
    f.rate === null
      ? ['no translation to typeset']
      : [`${f.overflowing} of ${f.typeset} ${plural(f.typeset, 'box', 'boxes')} ${f.overflowing === 1 ? 'overflows' : 'overflow'} (${percent(f.rate)})`]
  if (f.untranslated > 0) parts.push(`${f.untranslated} ${plural(f.untranslated, 'box', 'boxes')} not translated`)
  return parts.join(' · ')
}
