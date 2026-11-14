// The typesetter's one font, and the measure the browser gives it. The measure and the drawing (TypesetLayer.tsx) both use
// FONT_FAMILY, so a line breaks where it is drawn. Common Simplified Chinese fonts, then the browser's sans-serif, and no web
// font: hanzi are 1 em wide in such fonts but Latin text varies, so the overflow rate is the browser's and may differ a
// little between machines (OUTLINE §8, "Typesetting and the overflow rate"). This module touches the DOM, so it has no test.
import type { Measure } from './typeset'

export const FONT_FAMILY =
  '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "WenQuanYi Micro Hei", sans-serif'

let context: CanvasRenderingContext2D | null | undefined // undefined: not asked for yet; null: the browser has no 2D canvas
let fontSet = ''

/** The width in pixels of `text` at `size` px in FONT_FAMILY, from a 2D canvas context's `measureText`. */
export const measureText: Measure = (text, size) => {
  if (context === undefined) context = document.createElement('canvas').getContext('2d')
  // Without a 2D canvas Konva cannot draw either; this keeps the editor standing with OUTLINE §8's estimate, 1 em a character.
  if (context === null) return Array.from(text).length * size
  const font = `${size}px ${FONT_FAMILY}`
  if (font !== fontSet) {
    context.font = font
    fontSet = font
  }
  return context.measureText(text).width
}
