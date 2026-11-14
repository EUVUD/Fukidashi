import type Konva from 'konva'
import { useLayoutEffect, useRef, useState } from 'react'
import { Group, Layer, Rect, Stage, Text } from 'react-konva'
import type { Segment } from './api'
import { FONT_FAMILY, measureText } from './canvasText'
import { LINE_HEIGHT, blockHeight, type Typeset } from './typeset'

interface Props {
  segments: readonly Segment[]
  results: readonly (Typeset | null)[] // one per segment, as `typesetPage` gives them
  natural: { w: number; h: number } // the image's natural size: the canvas draws in these pixels
}

// The translations drawn over the page image (OUTLINE §8, "Typesetting and the overflow rate"). The Konva Stage draws in
// image pixels, scaled to the size the image is shown at, and follows it when the window or the layout resizes it. It takes
// no pointer events, so the boxes of the DOM above it keep their hover and badges. A box with nothing to typeset is not
// covered: the original shows through.
export function TypesetLayer({ segments, results, natural }: Props) {
  const layer = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0) // the layer fills the image's frame, so this is the width the image is shown at
  useLayoutEffect(() => {
    const el = layer.current
    if (!el) return
    setWidth(el.getBoundingClientRect().width)
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Konva clears a canvas it resizes and draws it again a frame later; drawing at once keeps that frame from showing the original.
  const stage = useRef<Konva.Stage>(null)
  useLayoutEffect(() => {
    stage.current?.draw()
  }, [width, segments, results, natural])

  const scale = width / natural.w
  return (
    <div className="typeset-layer" ref={layer} aria-hidden="true">
      {width > 0 && (
        <Stage ref={stage} width={width} height={natural.h * scale} scaleX={scale} scaleY={scale}>
          <Layer listening={false}>
            {segments.map((s, i) => {
              const t = results[i]
              return t ? <TypesetBox key={s.index} segment={s} typeset={t} /> : null
            })}
          </Layer>
        </Stage>
      )}
    </div>
  )
}

// One box: a white rectangle over it, and the lines as `typeset` laid them out (no wrapping of Konva's own), centred in the
// box and clipped to it. A line wider or a block taller than the box (an overflow) starts at the box's edge instead, so the
// start of the text stays readable.
function TypesetBox({ segment: s, typeset: t }: { segment: Segment; typeset: Typeset }) {
  const widest = Math.max(...t.lines.map((line) => measureText(line, t.fontSize)))
  return (
    <Group clipX={s.x} clipY={s.y} clipWidth={s.w} clipHeight={s.h}>
      <Rect x={s.x} y={s.y} width={s.w} height={s.h} fill="#fff" />
      <Text
        id={`region-${s.index}`}
        name="typeset-text"
        x={s.x + Math.max(0, (s.w - widest) / 2)}
        y={s.y + Math.max(0, (s.h - blockHeight(t)) / 2)}
        text={t.lines.join('\n')}
        fontSize={t.fontSize}
        fontFamily={FONT_FAMILY}
        lineHeight={LINE_HEIGHT}
        align="center"
        wrap="none"
        fill="#000"
      />
    </Group>
  )
}
