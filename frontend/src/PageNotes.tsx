import type { PageReview } from './api'
import { aboutText, reviewLine, reviewSubject, type Note, type RegionNotes } from './reviewNotes'

// How the page's review ended, near the run line. The words are reviewNotes.ts's.
export function ReviewLine({ review }: { review: PageReview }) {
  return (
    <p className="review-line">
      <strong>{reviewSubject(review)}:</strong> {reviewLine(review)}
    </p>
  )
}

// A box's marker on the image: "?" for a Flag, "!" for an Issue, each with its count when there are several. The symbols tell
// the two apart, the colours only echo them. Hollow when none of the region's notes is about the translation on show. The
// box carries the words (aria-label and title), so the marker is for the eye only.
export function NoteMarks({ notes }: { notes: RegionNotes }) {
  return (
    <span className={notes.current === 0 ? 'marks earlier' : 'marks'} aria-hidden="true">
      {notes.flags > 0 && <span className="mark flag">?{notes.flags > 1 ? notes.flags : ''}</span>}
      {notes.issues > 0 && <span className="mark issue">!{notes.issues > 1 ? notes.issues : ''}</span>}
    </span>
  )
}

// The notes of one region in its row, under its translation: each with its kind and round, its text (Simplified Chinese),
// its suggestion when it has one, and, when it is not about the translation on show, a line that says what it was about.
// A region with no notes has no list.
export function NoteList({ region, notes }: { region: number; notes: readonly Note[] }) {
  if (notes.length === 0) return null
  return (
    <ul className="notes" aria-label={`Review notes of region ${region}`}>
      {notes.map((note, i) => {
        const about = aboutText(note.about)
        return (
          <li key={i} className={`note ${note.kind}${about ? ' earlier' : ''}`}>
            <p className="note-head">
              <span className="tag">{note.label}</span>
              {about && <span className="about">{about}</span>}
            </p>
            <p className="note-text" lang="zh-Hans">{note.text}</p>
            {note.suggestion && (
              <p className="note-suggestion">
                <span className="muted">Suggestion:</span> <span lang="zh-Hans">{note.suggestion}</span>
              </p>
            )}
          </li>
        )
      })}
    </ul>
  )
}
