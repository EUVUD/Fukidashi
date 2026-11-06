import { useEffect, useRef, type KeyboardEvent } from 'react'

interface Props {
  index: number
  draft: string
  error: string | null
  saving: boolean // the request is in flight
  disabled: boolean // the page is being translated
  onChange: (draft: string) => void
  onSave: () => void
  onCancel: () => void
}

// The zh-Hans line of one row while a person edits it: a textarea with Save and Cancel.
// Esc cancels, Ctrl+Enter or Cmd+Enter saves.
export function TranslationEditor({ index, draft, error, saving, disabled, onChange, onSave, onCancel }: Props) {
  const busy = saving || disabled
  const field = useRef<HTMLTextAreaElement>(null)

  // Open with the caret after the text, so typing goes on from there.
  useEffect(() => {
    const el = field.current
    el?.setSelectionRange(el.value.length, el.value.length)
  }, [])
  // A disabled field loses focus: take it back when the row can be edited again (after a failed save, say).
  useEffect(() => {
    if (!busy) field.current?.focus()
  }, [busy])

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.nativeEvent.isComposing) return // an input method's own Enter and Esc are not ours
    if (e.key === 'Escape') {
      e.preventDefault()
      onCancel()
    } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      onSave()
    }
  }

  return (
    <div className="editor">
      <textarea
        ref={field}
        className="edit-text"
        lang="zh-Hans"
        rows={3}
        value={draft}
        disabled={busy}
        aria-label={`Translation of region ${index}`}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {error && <p className="error" role="alert">Saving failed: {error}</p>}
      <div className="editor-actions">
        <button type="button" onClick={onSave} disabled={busy}>
          {saving ? 'Saving…' : 'Save'}
        </button>
        <button type="button" className="secondary" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
        <span className="muted hint">Ctrl/Cmd+Enter saves · Esc cancels</span>
      </div>
    </div>
  )
}
