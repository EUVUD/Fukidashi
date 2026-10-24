import type { MouseEvent, ReactNode } from 'react'

interface Props {
  to: string
  navigate: (path: string) => void
  className?: string
  title?: string
  children: ReactNode
}

// A real <a href>, so open-in-new-tab keeps working; a plain click navigates without reloading.
export function Link({ to, navigate, className, title, children }: Props) {
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return
    e.preventDefault()
    navigate(to)
  }
  return (
    <a href={to} onClick={onClick} className={className} title={title}>
      {children}
    </a>
  )
}
