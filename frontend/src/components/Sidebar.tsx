import type { ReactNode } from 'react'
import type { DocumentsState } from '../hooks/useDocuments.ts'
import type { Theme } from '../types.ts'
import { DocumentList } from './DocumentList.tsx'
import { Icon } from './Icon.tsx'
import { ThemeSwitch } from './ThemeSwitch.tsx'

type Props = {
  // In the phone drawer the top bar already holds the page's h1.
  brandTag: 'h1' | 'h2'
  demo: boolean
  streaming: boolean
  documents: DocumentsState
  readOnly: boolean
  stale: boolean
  upload: ReactNode
  theme: Theme
  onTheme: (theme: Theme) => void
  onNewChat: () => void
  onDemoChange: (demo: boolean) => void
}

export function Sidebar({ brandTag: Brand, demo, streaming, documents, readOnly, stale, upload, theme, onTheme, onNewChat, onDemoChange }: Props) {
  return (
    <div className="sidebar">
      <div className="sidebar__top">
        <Brand className="brand">EvidenceRAG</Brand>
        <button type="button" className="btn btn--block" onClick={onNewChat}>
          <Icon name="plus" />
          New chat
        </button>
      </div>
      <div className="sidebar__middle">
        <DocumentList state={documents} demo={demo} readOnly={readOnly} streaming={streaming} stale={stale} />
      </div>
      <div className="sidebar__bottom">
        {upload}
        <div className="sidebar__settings">
          <ThemeSwitch theme={theme} onChange={onTheme} />
          <label className="demo-toggle">
            <input type="checkbox" checked={demo} onChange={(event) => onDemoChange(event.target.checked)} />
            Demo data
          </label>
        </div>
      </div>
    </div>
  )
}
