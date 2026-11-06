import type { HealthState } from '../types.ts'
import { Icon } from './Icon.tsx'
import { StatusLine } from './StatusLine.tsx'

type Props = { health: HealthState; demo: boolean; phone: boolean; onOpenDocuments: () => void }

export function TopBar({ health, demo, phone, onOpenDocuments }: Props) {
  return (
    <header className="topbar">
      {phone && (
        <div className="topbar__phone">
          <button type="button" className="icon-btn" aria-label="Open documents" aria-haspopup="dialog" onClick={onOpenDocuments}>
            <Icon name="menu" size={20} />
          </button>
          <h1 className="brand">EvidenceRAG</h1>
        </div>
      )}
      <StatusLine health={health} demo={demo} />
    </header>
  )
}
