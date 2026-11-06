import type { Turn, TurnStatus } from '../types.ts'
import { Icon, type IconName } from './Icon.tsx'

const LOOK: Record<Exclude<TurnStatus, 'running'>, { icon: IconName; title: string }> = {
  verified: { icon: 'check-circle', title: 'Verified' },
  abstained: { icon: 'warning', title: 'Not enough evidence' },
  failed: { icon: 'x-circle', title: 'Failed' },
  stopped: { icon: 'stop', title: 'Stopped' },
  demo: { icon: 'info', title: 'Demo answer, nothing was checked' },
}

function description(turn: Turn): string {
  switch (turn.status) {
    case 'verified':
      return 'Checked against the cited passages.'
    case 'abstained':
      return turn.sources.length === 0
        ? 'No passages were found, so there was nothing to check an answer against.'
        : 'The draft could not be backed by the passages found, so it was withheld.'
    case 'failed':
      return turn.error
    case 'stopped':
      return 'You stopped this answer before it was checked.'
    default:
      return ''
  }
}

function UncheckedDraft({ turn, label }: { turn: Turn; label: string }) {
  return (
    <details className="disclosure disclosure--draft">
      <summary>{label}</summary>
      <p className="draft-text">{turn.draft}</p>
    </details>
  )
}

export function Verdict({ turn }: { turn: Turn }) {
  if (turn.status === 'running') return null
  const { icon, title } = LOOK[turn.status]
  const claims = turn.verification?.unsupported_claims ?? []
  const text = description(turn)

  return (
    <div className={`verdict verdict--${turn.status}`}>
      <p className="verdict__head">
        <Icon name={icon} size={18} className="verdict__icon" />
        <strong className="verdict__title">{title}</strong>
      </p>
      {text && <p className="verdict__text">{text}</p>}

      {turn.status === 'verified' && turn.verification?.reason && (
        <details className="disclosure">
          <summary>Why it passed</summary>
          <p>{turn.verification.reason}</p>
        </details>
      )}

      {turn.status === 'abstained' && turn.verification?.reason && (
        <p className="verdict__text verdict__reason">
          Reason: {turn.verification.reason}
        </p>
      )}

      {turn.status === 'abstained' && claims.length > 0 && (
        <div className="removed">
          <p className="verdict__text">Sentences removed because they could not be supported:</p>
          <ul>
            {claims.map((claim, index) => (
              <li key={index}>{claim}</li>
            ))}
          </ul>
        </div>
      )}

      {turn.status === 'abstained' && turn.sources.length > 0 && turn.draft && (
        <UncheckedDraft turn={turn} label="Rejected draft, unverified" />
      )}
      {(turn.status === 'failed' || turn.status === 'stopped') && turn.draft && (
        <UncheckedDraft turn={turn} label="Partial draft, unverified" />
      )}
    </div>
  )
}
