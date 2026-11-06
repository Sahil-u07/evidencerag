import { useEffect, useState, type ReactNode } from 'react'
import type { Viewport } from '../hooks/useViewport.ts'
import type { Turn } from '../types.ts'
import { AnswerText } from './AnswerText.tsx'
import { Icon } from './Icon.tsx'
import { Progress } from './Progress.tsx'
import { Verdict } from './Verdict.tsx'

type Props = {
  turn: Turn
  selected: boolean
  evidenceOpen: boolean
  viewport: Viewport
  activeId: number | null
  busy: boolean
  inlineEvidence: ReactNode
  onCite: (id: number) => void
  onHover: (id: number | null) => void
  onShowEvidence: () => void
  onHideEvidence: () => void
  onRetry: () => void
}

export function TurnView(props: Props) {
  const { turn, selected, evidenceOpen, viewport, activeId, busy, inlineEvidence } = props
  const running = turn.status === 'running'
  const inlineOpen = viewport === 'phone' && selected && evidenceOpen
  const count = turn.sources.length

  return (
    <article
      className={selected && viewport === 'desktop' ? 'turn turn--selected' : 'turn'}
      aria-busy={running}
      aria-labelledby={`question-${turn.id}`}
    >
      <h2 className="turn__question" id={`question-${turn.id}`}>
        {turn.question}
      </h2>

      <Progress turn={turn} />

      {running && turn.draft && (
        <div className="draft">
          <p className="draft__label">
            <Icon name="warning" />
            Draft, not checked yet
          </p>
          <p className="draft-text">{turn.draft}</p>
        </div>
      )}

      <Verdict turn={turn} />

      {turn.answer && (
        <div className={turn.status === 'abstained' ? 'turn__answer abstention' : 'turn__answer'}>
          <AnswerText
            text={turn.answer}
            sources={turn.sources}
            activeId={selected ? activeId : null}
            onCite={props.onCite}
            onHover={props.onHover}
          />
        </div>
      )}

      <div className="turn__actions">
        {(turn.status === 'failed' || turn.status === 'stopped') && (
          <button type="button" className="btn" disabled={busy} onClick={props.onRetry}>
            <Icon name="retry" />
            Retry
          </button>
        )}
        {turn.answer && <CopyButton text={turn.answer} />}
        {count > 0 && viewport !== 'phone' && (
          <button
            type="button"
            className="btn btn--quiet"
            aria-haspopup={viewport === 'tablet' ? 'dialog' : undefined}
            onClick={props.onShowEvidence}
          >
            <Icon name="evidence" />
            Evidence ({count})
          </button>
        )}
        {count > 0 && viewport === 'phone' && (
          <button
            type="button"
            className="btn btn--quiet"
            aria-expanded={inlineOpen}
            onClick={inlineOpen ? props.onHideEvidence : props.onShowEvidence}
          >
            <Icon name="evidence" />
            {inlineOpen ? 'Hide evidence' : `Show evidence (${count})`}
          </button>
        )}
      </div>

      {inlineOpen && <div className="turn__evidence">{inlineEvidence}</div>}
    </article>
  )
}

function CopyButton({ text }: { text: string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle')

  useEffect(() => {
    if (state === 'idle') return
    const timer = setTimeout(() => setState('idle'), 2000)
    return () => clearTimeout(timer)
  }, [state])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setState('copied')
    } catch {
      setState('failed')
    }
  }

  return (
    <button type="button" className="btn btn--quiet" onClick={() => void copy()}>
      <Icon name={state === 'copied' ? 'check' : 'copy'} />
      {state === 'copied' ? 'Copied' : state === 'failed' ? 'Copy failed' : 'Copy answer'}
    </button>
  )
}
