import { useEffect, type KeyboardEvent, type RefObject } from 'react'
import { API_BASE } from '../api.ts'
import type { HealthState } from '../types.ts'
import { Icon } from './Icon.tsx'

type Props = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onStop: () => void
  running: boolean
  health: HealthState
  demo: boolean
  indexing: boolean
  noDocuments: boolean
  inputRef: RefObject<HTMLTextAreaElement | null>
}

function blockReason(health: HealthState, demo: boolean, indexing: boolean): string {
  if (demo) return ''
  if (health.kind === 'offline') return `Sending is off because the backend at ${API_BASE} is offline.`
  if (health.kind === 'connecting') return 'Sending starts once the backend answers.'
  return indexing ? 'Sending is off while the index rebuilds.' : ''
}

export function Composer(props: Props) {
  const { value, onChange, onSend, onStop, running, health, demo, indexing, noDocuments, inputRef } = props
  const blockedReason = blockReason(health, demo, indexing)
  const hint = noDocuments
    ? 'No documents yet. You can still ask, but the answer will abstain.'
    : 'Enter to send, Shift+Enter for a new line.'
  const canSend = !running && !blockedReason && value.trim() !== ''

  // Grows with the text. The CSS max-height caps it at about 6 lines.
  useEffect(() => {
    const input = inputRef.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${input.scrollHeight + 2}px`
  }, [value, inputRef])

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return
    event.preventDefault()
    if (canSend) onSend()
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault()
        if (canSend) onSend()
      }}
    >
      <div className="composer__inner">
        <div className="composer__row">
          <label className="sr-only" htmlFor="question">
            Your question
          </label>
          <textarea
            id="question"
            ref={inputRef}
            rows={1}
            value={value}
            placeholder="Ask about your documents"
            aria-describedby="composer-hint"
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={onKeyDown}
          />
          {running ? (
            <button type="button" className="btn composer__button" onClick={onStop}>
              <Icon name="stop" />
              Stop
            </button>
          ) : (
            <button type="submit" className="btn btn--primary composer__button" disabled={!canSend}>
              <Icon name="send" />
              Send
            </button>
          )}
        </div>
        <p id="composer-hint" className={blockedReason ? 'composer__hint composer__hint--blocked' : 'composer__hint'}>
          {blockedReason || hint}
        </p>
      </div>
    </form>
  )
}
