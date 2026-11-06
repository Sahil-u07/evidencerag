import { Fragment } from 'react'
import { formatMs } from '../lib/format.ts'
import type { Turn } from '../types.ts'
import { Icon } from './Icon.tsx'

const STAGES = ['Retrieval', 'Generation', 'Verification']

export function Progress({ turn }: { turn: Turn }) {
  if (turn.status !== 'running') {
    const parts = STAGES.flatMap((name, index) => {
      const ms = turn.stages[index + 1]
      return ms === undefined ? [] : [`${name} ${formatMs(ms)}`]
    })
    if (turn.latencyMs !== null) parts.push(`Total ${formatMs(turn.latencyMs)}`)
    if (parts.length === 0) return null
    return (
      <p className="timings">
        {parts.map((part, index) => (
          <Fragment key={part}>
            {index > 0 && ' · '}
            <span className="timings__part">{part}</span>
          </Fragment>
        ))}
      </p>
    )
  }

  const runningIndex = STAGES.findIndex((_, index) => turn.stages[index + 1] === undefined)

  return (
    <ol className="progress" aria-label="Progress">
      {STAGES.map((name, index) => {
        const ms = turn.stages[index + 1]
        const state = ms !== undefined ? 'done' : index === runningIndex ? 'running' : 'pending'
        return (
          <li key={name} className={`progress__stage progress__stage--${state}`}>
            {state === 'done' && <Icon name="check" />}
            {state === 'running' && <Icon name="spinner" className="spin" />}
            {state === 'pending' && <Icon name="dot" />}
            <span>{name}</span>
            {ms !== undefined && <span className="progress__ms">{formatMs(ms)}</span>}
            {state === 'running' && <span className="progress__ms">running</span>}
          </li>
        )
      })}
    </ol>
  )
}
