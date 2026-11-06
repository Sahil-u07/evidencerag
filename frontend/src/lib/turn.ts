import type { Source, Turn, Verification } from '../types.ts'
import type { SseEvent } from './sse.ts'

export function newTurn(id: number, question: string, demo: boolean): Turn {
  return {
    id,
    question,
    demo,
    status: 'running',
    stages: {},
    sources: [],
    draft: '',
    answer: '',
    verification: null,
    error: '',
    latencyMs: null,
  }
}

type Fields = Record<string, unknown>

function fields(data: unknown): Fields {
  return typeof data === 'object' && data !== null ? (data as Fields) : {}
}

function isSource(value: unknown): value is Source {
  const { evidence_id, source, page, text } = fields(value)
  return typeof evidence_id === 'number' && typeof source === 'string' && (page === null || typeof page === 'number') && typeof text === 'string'
}

function isVerification(value: Fields): value is Verification {
  const { supported, cited_evidence, unsupported_claims, reason } = value
  return (
    typeof supported === 'boolean' &&
    typeof reason === 'string' &&
    Array.isArray(cited_evidence) &&
    cited_evidence.every((id) => typeof id === 'number') &&
    Array.isArray(unsupported_claims) &&
    unsupported_claims.every((claim) => typeof claim === 'string')
  )
}

function fail(turn: Turn, message: string): Turn {
  return { ...turn, status: 'failed', answer: '', error: message }
}

export function applyEvent(turn: Turn, { event, data }: SseEvent): Turn {
  // A verdict is final: nothing a stream sends afterwards can upgrade it, only an error can.
  if (turn.status !== 'running' && event !== 'error') return turn

  const value = fields(data)
  const unreadable = () => fail(turn, `The backend sent a ${event} event that could not be read.`)

  switch (event) {
    case 'stage': {
      const { i, ms } = value
      return typeof i === 'number' && typeof ms === 'number' ? { ...turn, stages: { ...turn.stages, [i]: ms } } : unreadable()
    }
    case 'sources': {
      const { items } = value
      return Array.isArray(items) && items.every(isSource) ? { ...turn, sources: items } : unreadable()
    }
    case 'token':
      return typeof value.t === 'string' ? { ...turn, draft: turn.draft + value.t } : unreadable()
    case 'verification':
      return isVerification(value) ? { ...turn, verification: value } : unreadable()
    case 'final': {
      const { answer, supported, latency_ms } = value
      if (typeof answer !== 'string' || typeof supported !== 'boolean' || typeof latency_ms !== 'number') return unreadable()
      const finished = { ...turn, answer, latencyMs: latency_ms }
      if (turn.demo) return { ...finished, status: 'demo' }
      if (!supported) return { ...finished, status: 'abstained' }
      return turn.verification?.supported
        ? { ...finished, status: 'verified' }
        : fail(turn, 'The backend sent an answer without a passing verification result.')
    }
    case 'error':
      return fail(turn, typeof value.message === 'string' ? value.message : 'The backend reported an error with no message.')
    default:
      return turn
  }
}

export function endOfStream(turn: Turn, connectionMessage: string): Turn {
  return turn.status === 'running' ? fail(turn, connectionMessage) : turn
}

export function verdictMessage(turn: Turn): string {
  switch (turn.status) {
    case 'verified':
      return 'Verified. Checked against the cited passages.'
    case 'abstained':
      return 'Not enough evidence. The answer was withheld.'
    case 'failed':
      return `Failed. ${turn.error}`
    case 'stopped':
      return 'Stopped. This answer was not finished.'
    case 'demo':
      return 'Demo answer, nothing was checked.'
    case 'running':
      return ''
  }
}
