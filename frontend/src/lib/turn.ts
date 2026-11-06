import type { StreamEvent, Turn } from '../types.ts'

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

export function applyEvent(turn: Turn, { event, data }: StreamEvent): Turn {
  switch (event) {
    case 'stage':
      return { ...turn, stages: { ...turn.stages, [data.i]: data.ms } }
    case 'sources':
      return { ...turn, sources: data.items }
    case 'token':
      return { ...turn, draft: turn.draft + data.t }
    case 'verification':
      return { ...turn, verification: data }
    case 'final':
      return {
        ...turn,
        answer: data.answer,
        latencyMs: data.latency_ms,
        status: turn.demo ? 'demo' : data.supported ? 'verified' : 'abstained',
      }
    case 'done':
      return turn
    case 'error':
      return { ...turn, status: 'failed', answer: '', error: data.message }
    default:
      return turn
  }
}

export function endOfStream(turn: Turn, connectionMessage: string): Turn {
  return turn.status === 'running' ? { ...turn, status: 'failed', error: connectionMessage } : turn
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
