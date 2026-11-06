import type { DocumentItem, Source, StreamEvent } from './types.ts'

export const demoDocuments: DocumentItem[] = [
  { filename: 'made-up-handbook.txt', extension: '.txt', size_bytes: 2048 },
  { filename: 'made-up-notes.md', extension: '.md', size_bytes: 512 },
]

const sources: Source[] = [
  {
    evidence_id: 1,
    source: 'made-up-handbook.txt',
    chunk_id: 'demo-1',
    page: null,
    score: 0,
    text: 'This passage is made-up demo text. It stands in for a passage from one of your documents.',
  },
  {
    evidence_id: 2,
    source: 'made-up-notes.md',
    chunk_id: 'demo-2',
    page: 2,
    score: 0,
    text: 'Demo mode makes no network calls. Nothing here came from a model and nothing was checked.',
  },
]

const answer = 'This is a made-up answer for demo mode. [Evidence 1]\nNothing here was retrieved or checked. [Evidence 2]'

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(timer)
      reject(new DOMException('Aborted', 'AbortError'))
    })
  })
}

export async function* demoStream(_question: string, signal: AbortSignal): AsyncGenerator<StreamEvent> {
  await wait(300, signal)
  yield { event: 'stage', data: { i: 1, ms: 300 } }
  yield { event: 'sources', data: { items: sources } }
  for (const word of answer.replace(/ \[Evidence \d\]/g, '').split(' ')) {
    await wait(60, signal)
    yield { event: 'token', data: { t: `${word} ` } }
  }
  yield { event: 'stage', data: { i: 2, ms: 600 } }
  yield { event: 'final', data: { answer, supported: false, latency_ms: 900 } }
  yield { event: 'done', data: { latency_ms: 900, supported: false } }
}
