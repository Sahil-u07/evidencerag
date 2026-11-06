import { createSseParser } from './lib/sse.ts'
import type { DocumentItem, Health, StreamEvent } from './types.ts'

export const API_BASE = (import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000').replace(/\/+$/, '')

export const UNREACHABLE = `Could not reach the backend at ${API_BASE}. Start the API, then try again.`

export function errorText(error: unknown): string {
  // fetch reports a refused or dropped connection as a TypeError.
  if (error instanceof TypeError) return UNREACHABLE
  return error instanceof Error ? error.message : String(error)
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (response.ok) return response

  const body: { detail?: unknown } = await response.json().catch(() => ({}))
  throw new Error(detailText(body.detail) ?? `The backend answered with status ${response.status}.`)
}

function detailText(detail: unknown): string | null {
  if (typeof detail === 'string') return detail
  // FastAPI validation errors are a list of { msg }.
  if (Array.isArray(detail)) return detail.map((item: { msg?: string }) => item.msg).join(' ')
  return null
}

export async function getHealth(signal: AbortSignal): Promise<Health> {
  return (await request('/health', { signal })).json()
}

export async function listDocuments(): Promise<DocumentItem[]> {
  const body: { documents: DocumentItem[] } = await (await request('/documents')).json()
  return body.documents
}

export async function uploadDocument(file: File): Promise<void> {
  const body = new FormData()
  body.append('file', file)
  await request('/documents/upload', { method: 'POST', body })
}

export async function deleteDocument(filename: string): Promise<void> {
  await request(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' })
}

export async function* streamQuery(query: string, signal: AbortSignal): AsyncGenerator<StreamEvent> {
  const response = await request('/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: 5 }),
    signal,
  })
  if (!response.body) return

  const feed = createSseParser()
  const decoder = new TextDecoder()
  const reader = response.body.getReader()
  for (;;) {
    const { done, value } = await reader.read()
    if (done) return
    for (const event of feed(decoder.decode(value, { stream: true }))) yield event as StreamEvent
  }
}
