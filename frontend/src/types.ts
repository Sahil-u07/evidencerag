export type Health = {
  status: string
  ready: boolean
  generator: string
  model: string
  generator_available: boolean
  max_upload_mb: number
  read_only: boolean
}

export type DocumentItem = {
  filename: string
  extension: string
  size_bytes: number
}

export type Source = {
  evidence_id: number
  source: string
  chunk_id: string
  page: number | null
  score: number
  text: string
}

export type Verification = {
  supported: boolean
  cited_evidence: number[]
  unsupported_claims: string[]
  reason: string
}

export type StreamEvent =
  | { event: 'stage'; data: { i: number; ms: number } }
  | { event: 'sources'; data: { items: Source[] } }
  | { event: 'token'; data: { t: string } }
  | { event: 'verification'; data: Verification }
  | { event: 'final'; data: { answer: string; supported: boolean; latency_ms: number } }
  | { event: 'done'; data: { latency_ms: number; supported: boolean } }
  | { event: 'error'; data: { message: string } }

export type TurnStatus = 'running' | 'verified' | 'abstained' | 'failed' | 'stopped' | 'demo'

export type Turn = {
  id: number
  question: string
  demo: boolean
  status: TurnStatus
  stages: Record<number, number>
  sources: Source[]
  draft: string
  answer: string
  verification: Verification | null
  error: string
  latencyMs: number | null
}

export type HealthState =
  | { kind: 'connecting' }
  | { kind: 'offline' }
  | { kind: 'model-missing'; model: string; maxUploadMb: number; readOnly: boolean }
  | { kind: 'ready'; model: string; maxUploadMb: number; readOnly: boolean }

export type Theme = 'system' | 'light' | 'dark'
