import { useCallback, useEffect, useState } from 'react'
import { getHealth } from '../api.ts'
import type { HealthState } from '../types.ts'

const POLL_MS = 15_000
const TIMEOUT_MS = 10_000

// The backend does not answer /health while it indexes an upload, so `paused` stops
// that silence from reading as offline.
export function useHealth(paused: boolean) {
  const [health, setHealth] = useState<HealthState>({ kind: 'connecting' })
  const [refreshes, setRefreshes] = useState(0)

  useEffect(() => {
    if (paused) return
    const controller = new AbortController()

    const check = async () => {
      try {
        const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(TIMEOUT_MS)])
        const { ready, generator_available, model, max_upload_mb, read_only } = await getHealth(signal)
        if (!ready) setHealth({ kind: 'connecting' })
        else setHealth({ kind: generator_available ? 'ready' : 'model-missing', model, maxUploadMb: max_upload_mb, readOnly: read_only })
      } catch {
        if (!controller.signal.aborted) setHealth({ kind: 'offline' })
      }
    }

    void check()
    const timer = setInterval(check, POLL_MS)
    return () => {
      clearInterval(timer)
      controller.abort()
    }
  }, [paused, refreshes])

  const refresh = useCallback(() => setRefreshes((count) => count + 1), [])
  return { health, refresh }
}
