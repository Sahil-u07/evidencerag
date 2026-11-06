import { useCallback, useRef, useState } from 'react'
import { errorText, streamQuery, UNREACHABLE } from '../api.ts'
import { demoStream } from '../demo.ts'
import { applyEvent, endOfStream, newTurn, verdictMessage } from '../lib/turn.ts'
import type { Turn } from '../types.ts'

export function useChat(demo: boolean, onFailed: () => void) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [announcement, setAnnouncement] = useState('')
  const active = useRef<AbortController | null>(null)
  const lastId = useRef(0)

  const run = useCallback(
    async (start: Turn) => {
      const controller = new AbortController()
      active.current = controller
      setAnnouncement('')

      let turn = start
      const publish = (next: Turn) => {
        turn = next
        setTurns((current) => current.map((item) => (item.id === next.id ? next : item)))
      }

      const stream = demo ? demoStream : streamQuery
      try {
        for await (const event of stream(start.question, controller.signal)) publish(applyEvent(turn, event))
        publish(endOfStream(turn, UNREACHABLE))
      } catch (error) {
        if (!controller.signal.aborted) publish({ ...turn, status: 'failed', answer: '', error: errorText(error) })
        else if (turn.status === 'running') publish({ ...turn, status: 'stopped' })
      }

      // New chat has already moved on, so this turn is gone and must not be announced.
      if (active.current !== controller) return
      active.current = null
      setAnnouncement(verdictMessage(turn))
      if (turn.status === 'failed') onFailed()
    },
    [demo, onFailed],
  )

  const send = useCallback(
    (question: string): number => {
      lastId.current += 1
      const turn = newTurn(lastId.current, question, demo)
      setTurns((current) => [...current, turn])
      void run(turn)
      return turn.id
    },
    [demo, run],
  )

  const retry = useCallback(
    (turn: Turn) => {
      const fresh = newTurn(turn.id, turn.question, demo)
      setTurns((current) => current.map((item) => (item.id === turn.id ? fresh : item)))
      void run(fresh)
    },
    [demo, run],
  )

  const stop = useCallback(() => active.current?.abort(), [])

  const clear = useCallback(() => {
    const running = active.current
    active.current = null
    running?.abort()
    setTurns([])
    setAnnouncement('')
  }, [])

  return { turns, running: turns.some((turn) => turn.status === 'running'), announcement, send, retry, stop, clear }
}
