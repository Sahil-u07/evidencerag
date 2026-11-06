import { useCallback, useState } from 'react'
import type { Viewport } from './useViewport.ts'

export type EvidenceTarget = { id: number; nonce: number }

export function useEvidence(viewport: Viewport) {
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null)
  const [target, setTarget] = useState<EvidenceTarget | null>(null)
  const [hovered, setHovered] = useState<{ turn: number; id: number } | null>(null)
  // Bound to the viewport it was opened in, so a citation clicked on desktop
  // does not pop open a slide-over when the window later narrows.
  const [openOn, setOpenOn] = useState<Viewport | null>(null)

  const select = useCallback((turn: number) => {
    setSelectedTurn(turn)
    setTarget(null)
    setOpenOn(null)
  }, [])

  const cite = useCallback(
    (turn: number, id: number) => {
      setSelectedTurn(turn)
      // The nonce lets a second click on the same citation scroll to it again.
      setTarget((current) => ({ id, nonce: (current?.nonce ?? 0) + 1 }))
      setOpenOn(viewport)
    },
    [viewport],
  )

  const show = useCallback(
    (turn: number) => {
      if (turn !== selectedTurn) setTarget(null)
      setSelectedTurn(turn)
      setOpenOn(viewport)
    },
    [selectedTurn, viewport],
  )

  const close = useCallback(() => setOpenOn(null), [])

  const hover = useCallback((turn: number, id: number | null) => {
    setHovered(id === null ? null : { turn, id })
  }, [])

  const reset = useCallback(() => {
    setSelectedTurn(null)
    setTarget(null)
    setHovered(null)
    setOpenOn(null)
  }, [])

  return { selectedTurn, target, hovered, open: openOn === viewport, select, cite, show, close, hover, reset }
}
