import { useEffect, useRef, type ReactNode } from 'react'
import type { Turn } from '../types.ts'

const NEAR_BOTTOM_PX = 48

export function Thread({ turns, children }: { turns: Turn[]; children: ReactNode }) {
  const scroller = useRef<HTMLDivElement>(null)
  const following = useRef(true)
  const lastTop = useRef(0)
  const count = turns.length

  useEffect(() => {
    following.current = true
  }, [count])

  useEffect(() => {
    const element = scroller.current
    if (element && following.current) element.scrollTop = element.scrollHeight
  }, [turns])

  return (
    <div
      ref={scroller}
      className="thread"
      role="region"
      aria-label="Conversation"
      tabIndex={0}
      onScroll={(event) => {
        const { scrollHeight, scrollTop, clientHeight } = event.currentTarget
        // Only an upward move counts as the user leaving, because new content can arrive
        // between a programmatic scroll and its event.
        if (scrollHeight - scrollTop - clientHeight < NEAR_BOTTOM_PX) following.current = true
        else if (scrollTop < lastTop.current) following.current = false
        lastTop.current = scrollTop
      }}
    >
      <div className="thread__column">{children}</div>
    </div>
  )
}
