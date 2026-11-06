import { useSyncExternalStore } from 'react'

export type Viewport = 'phone' | 'tablet' | 'desktop'

function current(): Viewport {
  if (window.matchMedia('(min-width: 1024px)').matches) return 'desktop'
  return window.matchMedia('(min-width: 768px)').matches ? 'tablet' : 'phone'
}

function subscribe(onChange: () => void) {
  window.addEventListener('resize', onChange)
  return () => window.removeEventListener('resize', onChange)
}

export function useViewport(): Viewport {
  return useSyncExternalStore(subscribe, current)
}
