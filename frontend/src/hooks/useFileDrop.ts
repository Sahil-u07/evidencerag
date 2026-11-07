import { useEffect, useState } from 'react'

function hasFiles(event: DragEvent): boolean {
  return Boolean(event.dataTransfer?.types.includes('Files'))
}

// When disabled, a dropped file is still swallowed so the browser does not open it.
export function useFileDrop(onFiles: (files: File[]) => void, enabled: boolean): boolean {
  const [dragging, setDragging] = useState(false)

  useEffect(() => {
    // dragenter and dragleave also fire for every child, so count depth.
    let depth = 0

    const enter = (event: DragEvent) => {
      if (!enabled || !hasFiles(event)) return
      depth += 1
      setDragging(true)
    }
    const leave = (event: DragEvent) => {
      if (!enabled || !hasFiles(event)) return
      depth -= 1
      if (depth <= 0) setDragging(false)
    }
    const over = (event: DragEvent) => {
      if (hasFiles(event)) event.preventDefault()
    }
    const drop = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      if (!enabled) return
      depth = 0
      setDragging(false)
      onFiles(Array.from(event.dataTransfer?.files ?? []))
    }

    window.addEventListener('dragenter', enter)
    window.addEventListener('dragleave', leave)
    window.addEventListener('dragover', over)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('dragover', over)
      window.removeEventListener('drop', drop)
    }
  }, [onFiles, enabled])

  return dragging
}
