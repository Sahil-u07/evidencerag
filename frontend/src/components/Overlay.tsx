import { useLayoutEffect, useRef, type ReactNode, type SyntheticEvent } from 'react'
import { Icon } from './Icon.tsx'

type Props = { label: string; side: 'left' | 'right'; onClose: () => void; children: ReactNode }

export function Overlay({ label, side, onClose, children }: Props) {
  const ref = useRef<HTMLDialogElement>(null)

  // A layout effect, so the panel is on screen before its children's effects scroll it.
  useLayoutEffect(() => {
    const dialog = ref.current
    const opener = document.activeElement as HTMLElement | null
    dialog?.showModal()
    return () => {
      dialog?.close()
      opener?.focus()
    }
  }, [])

  const onCancel = (event: SyntheticEvent) => {
    event.preventDefault()
    onClose()
  }

  // The browser can close a dialog itself on a repeated Escape, without a cancel event.
  const onNativeClose = () => {
    if (!ref.current?.open) onClose()
  }

  return (
    <dialog
      ref={ref}
      className={`overlay overlay--${side}`}
      aria-label={label}
      onCancel={onCancel}
      onClose={onNativeClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="overlay__body">
        <button type="button" className="icon-btn overlay__close" aria-label={`Close ${label.toLowerCase()}`} onClick={onClose}>
          <Icon name="close" size={20} />
        </button>
        {children}
      </div>
    </dialog>
  )
}
