import { useLayoutEffect, useRef, type KeyboardEvent, type ReactNode, type SyntheticEvent } from 'react'
import { Icon } from './Icon.tsx'

const FOCUSABLE =
  'a[href], button:not([disabled]), textarea:not([disabled]), summary, input:not([disabled]):not([type="radio"]), input[type="radio"]:checked, [tabindex="0"]'

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

  const trapTab = (event: KeyboardEvent<HTMLDialogElement>) => {
    if (event.key !== 'Tab') return
    const items = Array.from(event.currentTarget.querySelectorAll<HTMLElement>(FOCUSABLE))
    const visible = items.filter((item) => item.getClientRects().length > 0)
    const first = visible[0]
    const last = visible[visible.length - 1]
    const active = document.activeElement
    if (event.shiftKey && (active === first || active === event.currentTarget)) {
      event.preventDefault()
      last?.focus()
    } else if (!event.shiftKey && active === last) {
      event.preventDefault()
      first?.focus()
    }
  }

  return (
    <dialog
      ref={ref}
      className={`overlay overlay--${side}`}
      aria-label={label}
      onCancel={onCancel}
      onClose={onNativeClose}
      onKeyDown={trapTab}
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
