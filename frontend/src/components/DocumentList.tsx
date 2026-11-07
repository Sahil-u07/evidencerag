import { useEffect, useRef, useState } from 'react'
import type { DocumentsState } from '../hooks/useDocuments.ts'
import { formatSize, originalName } from '../lib/format.ts'
import type { DocumentItem } from '../types.ts'
import { Icon } from './Icon.tsx'

type Props = { state: DocumentsState; demo: boolean; readOnly: boolean; streaming: boolean; stale: boolean }

export function DocumentList({ state, demo, readOnly, streaming, stale }: Props) {
  const { documents, loaded, deleting, uploading } = state
  const error = state.deleteError || state.loadError
  const [confirming, setConfirming] = useState<string | null>(null)
  const heading = useRef<HTMLHeadingElement>(null)

  const confirmDelete = async (filename: string) => {
    setConfirming(null)
    await state.remove(filename)
    // The row and its button are gone, so give focus somewhere that still exists.
    heading.current?.focus()
  }

  return (
    <section className="documents" aria-labelledby="documents-heading">
      <h2 id="documents-heading" className="section-label" ref={heading} tabIndex={-1}>
        Documents{loaded ? ` (${documents.length})` : ''}
      </h2>
      {stale && <p className="message">The backend is offline, so this list may be out of date.</p>}
      {error && (
        <p className="message message--error" role="alert">
          <Icon name="x-circle" />
          <span>{error}</span>
        </p>
      )}
      {!loaded && !stale && !error && <p className="documents__empty">Loading documents…</p>}
      {loaded && documents.length === 0 && (
        <p className="documents__empty">
          {readOnly ? 'No sample documents are loaded.' : 'No documents yet. Upload a PDF, TXT or MD file to ask questions about it.'}
        </p>
      )}
      <ul className="documents__list">
        {documents.map((doc) => (
          <DocumentRow
            key={doc.filename}
            doc={doc}
            canDelete={!demo && !readOnly}
            busy={deleting === doc.filename}
            // A rebuild stalls a running answer, and the backend does not queue rebuilds.
            locked={streaming || deleting !== null || uploading !== null}
            confirming={confirming === doc.filename}
            onAsk={() => setConfirming(doc.filename)}
            onCancel={() => setConfirming(null)}
            onConfirm={() => void confirmDelete(doc.filename)}
          />
        ))}
      </ul>
    </section>
  )
}

type RowProps = {
  doc: DocumentItem
  canDelete: boolean
  busy: boolean
  locked: boolean
  confirming: boolean
  onAsk: () => void
  onCancel: () => void
  onConfirm: () => void
}

function DocumentRow({ doc, canDelete, busy, locked, confirming, onAsk, onCancel, onConfirm }: RowProps) {
  const name = originalName(doc.filename)
  const trash = useRef<HTMLButtonElement>(null)
  const cancel = useRef<HTMLButtonElement>(null)
  const wasConfirming = useRef(false)

  useEffect(() => {
    if (confirming) cancel.current?.focus()
    else if (wasConfirming.current) trash.current?.focus()
    wasConfirming.current = confirming
  }, [confirming])

  if (confirming) {
    return (
      <li className="doc doc--confirm">
        <p className="doc__question">Delete {name}?</p>
        <div className="doc__actions">
          <button type="button" className="btn btn--danger" disabled={locked} onClick={onConfirm}>
            Delete
          </button>
          <button type="button" className="btn" ref={cancel} onClick={onCancel}>
            Cancel
          </button>
        </div>
      </li>
    )
  }

  return (
    <li className="doc">
      <div className="doc__text">
        <span className="doc__name" title={name}>
          {name}
        </span>
        <span className="doc__meta">
          {busy ? 'Deleting…' : `${doc.extension.replace('.', '').toUpperCase()}, ${formatSize(doc.size_bytes)}`}
        </span>
      </div>
      {canDelete && (
        <button ref={trash} type="button" className="icon-btn" aria-label={`Delete ${name}`} disabled={locked} onClick={onAsk}>
          <Icon name="trash" />
        </button>
      )}
    </li>
  )
}
