import { useRef } from 'react'
import { Icon } from './Icon.tsx'

type Props = {
  demo: boolean
  uploading: string | null
  deleting: boolean
  error: string
  maxUploadMb: number
  onFiles: (files: File[]) => void
}

export function UploadControl({ demo, uploading, deleting, error, maxUploadMb, onFiles }: Props) {
  const input = useRef<HTMLInputElement>(null)

  return (
    <div className="upload">
      <input
        ref={input}
        type="file"
        accept=".pdf,.txt,.md"
        hidden
        onChange={(event) => {
          onFiles(Array.from(event.target.files ?? []))
          // Clear it so picking the same file again still fires change.
          event.target.value = ''
        }}
      />
      <button
        type="button"
        className="btn btn--block"
        disabled={demo || deleting || uploading !== null}
        aria-busy={uploading !== null}
        onClick={() => input.current?.click()}
      >
        <Icon name={uploading ? 'spinner' : 'upload'} className={uploading ? 'spin' : undefined} />
        {uploading ? 'Indexing…' : 'Upload a document'}
      </button>
      <div aria-live="polite">
        {uploading && <p className="upload__note upload__name">Indexing {uploading}. This can take a while.</p>}
        {error && (
          <p className="message message--error">
            <Icon name="x-circle" />
            <span>{error}</span>
          </p>
        )}
      </div>
      {!uploading && !error && (
        <p className="upload__note">
          {demo ? 'Uploads are off in demo mode.' : `PDF, TXT or MD, up to ${maxUploadMb} MB. You can also drop a file anywhere.`}
        </p>
      )}
    </div>
  )
}
