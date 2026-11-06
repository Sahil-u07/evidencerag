import type { ReactNode } from 'react'

const SAMPLES = [
  'Summarise the main points of this document',
  'What are the key terms and how are they defined?',
  'What dates, numbers or deadlines are mentioned?',
]

type Props = {
  noDocuments: boolean
  hasDocuments: boolean
  upload: ReactNode
  onSample: (question: string) => void
}

export function EmptyState({ noDocuments, hasDocuments, upload, onSample }: Props) {
  return (
    <div className="empty">
      <h2 className="empty__title">{noDocuments ? 'Upload a document to begin' : 'Ask about your documents'}</h2>
      <p className="empty__text">
        Answers come only from your files. Each one shows the passages behind it and is checked against them. When the
        evidence isn't there, the app says so instead of answering.
      </p>
      {noDocuments && (upload ?? <p className="empty__text">Use the upload button in the left column, or drop a file anywhere in this window.</p>)}
      {hasDocuments && (
        <div className="samples">
          <h3 className="section-label">Try asking</h3>
          <ul>
            {SAMPLES.map((sample) => (
              <li key={sample}>
                <button type="button" className="sample" onClick={() => onSample(sample)}>
                  {sample}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
