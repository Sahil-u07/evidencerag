import type { ReactNode } from 'react'

const SAMPLES = [
  'Summarise the main points of this document',
  'What are the key terms and how are they defined?',
  'What dates, numbers or deadlines are mentioned?',
]

const RFC_SAMPLES = [
  'What does the iss claim in a JWT mean?',
  'How does the TCP three-way handshake work?',
  'What does an HTTP 404 status code mean?',
]

type Props = {
  noDocuments: boolean
  readOnly: boolean
  hasDocuments: boolean
  upload: ReactNode
  onSample: (question: string) => void
}

export function EmptyState({ noDocuments, readOnly, hasDocuments, upload, onSample }: Props) {
  return (
    <div className="empty">
      <h2 className="empty__title">{noDocuments && !readOnly ? 'Upload a document to begin' : readOnly ? 'Ask about the sample documents' : 'Ask about your documents'}</h2>
      <p className="empty__text">
        Answers come only from your files. Each one shows the passages behind it and is checked against them. When the
        evidence isn't there, the app says so instead of answering.
      </p>
      {noDocuments && !readOnly && (upload ?? <p className="empty__text">Use the upload button in the left column, or drop a file anywhere in this window.</p>)}
      {hasDocuments && (
        <div className="samples">
          <h3 className="section-label">Try asking</h3>
          <ul>
            {(readOnly ? RFC_SAMPLES : SAMPLES).map((sample) => (
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
