import { originalName, splitCitations, splitLines } from '../lib/format.ts'
import type { Source } from '../types.ts'

type Props = {
  text: string
  sources: Source[]
  activeId: number | null
  onCite: (id: number) => void
  onHover: (id: number | null) => void
}

function evidenceLabel(source: Source): string {
  const page = source.page === null ? '' : `, page ${source.page}`
  return `Evidence ${source.evidence_id} from ${originalName(source.source)}${page}`
}

export function AnswerText({ text, sources, activeId, onCite, onHover }: Props) {
  return (
    <div className="answer">
      {splitLines(text).map((line, index) => (
        <p key={index} className={line.marker ? 'answer__line answer__line--item' : 'answer__line'}>
          {line.marker && <span className="answer__marker">{line.marker}</span>}
          <span>
            {splitCitations(line.text).map((part, partIndex) => {
              if ('text' in part) return part.text
              const source = sources.find((item) => item.evidence_id === part.cite)
              if (!source) return part.raw
              return (
                <button
                  key={partIndex}
                  type="button"
                  className={activeId === part.cite ? 'cite cite--active' : 'cite'}
                  aria-label={evidenceLabel(source)}
                  title={evidenceLabel(source)}
                  onClick={() => onCite(part.cite)}
                  onMouseEnter={() => onHover(part.cite)}
                  onMouseLeave={() => onHover(null)}
                  onFocus={() => onHover(part.cite)}
                  onBlur={() => onHover(null)}
                >
                  {part.cite}
                </button>
              )
            })}
          </span>
        </p>
      ))}
    </div>
  )
}
