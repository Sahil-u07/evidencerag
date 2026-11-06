import { useEffect, useRef, useState } from 'react'
import type { EvidenceTarget } from '../hooks/useEvidence.ts'
import { originalName, splitCitations } from '../lib/format.ts'
import type { Source, Turn } from '../types.ts'
import { Icon } from './Icon.tsx'

type Props = {
  turn: Turn | null
  target: EvidenceTarget | null
  hoveredId: number | null
  inline?: boolean
}

function citedIds(turn: Turn): number[] {
  if (turn.verification) return turn.verification.cited_evidence
  return splitCitations(turn.answer).flatMap((part) => ('cite' in part ? [part.cite] : []))
}

function citedLabel(turn: Turn, cited: boolean): string {
  if (turn.status === 'verified' || turn.status === 'demo') return cited ? 'Cited' : 'Not cited'
  if (turn.status === 'abstained') return cited ? 'Cited in the rejected draft' : 'Not cited'
  return ''
}

export function EvidencePanel({ turn, target, hoveredId, inline = false }: Props) {
  const list = useRef<HTMLOListElement>(null)
  const scrollTo = (id: number, block: ScrollLogicalPosition) => {
    list.current?.querySelector(`[data-evidence="${id}"]`)?.scrollIntoView({ block })
  }

  useEffect(() => {
    if (target) scrollTo(target.id, 'center')
  }, [target])

  useEffect(() => {
    if (hoveredId !== null) scrollTo(hoveredId, 'nearest')
  }, [hoveredId])

  return (
    <section className={inline ? 'evidence evidence--inline' : 'evidence'} aria-label="Evidence">
      {!inline && (
        <header className="evidence__header">
          <h2 className="section-label">Evidence</h2>
          {turn && <p className="evidence__for">{turn.question}</p>}
        </header>
      )}
      <div className="evidence__scroll" role={inline ? undefined : 'region'} aria-label={inline ? undefined : 'Passages'} tabIndex={inline ? undefined : 0}>
        <EvidenceBody turn={turn} target={target} hoveredId={hoveredId} listRef={list} />
      </div>
    </section>
  )
}

type BodyProps = {
  turn: Turn | null
  target: EvidenceTarget | null
  hoveredId: number | null
  listRef: React.RefObject<HTMLOListElement | null>
}

function EvidenceBody({ turn, target, hoveredId, listRef }: BodyProps) {
  if (!turn) {
    return (
      <p className="evidence__note">
        Ask a question and the passages behind its answer appear here, in rank order, with the ones the answer cites
        marked.
      </p>
    )
  }

  if (turn.sources.length === 0) {
    if (turn.status === 'running') return <p className="evidence__note">Searching your documents…</p>
    const endedEarly = (turn.status === 'failed' || turn.status === 'stopped') && turn.stages[1] === undefined
    return (
      <p className="evidence__note evidence__none">
        {endedEarly ? 'The request ended before any passages were found.' : 'No passages were found for this question.'}
      </p>
    )
  }

  const cited = citedIds(turn)
  return (
    <>
      <p className="evidence__note">
        {turn.sources.length} {turn.sources.length === 1 ? 'passage' : 'passages'}, in rank order.
      </p>
      <ol ref={listRef} className="passages">
        {turn.sources.map((source) => (
          <Passage
            key={source.evidence_id}
            source={source}
            label={citedLabel(turn, cited.includes(source.evidence_id))}
            cited={cited.includes(source.evidence_id)}
            active={target?.id === source.evidence_id}
            hovered={hoveredId === source.evidence_id}
          />
        ))}
      </ol>
    </>
  )
}

type PassageProps = { source: Source; label: string; cited: boolean; active: boolean; hovered: boolean }

function Passage({ source, label, cited, active, hovered }: PassageProps) {
  const [expanded, setExpanded] = useState(false)
  const [clamped, setClamped] = useState(false)
  const text = useRef<HTMLParagraphElement>(null)

  // Runs after Overlay's layout effect has opened the dialog. A closed dialog measures as zero.
  useEffect(() => {
    const element = text.current
    if (element && !expanded) setClamped(element.scrollHeight > element.clientHeight + 1)
  }, [expanded, source.text])

  const name = originalName(source.source)
  const classes = ['passage', cited && 'passage--cited', active && 'passage--active', hovered && 'passage--hovered']

  return (
    <li className={classes.filter(Boolean).join(' ')} data-evidence={source.evidence_id} aria-current={active ? 'true' : undefined}>
      <div className="passage__head">
        <span className="passage__rank">
          <span className="sr-only">Rank </span>
          {source.evidence_id}
        </span>
        <span className="passage__source">
          <span className="passage__name" title={name}>
            {name}
          </span>
          {source.page !== null && <span className="passage__page">page {source.page}</span>}
        </span>
      </div>
      {label && (
        <p className={cited ? 'passage__cited passage__cited--yes' : 'passage__cited'}>
          {cited && <Icon name="check" size={14} />}
          {label}
        </p>
      )}
      <p ref={text} className={expanded ? 'passage__text' : 'passage__text passage__text--clamped'}>
        {source.text}
      </p>
      {clamped && (
        <button type="button" className="link-btn" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </li>
  )
}
