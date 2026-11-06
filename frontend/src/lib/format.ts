export type CitationPart = { text: string } | { cite: number; raw: string }
export type AnswerLine = { text: string; marker?: string }

export function originalName(storedName: string): string {
  return storedName.replace(/^[0-9a-f]{32}_/, '')
}

export function formatSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${Number(value.toFixed(unit === 0 ? 0 : 1))} ${units[unit]}`
}

export function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

export function splitCitations(text: string): CitationPart[] {
  const parts: CitationPart[] = []
  let last = 0
  for (const match of text.matchAll(/\[Evidence\s+(\d+)\]/gi)) {
    if (match.index > last) parts.push({ text: text.slice(last, match.index) })
    parts.push({ cite: Number(match[1]), raw: match[0] })
    last = match.index + match[0].length
  }
  if (last < text.length) parts.push({ text: text.slice(last) })
  return parts
}

export function splitLines(text: string): AnswerLine[] {
  const lines: AnswerLine[] = []
  for (const line of text.split('\n')) {
    if (!line.trim()) continue
    const match = line.match(/^\s*(\d+[.)]|[-*•])\s+(.*)$/)
    lines.push(match ? { marker: match[1], text: match[2] } : { text: line.trim() })
  }
  return lines
}

export function checkUpload(name: string, sizeBytes: number, maxMb: number): string {
  if (!/\.(pdf|txt|md)$/i.test(name)) return `${name} can't be uploaded. Use a PDF, TXT or MD file.`
  if (sizeBytes > maxMb * 1024 * 1024) return `${name} is ${formatSize(sizeBytes)}. The limit is ${maxMb} MB.`
  return ''
}
