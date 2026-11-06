export type SseEvent = { event: string; data: unknown }

export function createSseParser(): (chunk: string) => SseEvent[] {
  let buffer = ''
  let event = 'message'
  let dataLines: string[] = []

  return (chunk) => {
    buffer += chunk
    // A trailing \r stays in the buffer because its \n may be in the next chunk.
    const lines = buffer.split(/\r\n|\n|\r(?!$)/)
    buffer = lines.pop() ?? ''

    const events: SseEvent[] = []
    for (const line of lines) {
      if (line === '') {
        if (dataLines.length > 0) events.push(decode(event, dataLines.join('\n')))
        event = 'message'
        dataLines = []
      } else if (line.startsWith('event:')) {
        event = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).replace(/^ /, ''))
      }
    }
    return events
  }
}

function decode(event: string, data: string): SseEvent {
  try {
    return { event, data: JSON.parse(data) }
  } catch {
    return { event: 'error', data: { message: `The backend sent a ${event} event that could not be read.` } }
  }
}
