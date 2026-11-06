import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createSseParser } from './sse.ts'

test('parses one event', () => {
  const feed = createSseParser()
  assert.deepEqual(feed('event: token\ndata: {"t":"Hi"}\n\n'), [{ event: 'token', data: { t: 'Hi' } }])
})

test('parses several events from one chunk', () => {
  const feed = createSseParser()
  const events = feed('event: stage\ndata: {"i":1,"ms":5}\n\nevent: token\ndata: {"t":"a"}\n\n')
  assert.deepEqual(events.map((e) => e.event), ['stage', 'token'])
})

test('keeps an event split across chunks', () => {
  const feed = createSseParser()
  assert.deepEqual(feed('event: to'), [])
  assert.deepEqual(feed('ken\ndata: {"t"'), [])
  assert.deepEqual(feed(':"Hi"}\n'), [])
  assert.deepEqual(feed('\n'), [{ event: 'token', data: { t: 'Hi' } }])
})

test('handles CRLF split between chunks', () => {
  const feed = createSseParser()
  assert.deepEqual(feed('event: done\r\ndata: {"latency_ms":1,"supported":true}\r\n\r'), [])
  assert.deepEqual(feed('\n'), [{ event: 'done', data: { latency_ms: 1, supported: true } }])
})

test('joins multi-line data with a newline', () => {
  const feed = createSseParser()
  assert.deepEqual(feed('event: final\ndata: {"answer":\ndata: "x"}\n\n'), [{ event: 'final', data: { answer: 'x' } }])
})

test('ignores comments and keeps going', () => {
  const feed = createSseParser()
  assert.deepEqual(feed(': ping\n\nevent: token\ndata: {"t":"a"}\n\n'), [{ event: 'token', data: { t: 'a' } }])
})

test('turns malformed JSON into an error event instead of throwing', () => {
  const feed = createSseParser()
  const [event] = feed('event: token\ndata: {not json\n\n')
  assert.equal(event.event, 'error')
  assert.match((event.data as { message: string }).message, /token/)
})

test('carries on after a malformed event', () => {
  const feed = createSseParser()
  const events = feed('event: token\ndata: oops\n\nevent: token\ndata: {"t":"a"}\n\n')
  assert.deepEqual(events[1], { event: 'token', data: { t: 'a' } })
})
