import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createSseParser } from './sse.ts'
import { applyEvent, endOfStream, newTurn, verdictMessage } from './turn.ts'
import type { Source, Turn } from '../types.ts'

const source: Source = { evidence_id: 1, source: 'a.txt', chunk_id: 'c', page: null, score: 2, text: 'x' }
const passed = { supported: true, cited_evidence: [1], unsupported_claims: [], reason: 'ok' }
const rejected = { supported: false, cited_evidence: [1], unsupported_claims: ['claim'], reason: 'no' }
const final = { answer: 'final [Evidence 1]', supported: true, latency_ms: 9 }

function running(): Turn {
  return newTurn(1, 'Why?', false)
}

function play(...events: { event: string; data: unknown }[]): Turn {
  return events.reduce(applyEvent, running())
}

test('a new turn is running', () => {
  assert.equal(running().status, 'running')
})

test('tokens build the draft and leave the answer empty', () => {
  const turn = play({ event: 'token', data: { t: 'Dr' } }, { event: 'token', data: { t: 'aft' } })
  assert.equal(turn.draft, 'Draft')
  assert.equal(turn.answer, '')
  assert.equal(turn.status, 'running')
})

test('stages collect timings', () => {
  assert.deepEqual(play({ event: 'stage', data: { i: 1, ms: 12 } }).stages, { 1: 12 })
})

test('sources are kept in the order sent', () => {
  assert.deepEqual(play({ event: 'sources', data: { items: [source] } }).sources, [source])
})

test('a passing verification and a supported final replace the draft and verify', () => {
  const turn = play({ event: 'token', data: { t: 'draft' } }, { event: 'verification', data: passed }, { event: 'final', data: final })
  assert.equal(turn.answer, final.answer)
  assert.equal(turn.status, 'verified')
  assert.equal(turn.latencyMs, 9)
})

test('a supported final with no verification result fails instead of verifying', () => {
  const turn = play({ event: 'token', data: { t: 'draft' } }, { event: 'final', data: final })
  assert.equal(turn.status, 'failed')
  assert.match(turn.error, /without a passing verification result/)
  assert.equal(turn.answer, '')
})

test('a supported final after a failing verification fails instead of verifying', () => {
  const turn = play({ event: 'verification', data: rejected }, { event: 'final', data: final })
  assert.equal(turn.status, 'failed')
  assert.equal(turn.answer, '')
})

test('an unsupported final abstains and keeps the rejected draft', () => {
  const turn = play(
    { event: 'token', data: { t: 'made up' } },
    { event: 'verification', data: rejected },
    { event: 'final', data: { answer: 'No evidence.', supported: false, latency_ms: 9 } },
  )
  assert.equal(turn.status, 'abstained')
  assert.equal(turn.answer, 'No evidence.')
  assert.equal(turn.draft, 'made up')
})

test('a demo turn is never verified, even with a passing verification', () => {
  const demo = newTurn(1, 'Q', true)
  assert.equal(applyEvent(demo, { event: 'final', data: final }).status, 'demo')
  assert.equal(applyEvent(applyEvent(demo, { event: 'verification', data: passed }), { event: 'final', data: final }).status, 'demo')
})

test('an error event fails the turn with the server message', () => {
  const turn = play({ event: 'error', data: { message: 'Run: ollama pull x' } })
  assert.equal(turn.status, 'failed')
  assert.equal(turn.error, 'Run: ollama pull x')
})

test('an error after final still fails the turn and hides the answer', () => {
  const turn = play({ event: 'verification', data: passed }, { event: 'final', data: final }, { event: 'error', data: { message: 'boom' } })
  assert.equal(turn.status, 'failed')
  assert.equal(turn.answer, '')
})

test('nothing but an error changes a finished turn', () => {
  const finished = play({ event: 'verification', data: rejected }, { event: 'final', data: { answer: 'No.', supported: false, latency_ms: 1 } })
  for (const event of [
    { event: 'token', data: { t: 'late' } },
    { event: 'sources', data: { items: [source] } },
    { event: 'stage', data: { i: 3, ms: 1 } },
    { event: 'verification', data: passed },
    { event: 'final', data: final },
    { event: 'done', data: { supported: true, latency_ms: 2 } },
  ]) {
    assert.equal(applyEvent(finished, event), finished, event.event)
  }
})

test('an error followed by a passing verification and final stays failed', () => {
  const turn = play({ event: 'error', data: { message: 'boom' } }, { event: 'verification', data: passed }, { event: 'final', data: final })
  assert.equal(turn.status, 'failed')
  assert.equal(turn.error, 'boom')
  assert.equal(turn.answer, '')
})

test('a malformed verification event followed by a valid final stays failed', () => {
  const feed = createSseParser()
  const events = feed(`event: verification\ndata: {oops\n\nevent: final\ndata: ${JSON.stringify(final)}\n\n`)
  const turn = events.reduce(applyEvent, running())
  assert.equal(turn.status, 'failed')
  assert.match(turn.error, /verification event that could not be read/)
})

test('events of the wrong shape fail the turn instead of being stored', () => {
  for (const [event, data] of [
    ['stage', { i: 'one', ms: 1 }],
    ['sources', { items: 5 }],
    ['sources', 5],
    ['sources', { items: [{ evidence_id: 1 }] }],
    ['sources', { items: [{ ...source, text: 7 }] }],
    ['sources', { items: [{ ...source, page: '3' }] }],
    ['token', { t: 5 }],
    ['token', null],
    ['verification', { supported: true }],
    ['verification', { ...passed, cited_evidence: 'all' }],
    ['verification', { ...passed, unsupported_claims: [1] }],
    ['final', { supported: true, latency_ms: 1 }],
    ['final', { answer: 'x', supported: 'yes', latency_ms: 1 }],
  ] as [string, unknown][]) {
    const turn = play({ event, data })
    assert.equal(turn.status, 'failed', JSON.stringify([event, data]))
    assert.match(turn.error, new RegExp(`${event} event that could not be read`))
    assert.deepEqual(turn.sources, [])
  }
})

test('an error event with no usable message still fails with words', () => {
  const turn = play({ event: 'error', data: {} })
  assert.equal(turn.status, 'failed')
  assert.notEqual(turn.error, '')
})

test('unknown events are ignored', () => {
  assert.equal(play({ event: 'ping', data: 1 }).status, 'running')
})

test('a stream that ends with no final fails', () => {
  const turn = endOfStream(play({ event: 'token', data: { t: 'x' } }), 'lost')
  assert.equal(turn.status, 'failed')
  assert.equal(turn.error, 'lost')
})

test('a finished turn is unchanged at the end of the stream', () => {
  const finished = play({ event: 'verification', data: passed }, { event: 'final', data: final })
  assert.equal(endOfStream(finished, 'lost').status, 'verified')
})

test('each verdict has one announcement', () => {
  const base = running()
  assert.match(verdictMessage({ ...base, status: 'verified' }), /Verified/)
  assert.match(verdictMessage({ ...base, status: 'abstained' }), /Not enough evidence/)
  assert.match(verdictMessage({ ...base, status: 'failed', error: 'boom' }), /boom/)
  assert.match(verdictMessage({ ...base, status: 'stopped' }), /stopped/i)
  assert.match(verdictMessage({ ...base, status: 'demo' }), /nothing was checked/)
})
