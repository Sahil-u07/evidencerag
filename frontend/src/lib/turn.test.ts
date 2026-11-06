import assert from 'node:assert/strict'
import { test } from 'node:test'
import { applyEvent, endOfStream, newTurn, verdictMessage } from './turn.ts'
import type { Source, Turn } from '../types.ts'

const source: Source = { evidence_id: 1, source: 'a.txt', chunk_id: 'c', page: null, score: 2, text: 'x' }

function running(): Turn {
  return newTurn(1, 'Why?', false)
}

test('a new turn is running', () => {
  assert.equal(running().status, 'running')
})

test('tokens build the draft and leave the answer empty', () => {
  const turn = applyEvent(applyEvent(running(), { event: 'token', data: { t: 'Dr' } }), { event: 'token', data: { t: 'aft' } })
  assert.equal(turn.draft, 'Draft')
  assert.equal(turn.answer, '')
  assert.equal(turn.status, 'running')
})

test('stages collect timings', () => {
  const turn = applyEvent(running(), { event: 'stage', data: { i: 1, ms: 12 } })
  assert.deepEqual(turn.stages, { 1: 12 })
})

test('sources are kept in the order sent', () => {
  const turn = applyEvent(running(), { event: 'sources', data: { items: [source] } })
  assert.deepEqual(turn.sources, [source])
})

test('a supported final replaces the draft text and verifies', () => {
  let turn = applyEvent(running(), { event: 'token', data: { t: 'draft' } })
  turn = applyEvent(turn, { event: 'final', data: { answer: 'final [Evidence 1]', supported: true, latency_ms: 9 } })
  assert.equal(turn.answer, 'final [Evidence 1]')
  assert.equal(turn.status, 'verified')
  assert.equal(turn.latencyMs, 9)
})

test('an unsupported final abstains and keeps the rejected draft', () => {
  let turn = applyEvent(running(), { event: 'token', data: { t: 'made up' } })
  turn = applyEvent(turn, { event: 'final', data: { answer: 'No evidence.', supported: false, latency_ms: 9 } })
  assert.equal(turn.status, 'abstained')
  assert.equal(turn.answer, 'No evidence.')
  assert.equal(turn.draft, 'made up')
})

test('a demo turn is never verified, even if final says supported', () => {
  const turn = applyEvent(newTurn(1, 'Q', true), { event: 'final', data: { answer: 'A', supported: true, latency_ms: 1 } })
  assert.equal(turn.status, 'demo')
})

test('an error event fails the turn with the server message', () => {
  const turn = applyEvent(running(), { event: 'error', data: { message: 'Run: ollama pull x' } })
  assert.equal(turn.status, 'failed')
  assert.equal(turn.error, 'Run: ollama pull x')
})

test('an error after final still fails the turn and hides the answer', () => {
  let turn = applyEvent(running(), { event: 'final', data: { answer: 'A', supported: true, latency_ms: 1 } })
  turn = applyEvent(turn, { event: 'error', data: { message: 'boom' } })
  assert.equal(turn.status, 'failed')
  assert.equal(turn.answer, '')
})

test('done does not change a finished verdict', () => {
  let turn = applyEvent(running(), { event: 'final', data: { answer: 'A', supported: false, latency_ms: 1 } })
  turn = applyEvent(turn, { event: 'done', data: { supported: false, latency_ms: 2 } })
  assert.equal(turn.status, 'abstained')
})

test('a stream that ends with no final fails', () => {
  const turn = endOfStream(applyEvent(running(), { event: 'token', data: { t: 'x' } }), 'lost')
  assert.equal(turn.status, 'failed')
  assert.equal(turn.error, 'lost')
})

test('a finished turn is unchanged at the end of the stream', () => {
  const finished = applyEvent(running(), { event: 'final', data: { answer: 'A', supported: true, latency_ms: 1 } })
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
