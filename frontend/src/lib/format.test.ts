import assert from 'node:assert/strict'
import { test } from 'node:test'
import { checkUpload, formatMs, formatSize, originalName, splitCitations, splitLines } from './format.ts'

const uuid = '0123456789abcdef0123456789abcdef'

test('strips the stored prefix from a file name', () => {
  assert.equal(originalName(`${uuid}_report final.pdf`), 'report final.pdf')
})

test('keeps underscores that belong to the original name', () => {
  assert.equal(originalName(`${uuid}_a_b_c.txt`), 'a_b_c.txt')
})

test('leaves names without the 32 hex prefix alone', () => {
  assert.equal(originalName('abc_report.pdf'), 'abc_report.pdf')
  assert.equal(originalName('rfc9293.pdf'), 'rfc9293.pdf')
})

test('formats sizes', () => {
  assert.equal(formatSize(0), '0 B')
  assert.equal(formatSize(512), '512 B')
  assert.equal(formatSize(1024), '1 KB')
  assert.equal(formatSize(1536), '1.5 KB')
  assert.equal(formatSize(10 * 1024 * 1024), '10 MB')
  assert.equal(formatSize(5 * 1024 ** 3), '5 GB')
})

test('formats timings', () => {
  assert.equal(formatMs(120.4), '120 ms')
  assert.equal(formatMs(3456), '3.5 s')
})

test('splits text and citations', () => {
  assert.deepEqual(splitCitations('One. [Evidence 2] Two [evidence  10].'), [
    { text: 'One. ' },
    { cite: 2, raw: '[Evidence 2]' },
    { text: ' Two ' },
    { cite: 10, raw: '[evidence  10]' },
    { text: '.' },
  ])
})

test('returns plain text when there are no citations', () => {
  assert.deepEqual(splitCitations('Plain.'), [{ text: 'Plain.' }])
  assert.deepEqual(splitCitations(''), [])
})

test('splits lines and marks numbered and bulleted ones', () => {
  assert.deepEqual(splitLines('Intro.\n\n1. First\n2) Second\n- Third'), [
    { text: 'Intro.' },
    { marker: '1.', text: 'First' },
    { marker: '2)', text: 'Second' },
    { marker: '-', text: 'Third' },
  ])
})

test('does not treat a decimal at the start of a line as a list marker', () => {
  assert.deepEqual(splitLines('3.5 percent grew.'), [{ text: '3.5 percent grew.' }])
})

test('accepts the allowed types in any case', () => {
  assert.equal(checkUpload('a.PDF', 10, 10), '')
  assert.equal(checkUpload('a.txt', 10, 10), '')
  assert.equal(checkUpload('a.md', 10, 10), '')
})

test('rejects a wrong type and names the file', () => {
  assert.match(checkUpload('notes.docx', 10, 10), /notes\.docx.*PDF, TXT or MD/)
})

test('rejects a file over the server limit and states the limit', () => {
  assert.match(checkUpload('big.txt', 11 * 1024 * 1024, 10), /big\.txt.*11 MB.*10 MB/)
  assert.equal(checkUpload('ok.txt', 10 * 1024 * 1024, 10), '')
})
