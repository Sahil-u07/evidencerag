# Harder benchmark design

## Why

Reranked Hybrid already sits at the limit of the current 20-question benchmark: Recall@1 is 0.650 (the maximum possible) and MRR is 1.000. The corpus has 33 chunks and the reranker sees 20 of them. Every file is Markdown, so the benchmark cannot detect the PDF bug where only 8 of 59 chunks survive fusion.

A new benchmark should be able to fail. Its headline score comes from questions that were never used for tuning.

## Goals

- Measure retrieval on a larger real corpus with PDFs and exact technical identifiers.
- Measure the verifier directly, without needing Ollama.
- Fix the retrieval and verifier bugs, and show each fix as a before and after number.
- Report results with confidence intervals so noise is not read as progress.

## Out of scope for this work

- End-to-end answer evaluation, which needs Ollama.
- UI changes, `main.py` restructuring, CI and README polish. These follow as separate work, except for the README results table.

## Benchmark sets

### v1 regression set

The current 20 questions, corpus and 50/10 word chunking stay frozen, including the stray `\#` characters in those files, because the labels are chunk IDs and any text change shifts them. The new script must reproduce the current numbers (0.650, 0.975, 0.975, 1.000, 0.973) before the old scripts are deleted.

### v2 corpus

Stored unmodified in `data/evaluation/v2/corpus/` with a `SOURCES.md` giving each file's origin and the IETF Trust notice. About 2 MB in total.

| File | Topic | Format | Source path on rfc-editor.org | Bytes |
|---|---|---|---:|---:|
| `rfc9293.pdf` | TCP | PDF | `/rfc/rfc9293.pdf` | 879,498 |
| `rfc7519.pdf` | JWT | PDF | `/rfc/pdfrfc/rfc7519.txt.pdf` | 45,856 |
| `rfc1035.txt` | DNS | Text | `/rfc/rfc1035.txt` | 122,549 |
| `rfc9110.txt` | HTTP semantics | Text | `/rfc/rfc9110.txt` | 502,941 |
| `rfc8446.txt` | TLS 1.3 | Text | `/rfc/rfc8446.txt` | 337,736 |
| `rfc6749.txt` | OAuth 2.0 | Text | `/rfc/rfc6749.txt` | 163,498 |

RFCs fit because they are public, real documents with page headers and footers, and they are full of exact identifiers such as `SYN-RECEIVED`, `Retry-After` and `invalid_grant`.

### v2 questions

`data/evaluation/v2/questions.json` holds 80 questions, split evenly between `dev` and `test`, with the same mix of types and documents in each split:

| Type | Count | What it tests |
|---|---:|---|
| `paraphrase` | 40 | Wording differs from the document |
| `identifier` | 24 | The question depends on an exact term |
| `multi_fact` | 16 | The answer needs two separate facts |

```json
{
  "id": "v2-017",
  "split": "dev",
  "type": "identifier",
  "query": "Which TCP state does a server enter after sending its reply to a connection request?",
  "evidence": ["short exact phrase from the document"]
}
```

Each `evidence` item is one required fact, written as an exact phrase of 5 to 15 words from a single sentence. A chunk is relevant if it contains any phrase after whitespace is normalized. Labels are text rather than chunk IDs, so they stay valid when chunking changes.

The loader rejects a question when:

- a phrase matches no chunk, or
- a phrase matches more than 3 chunks under the default chunking, because it is too common to prove anything.

## Metrics

Reported per question type and overall:

- Hit@1, Hit@3 and Hit@5: at least one relevant chunk in the top k.
- Recall@k: facts covered in the top k, divided by the number of facts.
- Capped recall@k: the same, divided by `min(k, facts)`, so a perfect system scores 1.0 at every k.
- MRR and nDCG@5.
- Median and p95 retrieval latency.
- 95% confidence intervals from a seeded bootstrap over questions.

v1 keeps its existing metrics, with capped recall added beside them.

## Verification benchmark

`data/evaluation/v2/verification_cases.json` holds 60 cases, split evenly between `dev` and `test`. Each case has an answer, its evidence passages and a `supported` label. Evidence comes from the v2 corpus.

| Expected result | Category |
|---|---|
| Supported | Verbatim claim |
| Supported | Paraphrased claim |
| Supported | Multi-sentence answer citing different passages |
| Supported | Sentence containing an abbreviation or decimal |
| Unsupported | Negated claim |
| Unsupported | Wrong number |
| Unsupported | Wrong entity |
| Unsupported | Made-up sentence with no citation |
| Unsupported | Citation pointing at an unrelated passage |

Cases run through `EvidenceVerifier.verify_grounding`. The headline metrics are the false accept rate (unsupported answers that passed) and the false reject rate (supported answers that were refused), with accuracy per category.

## Fixes

### Retrieval

- **Chunk IDs:** paged documents become `{source}:p{page}:{n}`. Other files keep `{source}:{n}`, so v1 labels stay valid. The index `VERSION` goes up so old indexes rebuild.
- **Production index:** `collect_source_paths` stops indexing the evaluation corpus, so answers come only from documents the user can see.
- **BM25 tokenizer:** remove stopwords and apply a light stem to alphabetic tokens. Tokens containing digits are left alone.
- **Chunker:** add sentence-aware chunking that respects paragraph breaks. Word chunking stays for v1. Sizes tried: 120 and 200 words. A chunk that crosses a PDF page break is not supported, because each chunk keeps one page number.
- **Candidate count:** try 10, 20 and 40 reranker candidates.
- **Models:** compare `BAAI/bge-small-en-v1.5` for embeddings and `cross-encoder/ms-marco-MiniLM-L-12-v2` for reranking against the current models. Downloads are approved separately once their sizes are known.

### Verifier

- A shared sentence splitter handles citations after punctuation, abbreviations and decimals.
- Every factual sentence needs a citation. An uncited sentence makes the answer unsupported.
- The lexical fallback never overrides an NLI contradiction. It also requires every number in the claim to appear in the evidence, and the claim and evidence to agree on negation.

## Tuning rules

1. Build the benchmark and record the shipped code's results on v1, dev and test before changing any retrieval or verifier code.
2. Tune on `dev` only. A tuning change stays if it raises dev nDCG@5 by at least 0.01 and median retrieval stays under 3 seconds on CPU. Bug fixes stay regardless.
3. Freeze the configuration, then run `test` once for the final report. If anything changes after that, the report says so and states the new run count.
4. Verifier thresholds follow the same rules, using the verification `dev` cases.

## Script and reports

`backend/scripts/benchmark.py` replaces `run_benchmark.py` and `run_reranker_benchmark.py`:

```bash
python -m scripts.benchmark retrieval --set v1
python -m scripts.benchmark retrieval --set dev
python -m scripts.benchmark retrieval --set test --final
python -m scripts.benchmark verification --set dev
python -m scripts.benchmark verification --set test --final
```

`test` refuses to run without `--final`. Reports go to `data/evaluation/reports/` as JSON with the git commit, configuration and per-question results. `data/evaluation/results/` is not used because it is gitignored.

## Testing

- Unit tests come first for each fix: chunk ID uniqueness across pages, the sentence splitter, uncited sentences, contradiction against the fallback, number and negation checks, and tokenizer behaviour on identifiers.
- Unit tests for evidence matching, capped recall, the bootstrap and the `--final` guard.
- The existing suite stays green, except for tests that assert the old buggy behaviour, which are updated and listed in the commit message.

## Done when

- v1 reproduces first, then stays at or above its current scores.
- v1, dev and test reports exist for both the shipped code and the final configuration.
- The README shows both results, with confidence intervals, and explains what the benchmark does not cover.

## Risks

- **Label bias:** questions written by the person building the system can favour it. Every question and label needs a read-through before the first baseline run, and the split must not change after that.
- **Score drop:** the v2 baseline will likely be well below 0.97. That is the point of the set, and the before and after comparison is what matters.
- **Sample size:** 40 test questions give wide intervals, so small differences must be reported as inconclusive.
