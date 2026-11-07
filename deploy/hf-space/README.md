---
title: EvidenceRAG
emoji: 📄
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: mit
pinned: false
---

# EvidenceRAG

EvidenceRAG answers questions from documents and shows the passages behind each answer. It retrieves with BM25 and
dense embeddings, reranks the results, generates with `qwen2.5:3b` through Ollama, and checks the answer against the
evidence. When the evidence isn't there, it says so.

This Space is a public, read-only demo. It holds six RFCs (TCP, JWT, DNS, HTTP semantics, TLS 1.3 and OAuth 2.0), and
uploads and deletes are turned off. Try "What does the iss claim in a JWT mean?".

The model runs on a shared CPU, so an answer can take a minute. The demo handles one question at a time. If it is
busy, wait a minute and ask again. Questions are limited to 500 characters.

To use your own documents, run it locally: https://github.com/Sahil-u07/evidencerag

The model is Qwen2.5-3B, released under the Qwen Research License (non-commercial use). The licence text is in the
image at `/app/licenses/qwen2.5-3b-LICENSE.txt`. The RFCs are unmodified copies from the RFC Editor.
