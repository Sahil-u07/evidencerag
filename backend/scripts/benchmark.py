import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

import numpy as np

from app.core import retrieval_defaults as defaults
from app.evaluation.evidence import (
    bootstrap_ci,
    relevant_chunk_count,
    score_question,
    validate_cases,
    validate_questions,
)
from app.evaluation.metrics import (
    capped_recall_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.reranking.cross_encoder import CrossEncoderReranker
from app.reranking.reranked_hybrid import RerankedHybridRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedder import TextEmbedder
from app.retrieval.hybrid import HybridRetriever


ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "data" / "evaluation"
V1_CORPUS = EVALUATION / "corpus"
V1_GOLDEN = EVALUATION / "golden_set.json"
V2 = EVALUATION / "v2"
V2_CORPUS = V2 / "corpus"
QUESTIONS_PATH = V2 / "questions.json"
CASES_PATH = V2 / "verification_cases.json"
CACHE_DIR = V2 / ".cache"
REPORTS_DIR = EVALUATION / "reports"

TOP_K = 5
KS = (1, 3, 5)


@dataclass(frozen=True)
class RetrievalConfig:
    chunking: str = defaults.CHUNKING
    chunk_size: int = defaults.CHUNK_SIZE
    chunk_overlap: int = defaults.CHUNK_OVERLAP
    candidate_k: int = defaults.CANDIDATE_K
    embedding_model: str = defaults.EMBEDDING_MODEL
    reranker_model: str = defaults.RERANKER_MODEL


V1_CONFIG_FIELDS = {"chunking": "words", "chunk_size": 50, "chunk_overlap": 10}


def v1_config(config: RetrievalConfig) -> RetrievalConfig:
    return RetrievalConfig(**{**asdict(config), **V1_CONFIG_FIELDS})


def chunk_corpus(paths: list[Path], config: RetrievalConfig) -> list:
    documents = [document for path in paths for document in load_document(path)]
    return chunk_documents(documents, chunk_size=config.chunk_size, overlap=config.chunk_overlap)


def v2_corpus_paths() -> list[Path]:
    return sorted(
        path for path in V2_CORPUS.iterdir() if path.suffix.lower() in {".pdf", ".txt"}
    )


def read_corpus_text() -> str:
    return "\n".join(
        document.text for path in v2_corpus_paths() for document in load_document(path)
    )


def embed_chunks(chunks: list, embedder: TextEmbedder, model_name: str) -> np.ndarray:
    # Re-embedding the RFCs takes minutes, and sweeps reuse the same chunks.
    digest = hashlib.sha256(model_name.encode())
    for chunk in chunks:
        digest.update(chunk.text.encode())

    cache_path = CACHE_DIR / f"{digest.hexdigest()[:24]}.npy"
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=False)

    embeddings = np.asarray(embedder.encode([chunk.text for chunk in chunks]), dtype=np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def build_systems(chunks: list, config: RetrievalConfig) -> dict:
    embedder = TextEmbedder(config.embedding_model)
    embeddings = embed_chunks(chunks, embedder, config.embedding_model)

    dense = DenseRetriever()
    dense.add(chunks, embeddings)
    bm25 = BM25Retriever()
    bm25.add(chunks)
    hybrid = HybridRetriever(embedder=embedder, dense_retriever=dense, bm25_retriever=bm25)
    reranked = RerankedHybridRetriever(hybrid, CrossEncoderReranker(config.reranker_model))

    return {
        "dense": lambda query: [
            item.chunk for item in dense.search(embedder.encode_query(query), top_k=TOP_K)
        ],
        "bm25": lambda query: [item.chunk for item in bm25.search(query, top_k=TOP_K)],
        "hybrid": lambda query: [item.chunk for item in hybrid.search(query, top_k=TOP_K)],
        "reranked_hybrid": lambda query: [
            item.result.chunk
            for item in reranked.search(query, top_k=TOP_K, candidate_k=config.candidate_k)
        ],
    }


def summarize(per_query: list[dict]) -> dict:
    def aggregate(rows: list[dict]) -> dict:
        metrics = [key for key in rows[0] if key not in {"id", "type", "latency_ms", "retrieved"}]
        return {
            metric: {
                "mean": mean(row[metric] for row in rows),
                "ci95": bootstrap_ci([row[metric] for row in rows]),
            }
            for metric in metrics
        }

    types = sorted({row["type"] for row in per_query})
    latencies = [row["latency_ms"] for row in per_query]

    return {
        "overall": aggregate(per_query),
        "by_type": {
            name: aggregate([row for row in per_query if row["type"] == name]) for name in types
        },
        "latency_ms": {
            "median": median(latencies),
            "p95": float(np.percentile(latencies, 95)),
        },
    }


def run_v1(config: RetrievalConfig) -> dict:
    config = v1_config(config)
    chunks = chunk_corpus(sorted(V1_CORPUS.glob("*.md")), config)
    golden = json.loads(V1_GOLDEN.read_text(encoding="utf-8"))
    results = {}

    for name, search in build_systems(chunks, config).items():
        per_query = []
        for item in golden:
            started = time.perf_counter()
            ids = [chunk.chunk_id for chunk in search(item["query"])]
            relevant = item["relevant_ids"]
            row = {
                "id": item["id"],
                "type": "v1",
                "latency_ms": (time.perf_counter() - started) * 1000,
                "mrr": reciprocal_rank(ids, relevant),
                "ndcg_at_5": ndcg_at_k(ids, relevant, 5),
                "retrieved": ids,
            }
            for k in KS:
                row[f"recall_at_{k}"] = recall_at_k(ids, relevant, k)
                row[f"capped_recall_at_{k}"] = capped_recall_at_k(ids, relevant, k)
            per_query.append(row)
        results[name] = {"summary": summarize(per_query), "per_query": per_query}

    return {"chunk_count": len(chunks), "question_count": len(golden), "systems": results}


def run_v2(split: str, config: RetrievalConfig) -> dict:
    chunks = chunk_corpus(v2_corpus_paths(), config)
    texts = [chunk.text for chunk in chunks]
    every_question = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = [item for item in every_question if item["split"] == split]

    # The too-common check only applies to the chunking the labels were written against.
    is_label_chunking = (config.chunking, config.chunk_size, config.chunk_overlap) == ("words", 50, 10)
    errors = validate_questions(every_question, texts, max_matches=3 if is_label_chunking else None)
    if errors:
        raise SystemExit("Invalid questions:\n" + "\n".join(errors))

    results = {}
    relevant_counts = {
        item["id"]: relevant_chunk_count(texts, item["evidence"]) for item in questions
    }

    for name, search in build_systems(chunks, config).items():
        per_query = []
        for item in questions:
            started = time.perf_counter()
            retrieved = search(item["query"])
            latency = (time.perf_counter() - started) * 1000
            per_query.append(
                {
                    "id": item["id"],
                    "type": item["type"],
                    "latency_ms": latency,
                    "retrieved": [chunk.chunk_id for chunk in retrieved],
                    **score_question(
                        [chunk.text for chunk in retrieved],
                        item["evidence"],
                        relevant_counts[item["id"]],
                    ),
                }
            )
        results[name] = {"summary": summarize(per_query), "per_query": per_query}

    return {"chunk_count": len(chunks), "question_count": len(questions), "systems": results}


def _as_evidence(passage: str) -> SimpleNamespace:
    chunk = SimpleNamespace(text=passage, source="case", page=None, chunk_id="case")
    return SimpleNamespace(result=SimpleNamespace(chunk=chunk), score=1.0)


def run_verification(split: str, verifier=None) -> dict:
    if verifier is None:
        from app.generation.verifier import EvidenceVerifier

        verifier = EvidenceVerifier()

    every_case = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    errors = validate_cases(every_case, read_corpus_text())
    if errors:
        raise SystemExit("Invalid verification cases:\n" + "\n".join(errors))

    per_case = []
    for case in (item for item in every_case if item["split"] == split):
        result = verifier.verify_grounding(
            case["answer"], [_as_evidence(passage) for passage in case["evidence"]]
        )
        per_case.append(
            {
                "id": case["id"],
                "category": case["category"],
                "expected": case["supported"],
                "predicted": result.supported,
                "reason": result.reason,
            }
        )

    def rate(rows: list[dict]) -> float:
        return sum(row["predicted"] != row["expected"] for row in rows) / len(rows) if rows else 0.0

    categories = sorted({row["category"] for row in per_case})

    return {
        "case_count": len(per_case),
        "accuracy": 1.0 - rate(per_case),
        "false_accept_rate": rate([row for row in per_case if not row["expected"]]),
        "false_reject_rate": rate([row for row in per_case if row["expected"]]),
        "by_category": {
            name: {"accuracy": 1.0 - rate([row for row in per_case if row["category"] == name])}
            for name in categories
        },
        "per_case": per_case,
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def write_report(label: str, kind: str, set_name: str, config: RetrievalConfig, result: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{label}-{kind}-{set_name}.json"
    run_count = 1

    if path.exists():
        run_count = json.loads(path.read_text(encoding="utf-8")).get("run_count", 1) + 1

    report = {
        "git_commit": _git("rev-parse", "HEAD"),
        # Reports written earlier in the same run should not mark the code as dirty.
        "git_dirty": bool(_git("status", "--porcelain", "--", ".", ":(exclude)data/evaluation/reports")),
        "run_count": run_count,
        "config": asdict(config),
        **result,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def print_retrieval(result: dict) -> None:
    header = ["hit_at_1", "hit_at_5", "recall_at_5", "capped_recall_at_5", "mrr", "ndcg_at_5"]
    print("chunks:", result["chunk_count"], "questions:", result["question_count"])
    print(f"{'system':<18}" + "".join(f"{name:>20}" for name in header) + f"{'median ms':>12}")

    for name, system in result["systems"].items():
        overall = system["summary"]["overall"]
        cells = "".join(
            f"{overall[key]['mean']:>20.3f}" if key in overall else f"{'-':>20}" for key in header
        )
        print(f"{name:<18}{cells}{system['summary']['latency_ms']['median']:>12.0f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the EvidenceRAG benchmarks.")
    commands = parser.add_subparsers(dest="kind", required=True)

    retrieval = commands.add_parser("retrieval")
    retrieval.add_argument("--set", dest="set_name", choices=["v1", "dev", "test"], required=True)
    retrieval.add_argument("--chunking", choices=["words", "sentences"])
    retrieval.add_argument("--chunk-size", type=int)
    retrieval.add_argument("--chunk-overlap", type=int)
    retrieval.add_argument("--candidates", type=int, dest="candidate_k")
    retrieval.add_argument("--embedding-model")
    retrieval.add_argument("--reranker-model")

    verification = commands.add_parser("verification")
    verification.add_argument("--set", dest="set_name", choices=["dev", "test"], required=True)

    for command in (retrieval, verification):
        command.add_argument("--final", action="store_true")
        command.add_argument("--label", default="run")

    args = parser.parse_args(argv)

    if args.set_name == "test" and not args.final:
        parser.error("the test split needs --final, and it must not inform tuning")

    overrides = {
        name: getattr(args, name)
        for name in (
            "chunking",
            "chunk_size",
            "chunk_overlap",
            "candidate_k",
            "embedding_model",
            "reranker_model",
        )
        if getattr(args, name, None) is not None
    }

    if args.final and overrides:
        parser.error("--final runs the committed defaults, so it takes no overrides")

    config = RetrievalConfig(**overrides)

    if args.kind == "verification":
        result = run_verification(args.set_name)
        print(json.dumps({key: value for key, value in result.items() if key != "per_case"}, indent=2))
    elif args.set_name == "v1":
        config = v1_config(config)
        result = run_v1(config)
        print_retrieval(result)
    else:
        result = run_v2(args.set_name, config)
        print_retrieval(result)

    print("report:", write_report(args.label, args.kind, args.set_name, config, result))


if __name__ == "__main__":
    main()
