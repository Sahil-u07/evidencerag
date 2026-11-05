import json
from types import SimpleNamespace

import pytest

from scripts import benchmark


def test_test_split_requires_final():
    with pytest.raises(SystemExit):
        benchmark.main(["retrieval", "--set", "test"])


def test_verification_test_split_requires_final():
    with pytest.raises(SystemExit):
        benchmark.main(["verification", "--set", "test"])


def test_final_rejects_config_overrides():
    with pytest.raises(SystemExit):
        benchmark.main(["retrieval", "--set", "test", "--final", "--candidates", "10"])


def test_summarize_means_and_intervals():
    per_query = [
        {"id": "a", "type": "paraphrase", "latency_ms": 10.0, "mrr": 1.0},
        {"id": "b", "type": "identifier", "latency_ms": 30.0, "mrr": 0.0},
    ]
    summary = benchmark.summarize(per_query)
    assert summary["overall"]["mrr"]["mean"] == 0.5
    assert summary["overall"]["mrr"]["ci95"][0] <= 0.5 <= summary["overall"]["mrr"]["ci95"][1]
    assert summary["by_type"]["paraphrase"]["mrr"]["mean"] == 1.0
    assert summary["latency_ms"]["median"] == 20.0


def test_run_verification_counts_false_accepts_and_rejects(tmp_path, monkeypatch):
    cases = [
        {"id": "1", "split": "dev", "category": "verbatim", "supported": True, "answer": "ok", "evidence": ["p"]},
        {"id": "2", "split": "dev", "category": "negation", "supported": False, "answer": "bad", "evidence": ["p"]},
        {"id": "3", "split": "test", "category": "negation", "supported": False, "answer": "bad", "evidence": ["p"]},
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    monkeypatch.setattr(benchmark, "CASES_PATH", path)
    monkeypatch.setattr(benchmark, "read_corpus_text", lambda: "p")

    fake = SimpleNamespace(
        verify_grounding=lambda answer, evidence: SimpleNamespace(supported=True, reason="x")
    )
    result = benchmark.run_verification("dev", verifier=fake)

    assert result["false_accept_rate"] == 1.0
    assert result["false_reject_rate"] == 0.0
    assert result["accuracy"] == 0.5
    assert result["by_category"]["negation"]["accuracy"] == 0.0
    assert len(result["per_case"]) == 2


def test_report_counts_repeat_test_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "REPORTS_DIR", tmp_path)
    config = benchmark.RetrievalConfig()
    first = benchmark.write_report("final", "retrieval", "test", config, {})
    benchmark.write_report("final", "retrieval", "test", config, {})
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["run_count"] == 2
    assert report["config"]["candidate_k"] == config.candidate_k
    assert "git_commit" in report


def test_v1_report_records_the_pinned_config(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(benchmark, "run_v1", lambda config: {"ran_with": config.chunk_size})
    monkeypatch.setattr(benchmark, "print_retrieval", lambda result: None)

    benchmark.main(["retrieval", "--set", "v1", "--chunk-size", "100", "--label", "x"])

    report = json.loads((tmp_path / "x-retrieval-v1.json").read_text(encoding="utf-8"))
    assert report["config"]["chunking"] == "words"
    assert report["config"]["chunk_size"] == 50
    assert report["config"]["chunk_overlap"] == 10
