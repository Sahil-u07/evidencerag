import json
from types import SimpleNamespace

import pytest

from scripts import benchmark


@pytest.fixture
def no_benchmark_runs(monkeypatch):
    # A broken guard must fail the test, not spend a held-out run.
    def refuse(*args, **kwargs):
        raise AssertionError("a benchmark ran")

    for name in ("run_v1", "run_v2", "run_verification"):
        monkeypatch.setattr(benchmark, name, refuse)


def test_test_split_requires_final(no_benchmark_runs):
    with pytest.raises(SystemExit):
        benchmark.main(["retrieval", "--set", "test"])


def test_verification_test_split_requires_final(no_benchmark_runs):
    with pytest.raises(SystemExit):
        benchmark.main(["verification", "--set", "test"])


def test_final_rejects_config_overrides(no_benchmark_runs):
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
        verify_grounding=lambda answer, evidence: SimpleNamespace(supported=True, reason="x"),
        model_name="nli-model",
        entailment_threshold=0.7,
    )
    result = benchmark.run_verification("dev", verifier=fake)

    assert result["false_accept_rate"] == 1.0
    assert result["false_reject_rate"] == 0.0
    assert result["accuracy"] == 0.5
    assert result["by_category"]["negation"]["accuracy"] == 0.0
    assert len(result["per_case"]) == 2
    assert result["false_accepts"] == 1
    assert result["unsupported_cases"] == 1
    assert result["false_rejects"] == 0
    assert result["supported_cases"] == 1
    assert result["false_accept_rate_ci95"] == (1.0, 1.0)
    assert result["false_reject_rate_ci95"] == (0.0, 0.0)
    low, high = result["accuracy_ci95"]
    assert low <= 0.5 <= high
    assert result["nli_model"] == "nli-model"
    assert result["entailment_threshold"] == 0.7


def test_run_verification_interval_is_none_without_cases_of_a_kind(tmp_path, monkeypatch):
    cases = [
        {"id": "1", "split": "dev", "category": "verbatim", "supported": True, "answer": "ok", "evidence": ["p"]},
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    monkeypatch.setattr(benchmark, "CASES_PATH", path)
    monkeypatch.setattr(benchmark, "read_corpus_text", lambda: "p")
    fake = SimpleNamespace(
        verify_grounding=lambda answer, evidence: SimpleNamespace(supported=True, reason="x"),
        model_name="m",
        entailment_threshold=0.7,
    )

    result = benchmark.run_verification("dev", verifier=fake)

    assert result["unsupported_cases"] == 0
    assert result["false_accept_rate_ci95"] is None


def test_print_retrieval_shows_the_three_passage_columns(capsys):
    overall = {"hit_at_3": {"mean": 0.8}, "capped_recall_at_3": {"mean": 0.6}}
    system = {"summary": {"overall": overall, "latency_ms": {"median": 5.0}}}

    benchmark.print_retrieval({"chunk_count": 1, "question_count": 1, "systems": {"dense": system}})

    output = capsys.readouterr().out
    assert "hit_at_3" in output
    assert "capped_recall_at_3" in output
    assert "0.800" in output
    assert "0.600" in output


def test_report_counts_repeat_test_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "REPORTS_DIR", tmp_path)
    config = benchmark.RetrievalConfig()
    first = benchmark.write_report("final", "retrieval", "test", config, {}, final=True)
    benchmark.write_report("final", "retrieval", "test", config, {}, final=True)
    report = json.loads(first.read_text(encoding="utf-8"))
    assert report["run_count"] == 2
    assert report["config"]["candidate_k"] == config.candidate_k
    assert report["set"] == "test"
    assert report["final"] is True
    assert "git_commit" in report


def test_verification_report_records_the_model_and_no_retrieval_config(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "REPORTS_DIR", tmp_path)
    result = {"nli_model": "nli-model", "entailment_threshold": 0.7}

    path = benchmark.write_report("x", "verification", "dev", benchmark.RetrievalConfig(), result)

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["nli_model"] == "nli-model"
    assert report["entailment_threshold"] == 0.7
    assert report["set"] == "dev"
    assert report["final"] is False
    assert "config" not in report


def test_v1_report_records_the_pinned_config(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(benchmark, "run_v1", lambda config: {"ran_with": config.chunk_size})
    monkeypatch.setattr(benchmark, "print_retrieval", lambda result: None)

    benchmark.main(["retrieval", "--set", "v1", "--chunk-size", "100", "--label", "x"])

    report = json.loads((tmp_path / "x-retrieval-v1.json").read_text(encoding="utf-8"))
    assert report["config"]["chunking"] == "words"
    assert report["config"]["chunk_size"] == 50
    assert report["config"]["chunk_overlap"] == 10
    assert report["set"] == "v1"
    assert report["final"] is False
