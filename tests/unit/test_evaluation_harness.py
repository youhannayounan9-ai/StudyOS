"""Unit tests for StudyOS Stage 7 Deterministic Evaluation Benchmark Harness."""

from scripts.evaluate import run_evaluation


def test_evaluation_benchmark_executes_offline_and_passes():
    """Verify that the evaluation benchmark runs completely offline and produces PASS status."""
    report = run_evaluation(live_model=False, verbose=False)
    assert report["benchmark_status"] == "PASS"
    assert report["duration_seconds"] > 0

    metrics = report["metrics"]
    # Check retrieval recall metrics
    assert metrics["retrieval"]["top1_accuracy"] == 1.0
    assert metrics["retrieval"]["recall_at_3_rate"] == 1.0
    assert metrics["retrieval"]["recall_at_5_rate"] == 1.0
    assert metrics["retrieval"]["mrr"] == 1.0

    # Check citation validation
    assert metrics["citations"]["passed"] is True

    # Check abstention
    assert metrics["abstention"]["passed"] is True

    # Check answer key isolation
    assert metrics["security_isolation"]["passed"] is True

    # Check learner & scheduler integrity
    assert metrics["learner_scheduler"]["passed"] is True

    # Check concept graph constraints
    assert metrics["concept_graph"]["passed"] is True

