from scripts.benchmark_inference import percentile, summarize_latencies


def test_percentile_uses_linear_interpolation():
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([10, 20, 30, 40], 95) == 38.5


def test_summarize_latencies_returns_core_metrics():
    summary = summarize_latencies([10, 20, 30, 40], elapsed_seconds=2)

    assert summary["count"] == 4
    assert summary["avg_ms"] == 25
    assert summary["p50_ms"] == 25
    assert summary["p95_ms"] == 38.5
    assert summary["throughput_rps"] == 2
