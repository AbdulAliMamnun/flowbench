import time

import torch

from flowbench.evaluation.latency import measure_latency


def test_measure_latency_reports_percentiles_in_ms() -> None:
    calls = 0

    def call() -> None:
        nonlocal calls
        calls += 1
        time.sleep(0.001)

    stats = measure_latency(call, torch.device("cpu"), warmup_iterations=3, timed_iterations=10)
    assert calls == 13
    assert stats.warmup_iterations == 3
    assert stats.timed_iterations == 10
    assert stats.device == "cpu"
    assert stats.p50_ms >= 1.0
    assert stats.p95_ms >= stats.p50_ms
    assert stats.min_ms <= stats.mean_ms <= stats.max_ms
    assert set(stats.as_dict()) >= {"p50_ms", "p95_ms", "mean_ms", "device"}
