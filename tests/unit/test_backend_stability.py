from vla_robot_sorting.evaluation.backend_stability import latency_summary, memory_summary


def test_latency_summary_reports_median_and_p95() -> None:
    result = latency_summary([0.4, 0.1, 0.2, 0.3])

    assert result["count"] == 4
    assert result["min_s"] == 0.1
    assert result["median_s"] == 0.25
    assert result["p95_s"] == 0.4
    assert result["max_s"] == 0.4


def test_memory_summary_detects_post_warmup_monotonic_growth() -> None:
    result = memory_summary([100, 200, 300, 310, 320, 330], warmup_ignore=2)

    assert result["tail_net_growth_mib"] == 30.0
    assert result["monotonic_growth_after_warmup"] is True


def test_memory_summary_accepts_plateau_or_noise() -> None:
    result = memory_summary([100, 200, 300, 310, 305, 307], warmup_ignore=2)

    assert result["monotonic_growth_after_warmup"] is False
    assert result["tail_span_mib"] == 10.0
