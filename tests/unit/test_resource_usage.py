from pathlib import Path

from vla_robot_sorting.evaluation.resource_usage import summarize_resource_usage


def test_summarizes_native_resource_files(tmp_path: Path) -> None:
    gpu = tmp_path / "gpu.csv"
    gpu.write_text(
        "2026/10/05 23:18:00, GPU, 10, 2, 512, 6144\n"
        "2026/10/05 23:18:01, GPU, 95, 24, 2279, 6144\n",
        encoding="utf-8",
    )
    timing = tmp_path / "time.txt"
    timing.write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:40.40\n"
        "Maximum resident set size (kbytes): 3784880\n"
        "Exit status: 0\n",
        encoding="utf-8",
    )

    result = summarize_resource_usage(gpu, timing)

    assert result == {
        "gpu_samples": 2,
        "max_gpu_utilization_percent": 95.0,
        "max_gpu_memory_utilization_percent": 24.0,
        "peak_vram_used_mib": 2279.0,
        "vram_total_mib": 6144.0,
        "max_host_rss_kib": 3784880,
        "elapsed_wall_clock": "1:40.40",
        "command_exit_status": 0,
    }
