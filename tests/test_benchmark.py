import json
import os
import subprocess
import sys
from pathlib import Path


def test_benchmark_emits_reproducible_schema(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark-results.json"
    environment = os.environ.copy()
    environment["HERMES_HOME"] = str(tmp_path / "hermes-home")
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_cross_validation.py"),
            "--seed",
            "42",
            "--samples",
            "8",
            "--warmup",
            "1",
            "--repetitions",
            "3",
            "--output",
            str(output),
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["hakua_memory_version"] == "0.3.3"
    assert len(metadata["git_commit_sha"]) == 40
    assert metadata["seed"] == 42
    assert metadata["number_of_samples"] == 8
    assert metadata["warmup_count"] == 1
    assert metadata["measurement_repetitions"] == 3
    for measurement in payload["measurements"].values():
        assert {"mean", "median", "stdev", "p50", "p95", "p99"} <= measurement.keys()
