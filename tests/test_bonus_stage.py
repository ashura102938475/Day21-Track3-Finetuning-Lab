import subprocess
import sys


def test_runner_repeats_child_failure_tail():
    result = subprocess.run(
        [sys.executable, "scripts/bonus_stage.py", "--command", sys.executable, "-c",
         "import sys; print('ROOT_CAUSE_MARKER', file=sys.stderr); raise SystemExit(7)"],
        capture_output=True, text=True,
    )
    assert result.returncode == 7
    assert "ROOT_CAUSE_MARKER" in result.stdout
    assert "failure tail" in result.stdout
