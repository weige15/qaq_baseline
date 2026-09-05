import os
from pathlib import Path
import subprocess
import tempfile
import unittest


MOCK = '''#!/usr/bin/env bash
set -eu
case "$*" in
  *--query-gpu=index,*) printf '0, mock, 24576, 100, 24000, 0\\n1, mock, 24576, 1000, 23000, 0\\n' ;;
  *--query-compute-apps=pid*)
    if [[ "${FAIL_QUERY:-0}" == 1 ]]; then exit 1; fi
    if [[ "$*" == *--id=0* || "${ALL_BUSY:-0}" == 1 ]]; then echo 999; fi ;;
  *--query-gpu=memory.free*) echo "${RECHECK_FREE:-23000}" ;;
  *--query-gpu=utilization.gpu*) echo 0 ;;
  *--query-compute-apps=gpu_uuid*) echo 'GPU-zero, 999, someone-else, 100' ;;
  *) exit 1 ;;
esac
'''


class PreflightTests(unittest.TestCase):
    def run_guard(self, **extra):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d)/"nvidia-smi"
            exe.write_text(MOCK)
            exe.chmod(0o755)
            env = {**os.environ, "VIRTUAL_ENV": d, "PATH": d + ":" + os.environ["PATH"], **extra}
            return subprocess.run(["bash", "scripts/gpu_preflight.sh", "--run", "bash", "-c",
                                   'echo "LAUNCHED:$CUDA_VISIBLE_DEVICES"'], env=env,
                                  text=True, capture_output=True)

    def test_skips_process_owning_most_empty_gpu(self):
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("LAUNCHED:1", result.stdout)

    def test_fail_closed_on_busy_changed_or_query_failure(self):
        for state in ({"ALL_BUSY": "1"}, {"RECHECK_FREE": "19000"}, {"FAIL_QUERY": "1"}):
            result = self.run_guard(**state)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("LAUNCHED", result.stdout)


if __name__ == "__main__":
    unittest.main()
