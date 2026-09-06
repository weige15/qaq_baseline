"""Create the auditable CPU/tiny-Qwen prerequisite gate for GPU smoke runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from qaq.evaluation import sha256, write_json
from qaq.on_demand import manifest_sha256, study_source_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/on-demand-v1/cpu-gate")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CPU gate requires CUDA_VISIBLE_DEVICES='' explicitly")
        protocol = json.loads(Path("configs/on_demand_protocol.json").read_text())
        inherited = protocol["inherits"]
        protected = {
            inherited["checkpoint"]: inherited["checkpoint_sha256"],
            inherited["examples"]: inherited["examples_sha256"],
            inherited["routes_repeat1"]: inherited["routes_repeat1_sha256"],
            inherited["routes_repeat2"]: inherited["routes_repeat2_sha256"],
            inherited["core_samples"]: inherited["core_samples_sha256"],
            inherited["integration_gate"]: inherited["integration_gate_sha256"],
            inherited["comparison_gate"]: inherited["comparison_gate_sha256"],
            inherited["router_checkpoint"]: inherited["router_checkpoint_sha256"],
            inherited["router_selection"]: inherited["router_selection_sha256"],
        }
        for name, expected in protected.items():
            if sha256(name) != expected:
                raise RuntimeError(f"protected input changed: {name}")
        manifest = study_source_manifest()
        write_json(out / "source_manifest.json", manifest)
        commands = [
            [sys.executable, "-m", "unittest", "tests.test_on_demand", "-v"],
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        ]
        records = []
        env = dict(os.environ, PYTHONPATH="src", CUDA_VISIBLE_DEVICES="")
        for index, command in enumerate(commands, 1):
            run = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            log = out / f"tests-{index}.log"
            log.write_text(run.stdout)
            records.append({"argv": command, "returncode": run.returncode,
                            "log": str(log), "log_sha256": sha256(log)})
            if run.returncode:
                raise RuntimeError(f"CPU test command {index} failed")
        if study_source_manifest() != manifest:
            raise RuntimeError("study source changed during CPU gate")
        result = {
            "stage": "cpu_and_tiny_qwen",
            "passed": True,
            "started_unix": started,
            "finished_unix": time.time(),
            "command": {"argv": sys.argv, "executable": sys.executable,
                        "cwd": os.getcwd(), "cuda_visible_devices": ""},
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True).splitlines(),
            "protocol_sha256": sha256("configs/on_demand_protocol.json"),
            "source_manifest_sha256": manifest_sha256(manifest),
            "protected_input_hashes": protected,
            "tests": records,
        }
        write_json(out / "results.json", result)
        print(json.dumps(result, indent=2))
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
