"""Run the frozen endpoint study once into a NEW append-only analysis directory."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from qaq.query_budget_analysis import analyze, load_pairs, read_json, require, sha256, verify_hashes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/query-budget-feasibility-v1"
SEAL = "results/query-budget-feasibility-v1/protocol-freeze.json"
SEAL_SHA256 = "5a8b1fbabd8a5dd981922c3cdfd10a951cc2c2bb5c31b958837a6e61d9aaa115"
SOURCES = ("QUERY_BUDGET_FEASIBILITY_PROTOCOL.md", "configs/query_budget_feasibility_protocol.json",
           "src/qaq/query_budget_analysis.py", "scripts/analyze_query_budget.py",
           "tests/test_query_budget_analysis.py")


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(OUTPUT) and out != OUTPUT and not out.exists(),
            f"output must be a NEW directory beneath {OUTPUT}: {out}")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES='' (CPU only)")
    require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "set PYTHONDONTWRITEBYTECODE=1 to preserve bytecode")
    verify_hashes(ROOT, {SEAL: SEAL_SHA256})
    seal = read_json(ROOT/SEAL)
    verify_hashes(ROOT, seal["files"])
    cfg = read_json(ROOT/"configs/query_budget_feasibility_protocol.json")
    initial = read_json(OUTPUT/"preflight/initial-state.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head == initial["head"], "HEAD changed since initial record; pause")
    source_hashes = {p: sha256(ROOT/p) for p in SOURCES}
    # Every required hash/repeat/identity is checked before any new outcome aggregate.
    pairs = load_pairs(ROOT, cfg)
    result = analyze(pairs, cfg)
    require(source_hashes == {p: sha256(ROOT/p) for p in SOURCES}, "analysis source changed during run")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/"analysis.json", result)
    write_new(out/"manifest.json", {"protocol_id": cfg["protocol_id"], "git_head": head,
        "protocol_freeze_sha256": SEAL_SHA256, "input_sha256": cfg["input_sha256"],
        "source_sha256": source_hashes, "python": platform.python_version(), "numpy": np.__version__,
        "analysis_sha256": sha256(out/"analysis.json"), "sample_count": len(pairs),
        "baseline_and_route_repeats_exact_except_declared_timing": True,
        "execution": "CPU-only deterministic saved-endpoint analysis; no model executed"})
    print(json.dumps({"decision": result["decision"], "analysis_sha256": sha256(out/"analysis.json")},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
