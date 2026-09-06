"""Audit exact paired outputs, transfer accounting, and memory for the new study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qaq.evaluation import sha256, write_json
from qaq.on_demand import manifest_sha256, study_source_manifest

ROOT = Path("results/on-demand-v1")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_run(path: Path, mode: str, count: int, smoke: bool) -> dict:
    result = json.loads((path / "results.json").read_text())
    command = json.loads((path / "command.json").read_text())
    hardware = json.loads((path / "hardware.json").read_text())
    source_manifest = json.loads((path / "source_manifest.json").read_text())
    samples = jsonl(path / "samples.jsonl")
    profiles = json.loads((path / "profiles.json").read_text())
    events = jsonl(path / "transfer_events.jsonl")
    assert result["protocol_id"] == "qaq-on-demand-v1"
    assert result["mode"] == command["args"]["mode"] == mode
    assert command["args"]["out"] == str(path)
    assert result["smoke_only"] == command["args"]["smoke"] == smoke
    assert result["sample_count"] == len(samples) == len(profiles) == count
    source_hash = manifest_sha256(source_manifest)
    assert study_source_manifest(path / "source") == source_manifest
    assert result["source_manifest_sha256"] == command["source_manifest_sha256"] == source_hash
    assert result["core_samples_exact"]
    assert result["samples_sha256"] == sha256(path / "samples.jsonl")
    assert result["profiles_sha256"] == sha256(path / "profiles.json")
    assert result["transfer_events_sha256"] == sha256(path / "transfer_events.jsonl")
    assert all(sample["task"] == profile["task"] == "wikitext2"
               and sample["index"] == profile["index"]
               and sample["profile"] == profile["profile"]
               for sample, profile in zip(samples, profiles))
    expected_blocks = list(range(72)) * count
    assert [event["block"] for event in events] == expected_blocks
    expected_bits = [bits for profile in profiles for bits in profile["profile"]]
    assert [event["bits"] for event in events] == expected_bits
    assert all(event["released"] for event in events)
    transfer = result["transfer"]
    assert len(events) == transfer["requests"] == transfer["view_closes"] == 72 * count
    assert sum(event["logical_plane_bits"] for event in events) == transfer["requested_plane_bits"]
    assert sum(event["requested_bytes"] for event in events) == transfer["requested_bytes"]
    assert sum(event["copied_bytes"] for event in events) == transfer["copied_bytes"]
    assert abs(sum(event["transfer_seconds"] for event in events)
               - transfer["transfer_seconds"]) < 1e-9
    assert transfer["max_active_slots"] == 1 and transfer["active_slots_at_end"] == 0
    storage = result["storage"]
    assert storage["active_slot_bytes"] == 0
    if mode == "resident":
        assert transfer["loads"] == transfer["releases"] == transfer["copied_bytes"] == 0
        assert storage["source_gpu_bytes"] == storage["source_storage_bytes"]
        assert storage["slot_capacity_bytes"] == 0
        assert result["startup"]["storage_h2d_copied_bytes"] == storage["source_storage_bytes"]
    else:
        assert transfer["loads"] == transfer["releases"] == transfer["requests"]
        assert transfer["copied_bytes"] > 0
        assert storage["source_gpu_bytes"] == 0 and storage["source_devices"] == ["cpu"]
        assert storage["slot_capacity_bytes"] > 0
        assert result["startup"]["storage_h2d_copied_bytes"] == 0
    return {"result": result, "command": command, "hardware": hardware,
            "source_manifest": source_manifest, "source_manifest_sha256": source_hash,
            "samples": samples, "profiles": profiles, "events": events, "path": str(path)}


def check_pair(resident_path: Path, ondemand_path: Path, count: int, smoke: bool) -> dict:
    resident = load_run(resident_path, "resident", count, smoke)
    ondemand = load_run(ondemand_path, "ondemand_sync", count, smoke)
    assert resident["samples"] == ondemand["samples"]
    assert resident["profiles"] == ondemand["profiles"]
    assert resident["result"]["metrics"] == ondemand["result"]["metrics"]
    assert resident["result"]["model_gpu_inventory"] == ondemand["result"]["model_gpu_inventory"]
    assert resident["source_manifest"] == ondemand["source_manifest"]
    for key in ("gpu", "properties", "cuda_runtime", "platform", "cpu"):
        assert resident["hardware"][key] == ondemand["hardware"][key], key
    assert resident["command"]["protocol_sha256"] == ondemand["command"]["protocol_sha256"]
    assert resident["command"]["input_hashes"] == ondemand["command"]["input_hashes"]
    assert resident["command"]["selected_profiles_sha256"] == ondemand["command"]["selected_profiles_sha256"]
    assert [event["requested_bytes"] for event in resident["events"]] == [
        event["requested_bytes"] for event in ondemand["events"]]
    resident_peak = resident["result"]["measurement"]["peak_allocated_bytes"]
    ondemand_peak = ondemand["result"]["measurement"]["peak_allocated_bytes"]
    return {
        "resident": str(resident_path),
        "ondemand_sync": str(ondemand_path),
        "outputs_exact": True,
        "profiles_exact": True,
        "source_manifest_sha256": resident["source_manifest_sha256"],
        "hardware": {key: resident["hardware"][key]
                     for key in ("gpu", "properties", "cuda_runtime", "platform", "cpu")},
        "metrics_exact": True,
        "sample_count": count,
        "requested_bytes": resident["result"]["transfer"]["requested_bytes"],
        "ondemand_copied_bytes": ondemand["result"]["transfer"]["copied_bytes"],
        "ondemand_loads": ondemand["result"]["transfer"]["loads"],
        "ondemand_releases": ondemand["result"]["transfer"]["releases"],
        "resident_peak_allocated_bytes": resident_peak,
        "ondemand_peak_allocated_bytes": ondemand_peak,
        "allocated_reduction_bytes": resident_peak - ondemand_peak,
        "ondemand_peak_is_lower": ondemand_peak < resident_peak,
        "resident_peak_reserved_bytes": resident["result"]["measurement"]["peak_reserved_bytes"],
        "ondemand_peak_reserved_bytes": ondemand["result"]["measurement"]["peak_reserved_bytes"],
        "resident_wall_seconds": resident["result"]["measurement"]["synchronized_wall_seconds"],
        "ondemand_wall_seconds": ondemand["result"]["measurement"]["synchronized_wall_seconds"],
        "ondemand_transfer_seconds": ondemand["result"]["transfer"]["transfer_seconds"],
        "resident_startup_seconds": resident["result"]["startup"]["seconds"],
        "ondemand_startup_seconds": ondemand["result"]["startup"]["seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["smoke", "repeats"])
    parser.add_argument("--resident", default=str(ROOT / "smoke-resident"))
    parser.add_argument("--ondemand", default=str(ROOT / "smoke-ondemand_sync"))
    parser.add_argument("--gate-out", default=str(ROOT / "smoke-gate.json"))
    parser.add_argument("--smoke-gate", default=str(ROOT / "smoke-gate.json"))
    args = parser.parse_args()
    if args.stage == "smoke":
        resident_path, ondemand_path = Path(args.resident), Path(args.ondemand)
        pair = check_pair(resident_path, ondemand_path, 1, True)
        resident = json.loads((resident_path / "results.json").read_text())
        ondemand = json.loads((ondemand_path / "results.json").read_text())
        full_count = json.loads(Path("configs/on_demand_protocol.json").read_text())["evaluation"]["repeat_examples"]
        projections = {
            "resident_seconds": resident["startup"]["seconds"]
                + full_count * resident["measurement"]["synchronized_wall_seconds"],
            "ondemand_sync_seconds": ondemand["startup"]["seconds"]
                + full_count * ondemand["measurement"]["synchronized_wall_seconds"],
            "formula": "fresh-run startup + 64 * one-window synchronized wall",
            "limit_seconds": 1800,
        }
        repeat_authorized = (pair["outputs_exact"] and pair["ondemand_peak_is_lower"]
                             and max(projections["resident_seconds"],
                                     projections["ondemand_sync_seconds"])
                             <= projections["limit_seconds"])
        gate = {"stage": "full_model_smoke", "passed": repeat_authorized,
                "repeat_authorized": repeat_authorized,
                "source_manifest_sha256": pair["source_manifest_sha256"], "pair": pair,
                "projected_full_jobs": projections,
                "stop_reason": None if repeat_authorized else (
                    "output mismatch, no allocated-memory reduction, or projected job above 30 minutes")}
        write_json(args.gate_out, gate)
        print(json.dumps(gate, indent=2))
        return

    smoke = json.loads(Path(args.smoke_gate).read_text())
    assert smoke["passed"] and smoke["repeat_authorized"]
    accepted_ids = ("1b", "2b", "3")
    pairs = [check_pair(ROOT / f"repeat-{run_id}-resident",
                        ROOT / f"repeat-{run_id}-ondemand_sync", 64, False)
             for run_id in accepted_ids]
    assert all(pair["source_manifest_sha256"] == smoke["source_manifest_sha256"]
               for pair in pairs)
    resident_samples = [jsonl(ROOT / f"repeat-{run_id}-resident/samples.jsonl")
                        for run_id in accepted_ids]
    ondemand_samples = [jsonl(ROOT / f"repeat-{run_id}-ondemand_sync/samples.jsonl")
                        for run_id in accepted_ids]
    assert resident_samples[0] == resident_samples[1] == resident_samples[2]
    assert ondemand_samples[0] == ondemand_samples[1] == ondemand_samples[2]
    protocol = json.loads(Path("configs/on_demand_protocol.json").read_text())
    core_samples = Path(protocol["inherits"]["core_samples"])
    assert sha256(core_samples) == protocol["inherits"]["core_samples_sha256"]
    expected = [row for row in jsonl(core_samples) if row["task"] == "wikitext2"]
    assert resident_samples[0] == ondemand_samples[0] == expected
    passed = all(pair["outputs_exact"] and pair["ondemand_peak_is_lower"] for pair in pairs)
    commands = {}
    for repeat_id, run_id in enumerate(accepted_ids, 1):
        for mode in ("resident", "ondemand_sync"):
            path = ROOT / f"repeat-{run_id}-{mode}" / "command.json"
            commands[(repeat_id, mode)] = json.loads(path.read_text())
    run_ids = [command["run_uuid"] for command in commands.values()]
    starts = [command["started_unix"] for command in commands.values()]
    assert len(set(run_ids)) == len(run_ids) == 6 and len(set(starts)) == 6
    assert commands[(1, "resident")]["started_unix"] < commands[(1, "ondemand_sync")]["started_unix"]
    assert commands[(2, "ondemand_sync")]["started_unix"] < commands[(2, "resident")]["started_unix"]
    assert commands[(3, "resident")]["started_unix"] < commands[(3, "ondemand_sync")]["started_unix"]
    rejected = []
    for run_id in ("1", "2"):
        resident_hardware = json.loads((ROOT / f"repeat-{run_id}-resident/hardware.json").read_text())
        ondemand_hardware = json.loads((ROOT / f"repeat-{run_id}-ondemand_sync/hardware.json").read_text())
        assert resident_hardware["properties"] != ondemand_hardware["properties"]
        rejected.append({"candidate": run_id, "reason": "physical_gpu_identity_mismatch",
                         "resident_properties": resident_hardware["properties"],
                         "ondemand_properties": ondemand_hardware["properties"]})
    gate = {
        "stage": "three_paired_wikitext2_repeats",
        "passed": passed,
        "pairs": pairs,
        "accepted_directory_ids": list(accepted_ids),
        "rejected_pairing_candidates": rejected,
        "cross_repeat_samples_exact": True,
        "exact_to_completed_core_wikitext_samples": True,
        "distinct_fresh_run_ids": True,
        "predeclared_pair_order_verified": True,
    }
    write_json(ROOT / "repeat-gate.json", gate)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
