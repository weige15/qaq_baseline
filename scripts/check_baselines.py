"""Verify ONLY the fresh baseline stage; never claims whole-goal completion."""
import argparse
import json
from pathlib import Path

from qaq.evaluation import sha256, summarize, write_json


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check(root):
    frozen = root / "frozen"
    hashes = json.loads((frozen / "freeze_hashes.json").read_text())
    for name, digest in hashes.items():
        assert sha256(frozen / name) == digest, f"frozen file changed: {name}"
    config = json.loads((frozen / "protocol.json").read_text())
    examples = load_jsonl(frozen / "examples.jsonl")
    keys = [(e["task"], e["index"]) for e in examples]
    assert len(keys) == len(set(keys)) == 576
    results, samples, repeat = {}, {}, {}
    for mode in ("fp16", "fixed8", "fixed4"):
        for r in (1, 2):
            directory = root / f"baseline-{mode}-r{r}"
            result = json.loads((directory / "results.json").read_text())
            command = json.loads((directory / "command.json").read_text())
            rows = load_jsonl(directory / "samples.jsonl")
            assert not result["smoke_only"] and result["mode"] == mode
            assert command["frozen_hashes"] == hashes
            assert result["sample_count"] == len(examples)
            assert [(s["task"], s["index"]) for s in rows] == keys
            assert not (directory / "failure.txt").exists()
            for row, ex in zip(rows, examples):
                if ex["task"] == "wikitext2":
                    assert row["scored_tokens"] == len(ex["tokens"]) - ex["prefix_length"]
                    assert row["scored_tokens"] == len(row["token_nll"])
                    assert row["nll_sum"] == sum(row["token_nll"])
                else:
                    assert row["gold"] == ex["gold"]
                    assert row["loglikelihoods"] == [-sum(ls) for ls in row["token_nll"]]
                    assert row["normalized_loglikelihoods"] == [
                        ll/len(choice) for ll, choice in zip(row["loglikelihoods"], ex["choices"])]
                    assert [len(ls) for ls in row["token_nll"]] == [
                        len(t)-n for t, n in ex["encoded_choices"]]
                    for pred, score in (("prediction", "loglikelihoods"),
                                        ("prediction_norm", "normalized_loglikelihoods")):
                        assert row[pred] == max(range(len(ex["choices"])), key=row[score].__getitem__)
            assert summarize(rows) == result["metrics"], f"raw metric mismatch: {directory}"
            # allow_nan=False rejects NaN/Inf anywhere, including token losses.
            json.dumps(result, allow_nan=False)
            json.dumps(rows, allow_nan=False)
            for task, metrics in result["metrics"].items():
                required = config["evaluation"]["wikitext_metrics" if task == "wikitext2"
                                                  else "multiple_choice_metrics"]
                assert set(metrics) == set(required), f"missing/extra metric: {mode}/{task}"
            if mode != "fp16":
                modules = json.loads((directory / "quantized_modules.json").read_text())
                assert len(modules) == 252
                assert len({m["name"] for m in modules}) == 252
                assert all(m["bits"] == int(mode[-1]) for m in modules)
            results[mode, r], samples[mode, r] = result, rows
        a, b = samples[mode, 1], samples[mode, 2]
        changed = sum(x["prediction"] != y["prediction"] or x["prediction_norm"] != y["prediction_norm"]
                      for x, y in zip(a,b) if x["task"] != "wikitext2")
        ll_delta = max(abs(i-j) for x,y in zip(a,b) if x["task"] != "wikitext2"
                       for i,j in zip(x["loglikelihoods"], y["loglikelihoods"]))
        nll_delta = abs(results[mode,1]["metrics"]["wikitext2"]["mean_nll"] -
                        results[mode,2]["metrics"]["wikitext2"]["mean_nll"])
        assert changed == 0 and ll_delta <= 1e-3 and nll_delta <= 1e-5, f"repeat mismatch: {mode}"
        repeat[mode] = {"changed_predictions": changed, "max_choice_logprob_delta": ll_delta,
                        "mean_nll_delta": nll_delta, "raw_samples_identical": a == b}
    fp = results["fp16",1]["metrics"]
    q8, q4 = results["fixed8",1]["metrics"], results["fixed4",1]["metrics"]
    for task in ("hellaswag", "arc_challenge"):
        assert fp[task]["acc_norm"] > .30, f"FP16 sanity: {task}"
        assert abs(q8[task]["acc_norm"] - fp[task]["acc_norm"]) <= .08, f"8-bit sanity: {task}"
        assert q4[task]["acc_norm"] > .25, f"4-bit sanity: {task}"
    ratios = {mode: results[mode,1]["metrics"]["wikitext2"]["token_perplexity"] /
              fp["wikitext2"]["token_perplexity"] for mode in ("fixed8", "fixed4")}
    assert .8 <= ratios["fixed8"] <= 1.25, f"8-bit PPL mismatch: {ratios}"
    assert .5 <= ratios["fixed4"] <= 3., f"4-bit PPL mismatch: {ratios}"
    return {"stage": "baseline_only", "passed": True, "repeatability": repeat,
            "ppl_ratios_vs_fp16": ratios, "metrics": {
                mode: results[mode,1]["metrics"] for mode in ("fp16", "fixed8", "fixed4")},
            "does_not_verify": ["integration", "stored_checkpoint", "router", "matched-budget comparison", "final report"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/core-v1"))
    args = parser.parse_args()
    try:
        report = check(args.root)
    except Exception as exc:
        write_json(args.root / "baseline-gate-failure.json", {"passed": False, "error": repr(exc)})
        raise
    write_json(args.root / "baseline-gate.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
