"""Freeze actual inputs before any new model scores. CPU only."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from qaq.evaluation import prepare_examples, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/core_protocol.json")
    parser.add_argument("--out", default="results/core-v1/frozen")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    write_json(out / "protocol.json", config)
    spec = config["model"]
    model_path = Path(snapshot_download(spec["id"], revision=spec["revision"],
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.md", "LICENSE"]))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    examples, data_manifest = prepare_examples(config, tokenizer)
    with (out / "examples.jsonl").open("x") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    write_json(out / "data_manifest.json", data_manifest)
    files = {p.name: sha256(p) for p in sorted(model_path.iterdir()) if p.is_file()}
    write_json(out / "model_manifest.json", {"id": spec["id"], "revision": spec["revision"],
        "local_path": str(model_path), "file_sha256": files})
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json(out / "environment.json", {"python": platform.python_version(),
        "platform": platform.platform(), "packages": packages})
    for command, filename in [(["python", "-m", "pip", "freeze"], "pip-freeze.txt"),
                               (["nvidia-smi", "-q"], "nvidia-smi.txt"),
                               (["lscpu"], "cpu.txt"), (["free", "-b"], "ram.txt")]:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        (out / filename).write_text(result.stdout)
    protected = ["protocol.json", "examples.jsonl", "data_manifest.json", "model_manifest.json",
                 "environment.json"]
    write_json(out / "freeze_hashes.json", {name: sha256(out/name) for name in protected})
    print(json.dumps({"frozen": str(out), "examples": len(examples), "datasets": data_manifest}, indent=2))


if __name__ == "__main__":
    main()
