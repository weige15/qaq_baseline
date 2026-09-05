"""CPU-only article-separated router inputs; no model inference or fitting."""
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from qaq.evaluation import sha256, write_json


def text_hash(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def spans(tokens, width=32):
    return {tuple(tokens[i:i+width]) for i in range(len(tokens)-width+1)}


def articles(rows):
    docs = []
    for i, row in enumerate(rows):
        text = row["text"]
        title = text.strip()
        if re.fullmatch(r"= [^=].*[^=] =", title):
            docs.append({"title": title, "start_row": i, "rows": []})
        if docs:
            docs[-1]["rows"].append(text)
    for index, doc in enumerate(docs):
        doc["article_index"] = index
        doc["text"] = "\n\n".join(doc.pop("rows"))
        doc["text_hash"] = text_hash(doc["text"])
        doc["title_hash"] = text_hash(doc["title"])
    return docs


def main():
    root = Path("results/core-v1")
    out = root / "router-frozen"
    out.mkdir(exist_ok=False)
    cfg = json.loads(Path("configs/router_protocol.json").read_text())
    base_hashes = json.loads((root / "frozen/freeze_hashes.json").read_text())
    for name, digest in base_hashes.items():
        assert sha256(root / "frozen" / name) == digest
    write_json(out / "protocol.json", cfg)
    shutil.copyfile("ROUTER_PROTOCOL.md", out / "ROUTER_PROTOCOL.md")
    write_json(out / "command.json", {"argv": sys.argv, "executable": sys.executable,
                                      "base_frozen_hashes": base_hashes})
    info = json.loads((root / "frozen/model_manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(info["local_path"], local_files_only=True)
    source, docs = {}, {}
    for split in ("train", "validation", "test"):
        file = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
        path = hf_hub_download(cfg["data_id"], file, revision=cfg["data_revision"], repo_type="dataset")
        source[split] = {"path": path, "file": file, "sha256": sha256(path)}
        docs[split] = articles(pq.read_table(path).to_pylist())
    final = [json.loads(s) for s in (root / "frozen/examples.jsonl").read_text().splitlines()]
    forbidden_spans = set()
    for ex in final:
        for tokens in ([ex["tokens"]] if ex["task"] == "wikitext2" else [t for t, _ in ex["encoded_choices"]]):
            forbidden_spans.update(spans(tokens, cfg["dedup_span_tokens"]))
    test_titles = {d["title_hash"] for d in docs["test"]}
    test_texts = {d["text_hash"] for d in docs["test"]}
    dev_titles = {d["title_hash"] for d in docs["validation"]}
    dev_texts = {d["text_hash"] for d in docs["validation"]}
    selected, rejected, candidate_counts = [], [], {}
    train_spans = set()
    for split, count in cfg["splits"].items():
        candidates = []
        forbidden_titles = test_titles | (dev_titles if split == "train" else set())
        forbidden_texts = test_texts | (dev_texts if split == "train" else set())
        for doc in docs[split]:
            reasons = []
            if doc["title_hash"] in forbidden_titles or doc["text_hash"] in forbidden_texts:
                reasons.append("cross_split_article_duplicate")
            tokens = tokenizer.encode(doc["text"], add_special_tokens=False)
            width = cfg["window_tokens"]
            clean = []
            for offset in range(0, len(tokens)-width+1, width):
                window = tokens[offset:offset+width]
                ng = spans(window, cfg["dedup_span_tokens"])
                if ng & forbidden_spans or (split == "validation" and ng & train_spans):
                    rejected.append({"split": split, "article_index": doc["article_index"],
                                     "offset": offset, "reason": "32_token_overlap"})
                else:
                    clean.append((offset, window))
            if not clean:
                reasons.append("no_clean_complete_window")
            if reasons:
                rejected.append({"split": split, "article_index": doc["article_index"], "reasons": reasons})
            else:
                candidates.append((doc, clean))
        candidate_counts[split] = len(candidates)
        if len(candidates) < count:
            write_json(out / "failure.json", {"split": split, "eligible": len(candidates), "required": count})
            raise RuntimeError("not enough clean articles; pause without reducing count")
        rng = random.Random(cfg["data_seed"])
        choices = sorted(rng.sample(range(len(candidates)), count))
        split_spans = set()
        for choice in choices:
            doc, windows = candidates[choice]
            offset, tokens = rng.choice(windows)
            context = tokens[:cfg["context_tokens"]]
            record = {**doc, "split": split, "offset": offset, "tokens": tokens,
                      "context_tokens": context, "context_sha256": hashlib.sha256(json.dumps(context).encode()).hexdigest()}
            selected.append(record)
            split_spans.update(spans(tokens, cfg["dedup_span_tokens"]))
        assert not split_spans & forbidden_spans
        if split == "train":
            train_spans = split_spans
        else:
            assert not split_spans & train_spans
    with (out / "examples.jsonl").open("x") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")
    write_json(out / "data_manifest.json", {"dataset": cfg["data_id"], "revision": cfg["data_revision"],
        "source": source, "counts": cfg["splits"], "eligible_articles": candidate_counts,
        "rejected": rejected, "final_32_token_spans": len(forbidden_spans),
        "train_dev_final_exact_span_overlap": 0,
        "limitation": "exact 32-token and normalized title/text dedup, not semantic/pretraining decontamination"})
    protected = ["protocol.json", "ROUTER_PROTOCOL.md", "examples.jsonl", "data_manifest.json", "command.json"]
    write_json(out / "freeze_hashes.json", {name: sha256(out / name) for name in protected})
    print(json.dumps({"counts": cfg["splits"], "eligible": candidate_counts, "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
