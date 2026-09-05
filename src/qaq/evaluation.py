"""Small, fixed-batch causal scorer. All metrics and samples are retained.

Prompt preprocessing follows the few reviewed rules in lm_eval 0.4.13's
hellaswag/utils.py and arc/arc_easy.yaml (MIT); no foreign execution or results.
This is not a full lm-evaluation-harness run. WT2 measures prefix-conditioned
TOKEN perplexity, not the paper's unspecified or harness word perplexity.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean_hellaswag(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def make_mc(task, row, index):
    if task == "hellaswag":
        context = clean_hellaswag(row["activity_label"] + ": " + row["ctx_a"] + " "
                                  + row["ctx_b"].capitalize())
        choices = [clean_hellaswag(s) for s in row["endings"]]
        gold = int(row["label"])
    elif task == "arc_challenge":
        context = f"Question: {row['question']}\nAnswer:"
        choices = list(row["choices"]["text"])
        gold = list(row["choices"]["label"]).index(row["answerKey"])
    else:
        raise ValueError(task)
    if not choices or any(not s for s in choices) or not 0 <= gold < len(choices):
        raise ValueError("invalid multiple-choice example")
    return {"task": task, "index": index, "source_id": row.get("id", row.get("ind")),
            "context": context, "choices": choices, "gold": gold}


def encode_pair(tokenizer, context, answer, max_length):
    prefix = tokenizer.encode(context, add_special_tokens=False)
    whole = tokenizer.encode(context + " " + answer, add_special_tokens=False)
    if not prefix or whole[:len(prefix)] != prefix or len(whole) <= len(prefix):
        raise ValueError("continuation tokenization boundary changed")
    if len(whole) > max_length:
        raise ValueError(f"sequence length {len(whole)} exceeds frozen cap {max_length}")
    return whole, len(prefix)


def prepare_examples(config, tokenizer):
    examples, manifest = [], {}
    for task, spec in config["data"].items():
        path = hf_hub_download(spec["id"], spec["file"], revision=spec["revision"],
                               repo_type="dataset")
        rows = pq.read_table(path).to_pylist()
        if task == "wikitext2":
            text = "\n\n".join(row["text"] for row in rows)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            width = spec["window_tokens"]
            candidates = len(tokens) // width
        else:
            candidates = len(rows)
        indices = sorted(random.Random(config["evaluation"]["seed"]).sample(
            range(candidates), spec["count"]))
        manifest[task] = {**spec, "sha256": sha256(path), "source_rows": len(rows),
                          "candidates": candidates, "indices": indices}
        for index in indices:
            if task == "wikitext2":
                ex = {"task": task, "index": index, "tokens": tokens[index*width:(index+1)*width],
                      "prefix_length": spec["prefix_tokens"]}
            else:
                ex = make_mc(task, rows[index], index)
                ex["encoded_choices"] = [encode_pair(tokenizer, ex["context"], choice,
                    config["evaluation"]["max_sequence_tokens"]) for choice in ex["choices"]]
            examples.append(ex)
        if task == "wikitext2":
            manifest[task]["joined_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            manifest[task]["total_tokens"] = len(tokens)
            manifest[task]["discarded_tail_tokens"] = len(tokens) % width
    return examples, manifest


@torch.inference_mode()
def continuation_nll(model, tokens, prefix_length):
    """Causal positions prefix-1..T-2 predict targets prefix..T-1 exactly once."""
    if not 1 <= prefix_length < len(tokens):
        raise ValueError("nonempty prefix and continuation required")
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=model.device)
    output = model(input_ids=inputs, use_cache=False)
    logits = output.logits[0, prefix_length-1:].float()
    target = torch.tensor(tokens[prefix_length:], dtype=torch.long, device=model.device)
    losses = F.cross_entropy(logits, target, reduction="none")
    if not torch.isfinite(losses).all():
        raise ValueError("nonfinite token loss")
    return losses.double().cpu().tolist()


def stderr(values):
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def summarize(samples):
    results = {}
    for task in sorted({s["task"] for s in samples}):
        rows = [s for s in samples if s["task"] == task]
        if task == "wikitext2":
            total = sum(s["nll_sum"] for s in rows)
            count = sum(s["scored_tokens"] for s in rows)
            means = [s["nll_sum"] / s["scored_tokens"] for s in rows]
            results[task] = {"nll_sum": total, "scored_tokens": count,
                             "mean_nll": total/count, "token_perplexity": math.exp(total/count),
                             "mean_window_nll_stderr": stderr(means)}
        else:
            acc = [float(s["prediction"] == s["gold"]) for s in rows]
            norm = [float(s["prediction_norm"] == s["gold"]) for s in rows]
            results[task] = {"acc": statistics.mean(acc), "acc_stderr": stderr(acc),
                             "acc_norm": statistics.mean(norm), "acc_norm_stderr": stderr(norm),
                             "count": len(rows)}
    return results


def evaluate(model, examples, path, route=None):
    """route(context_tokens) may set weights, but never receives suffix or gold."""
    samples = []
    with open(path, "x") as f:
        for i, ex in enumerate(examples):
            if ex["task"] == "wikitext2":
                prefix = ex["tokens"][:ex["prefix_length"]]
            else:
                tokens, length = ex["encoded_choices"][0]
                prefix = tokens[:length]
                assert all(t[:n] == prefix for t, n in ex["encoded_choices"])
            profile = route(prefix) if route else None
            if ex["task"] == "wikitext2":
                losses = continuation_nll(model, ex["tokens"], ex["prefix_length"])
                row = {"task": ex["task"], "index": ex["index"], "nll_sum": sum(losses),
                       "scored_tokens": len(losses), "token_nll": losses}
            else:
                losses = [continuation_nll(model, t, n) for t, n in ex["encoded_choices"]]
                lls = [-sum(ls) for ls in losses]
                norm = [ll/len(choice) for ll, choice in zip(lls, ex["choices"])]
                row = {"task": ex["task"], "index": ex["index"], "gold": ex["gold"],
                       "loglikelihoods": lls, "normalized_loglikelihoods": norm,
                       "token_nll": losses,
                       "prediction": max(range(len(lls)), key=lls.__getitem__),
                       "prediction_norm": max(range(len(norm)), key=norm.__getitem__)}
            row["profile"] = profile
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            samples.append(row)
            if i % 16 == 0:
                print(f"evaluated {i+1}/{len(examples)}: {ex['task']}", flush=True)
    return summarize(samples)
