"""Causal local-distillation router and exact per-type precision quotas.

This is a correctness reference: an extra fixed8 probe and FP16 matmuls, not
an on-demand loader or a low-bit performance kernel.
"""
from collections import Counter
import hashlib
import json
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from qaq.model import NestedLinear, blocks, set_profile


PRECISIONS = (4, 6, 8)


def hidden_features(hidden, groups=64):
    h = hidden.detach().float()[0]
    if hidden.shape[0] != 1 or h.shape[-1] % groups:
        raise ValueError("one query and divisible channel groups required")
    grouped = h.reshape(h.shape[0], groups, -1)
    token_rms = h.square().mean(-1).sqrt()
    return torch.cat((grouped.mean((0, 2)), grouped.square().mean((0, 2)).sqrt(),
                      torch.stack((h.square().mean().sqrt(), h.abs().max(),
                                   token_rms.std(unbiased=False), h.new_tensor(math.log(len(h)))))))


def select_features(raw, groups):
    if raw.shape[-1] != 132 or groups not in (32, 64):
        raise ValueError("expected frozen132 raw features and32/64 groups")
    if groups == 64:
        return raw
    shape = (*raw.shape[:-1], 32, 2)
    return torch.cat((raw[..., :64].reshape(shape).mean(-1),
                      raw[..., 64:128].square().reshape(shape).mean(-1).sqrt(), raw[..., 128:]), -1)


@torch.inference_mode()
def collect_context(model, tokens, teacher=None):
    """Fixed8 features; optional local FP16 teacher targets at identical inputs.

    Hook replays call .forward directly, bypassing hooks and preventing recursion.
    Only the original fixed8 outputs propagate to later blocks.
    """
    block_list = list(blocks(model))
    teacher_blocks = list(blocks(teacher)) if teacher is not None else None
    set_profile(model, [8] * len(block_list))
    features, errors, handles = [], [], []

    def hook_for(index):
        def hook(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            features.append(hidden_features(hidden).cpu())
            if teacher_blocks is None:
                return
            reference = teacher_blocks[index].forward(*args, **kwargs)
            reference = reference[0] if isinstance(reference, tuple) else reference
            reference = reference.float()
            denominator = reference.square().mean().clamp_min(1e-12)
            values = []
            for bits in PRECISIONS:
                if bits == 8:
                    actual = output
                else:
                    for linear in module.children():
                        if isinstance(linear, NestedLinear):
                            linear.set_bits(bits)
                    actual = module.forward(*args, **kwargs)
                actual = actual[0] if isinstance(actual, tuple) else actual
                values.append(((actual.float()-reference).square().mean()/denominator).cpu())
            errors.append(torch.stack(values))
            # Keep only one block's reconstruction live during label collection.
            for linear in module.children():
                if isinstance(linear, NestedLinear):
                    linear.set_bits(8)
                    linear.active_weight = None
        return hook

    try:
        for i, block in enumerate(block_list):
            handles.append(block.register_forward_hook(hook_for(i), with_kwargs=True))
        inputs = torch.tensor([tokens], dtype=torch.long, device=model.device)
        model.model(input_ids=inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    result = torch.stack(features)
    targets = torch.stack(errors) if teacher is not None else None
    if len(result) != len(block_list) or not torch.isfinite(result).all():
        raise ValueError("invalid feature collection")
    if targets is not None and (not torch.isfinite(targets).all() or (targets < 0).any()):
        raise ValueError("invalid local distillation errors")
    return result, targets


def quota_profile(costs):
    """Global minimum for each type; deterministic installed SciPy tie handling."""
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[1] != 3 or costs.shape[0] % 6 or not np.isfinite(costs).all():
        raise ValueError("finite [blocks,3] costs, with each type divisible by3, required")
    profile = np.zeros(len(costs), dtype=int)
    quota = len(costs) // 6
    slots = np.repeat(np.arange(3), quota)
    for kind in (0, 1):
        rows, columns = linear_sum_assignment(costs[kind::2, :][:, slots])
        profile[2*rows+kind] = np.array(PRECISIONS)[slots[columns]]
    return profile.tolist()


def random_profile(tokens, seed=1618, nblocks=72):
    digest = hashlib.sha256(json.dumps(tokens).encode()).digest()
    rng = random.Random(seed + int.from_bytes(digest, "big"))
    if nblocks % 6:
        raise ValueError("invalid block count")
    profile = [0]*nblocks
    for kind in (0, 1):
        levels = list(PRECISIONS)*(nblocks//6)
        rng.shuffle(levels)
        profile[kind::2] = levels
    return profile


def check_budget(profile, counts):
    if len(profile) != len(counts) or len(counts) % 6 or any(c <= 0 for c in counts):
        raise ValueError("profile/count mismatch")
    quota = len(profile)//6
    for kind in (0, 1):
        if Counter(profile[kind::2]) != {b: quota for b in PRECISIONS}:
            raise ValueError("precision quota mismatch")
        if len(set(counts[kind::2])) != 1:
            raise ValueError("per-type quotas require equal block sizes")
    total = sum(b*c for b, c in zip(profile, counts))
    if total != 6*sum(counts):
        raise ValueError("weighted budget mismatch")
    return {"quantized_parameters": sum(counts), "selected_weight_bits": total, "mean_bits": total/sum(counts)}


def distribution(profiles):
    p = np.array(profiles, dtype=int)
    frequencies = Counter(tuple(row) for row in profiles)
    varying = (p != p[0]).any(axis=0)
    return {"queries": len(profiles), "unique_profiles": len(frequencies),
            "largest_profile_fraction": max(frequencies.values())/len(profiles),
            "varying_attention_blocks": int(varying[::2].sum()),
            "varying_ffn_blocks": int(varying[1::2].sum()),
            "per_block_counts": [{str(b): int((column == b).sum()) for b in PRECISIONS} for column in p.T],
            "precision_counts": {str(b): int((p == b).sum()) for b in PRECISIONS},
            "profile_frequencies": [{"profile": list(k), "count": v} for k, v in frequencies.most_common()]}


class BlockRouter(nn.Module):
    """72 independent two-layer MLPs, vectorized for small CPU-only fitting."""
    def __init__(self, features, hidden, nblocks=72):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(nblocks, features, hidden)/math.sqrt(features))
        self.b1 = nn.Parameter(torch.zeros(nblocks, 1, hidden))
        self.w2 = nn.Parameter(torch.randn(nblocks, hidden, 3)/math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(nblocks, 1, 3))

    def forward(self, x):
        h = torch.relu(torch.bmm(x.transpose(0, 1), self.w1)+self.b1)
        return (torch.bmm(h, self.w2)+self.b2).transpose(0, 1)


def normalize_features(x, normalization):
    scaled = (x-normalization["x_mean"])/normalization["x_std"]
    return torch.where(normalization["x_mask"], scaled, 0.0)


@torch.inference_mode()
def router_costs(router, raw, spec, normalization):
    x = normalize_features(select_features(raw, spec["groups"]), normalization)
    y = router(x)
    if spec["objective"] == "log_nmse":
        return (y*normalization["y_std"]+normalization["y_mean"]).clamp(-30, 10).exp()
    return -y.log_softmax(-1)


def load_router(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    spec = saved["attempt"]
    router = BlockRouter(2*spec["groups"]+4, spec["hidden"])
    router.load_state_dict(saved["state_dict"], strict=True)
    return router.eval(), saved
