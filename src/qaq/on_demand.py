"""Packed two's-complement weights for the separate synchronous-loading study.

This module leaves the completed ``NestedLinear`` checkpoint format unchanged.
Both storage modes below use the same selected-plane unpacking, the existing
``qaq.quantization.reconstruct`` function, and ordinary FP16 ``F.linear``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qaq.model import NestedLinear, blocks
from qaq.quantization import PRECISIONS, reconstruct

STORAGE_MODES = ("resident", "ondemand_sync")
STUDY_SOURCE_PATHS = (
    "ON_DEMAND_PROTOCOL.md",
    "configs/on_demand_protocol.json",
    "scripts/check_on_demand.py",
    "scripts/check_on_demand_cpu.py",
    "scripts/gpu_preflight.sh",
    "scripts/run_on_demand.py",
    "src/qaq/__init__.py",
    "src/qaq/bitplanes.py",
    "src/qaq/evaluation.py",
    "src/qaq/model.py",
    "src/qaq/on_demand.py",
    "src/qaq/quantization.py",
    "tests/test_on_demand.py",
)


def study_source_manifest(root: str | Path = ".") -> dict[str, str]:
    root = Path(root)
    result = {}
    for name in STUDY_SOURCE_PATHS:
        digest = hashlib.sha256()
        with open(root / name, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def manifest_sha256(manifest: dict[str, str]) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def packed_bytes(elements: int) -> int:
    if elements <= 0:
        raise ValueError("elements must be positive")
    return (elements + 7) // 8


def pack_twos_complement(q: torch.Tensor) -> torch.Tensor:
    """Pack an int8 tensor into eight LSB-first planes, eight codes per byte.

    Plane 0 is the least-significant code bit and plane 7 is the two's-
    complement sign bit. Within each packed byte, flattened code ``i`` uses bit
    ``i % 8``. Padding bits in the final byte are zero.
    """
    if q.dtype != torch.int8 or q.numel() == 0:
        raise ValueError("q must be a nonempty int8 tensor")
    raw = q.detach().cpu().contiguous().numpy().view(np.uint8).reshape(-1)
    planes = [np.packbits((raw >> bit) & 1, bitorder="little") for bit in range(8)]
    return torch.from_numpy(np.stack(planes)).contiguous()


def unpack_twos_complement(
    selected_planes: torch.Tensor,
    elements: int,
    bits: int,
) -> torch.Tensor:
    """Unpack the selected high planes to int8 codes with low bits cleared."""
    if bits not in PRECISIONS:
        raise ValueError(f"bits must be one of {PRECISIONS}")
    if (selected_planes.dtype != torch.uint8 or selected_planes.ndim != 2
            or selected_planes.shape != (bits, packed_bytes(elements))):
        raise ValueError("selected plane shape/dtype does not match elements and bits")
    lanes = torch.arange(8, device=selected_planes.device, dtype=torch.uint8)
    unsigned = torch.zeros(elements, device=selected_planes.device, dtype=torch.int16)
    first = 8 - bits
    for row, position in enumerate(range(first, 8)):
        values = ((selected_planes[row, :, None] >> lanes) & 1).reshape(-1)[:elements]
        unsigned.add_(values.to(torch.int16), alpha=1 << position)
    signed = torch.where(unsigned >= 128, unsigned - 256, unsigned)
    return signed.to(torch.int8)


def reconstruct_packed(
    selected_planes: torch.Tensor,
    q_shape: tuple[int, int, int],
    scale: torch.Tensor,
    bits: int,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Use packed high planes but preserve the core reconstruction exactly."""
    elements = int(np.prod(q_shape))
    q = unpack_twos_complement(selected_planes, elements, bits).reshape(q_shape)
    return reconstruct(q, scale, bits, dtype)


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    q_shape: tuple[int, int, int]
    packed_start: int
    packed_stop: int
    scale_start: int
    scale_stop: int
    scale_shape: tuple[int, int, int]
    dtype: torch.dtype

    @property
    def elements(self) -> int:
        return int(np.prod(self.q_shape))


@dataclass
class BlockPayload:
    planes: torch.Tensor
    scales: torch.Tensor
    projections: tuple[ProjectionSpec, ...]

    @property
    def scale_bytes(self) -> int:
        return self.scales.numel() * self.scales.element_size()


@dataclass
class ActivePayload:
    block: int
    bits: int
    planes: torch.Tensor
    scales: torch.Tensor


class PackedLinear(nn.Module):
    """Projection metadata only; reconstructed weights never persist after call."""

    def __init__(self, spec: ProjectionSpec, bias: torch.Tensor | None):
        super().__init__()
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "manager", None)
        self.register_buffer("bias", bias)
        self.in_features = spec.q_shape[1] * spec.q_shape[2]
        self.out_features = spec.q_shape[0]

    def bind(self, manager: "PackedStorage") -> None:
        object.__setattr__(self, "manager", manager)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.manager is None:
            raise RuntimeError("packed linear is not bound to storage")
        weight = self.manager.reconstruct(self.spec)
        return F.linear(inputs, weight, self.bias)


class PackedStorage:
    """Own packed block payloads and one optional reusable on-demand GPU slot."""

    def __init__(self, payloads: list[BlockPayload], mode: str):
        if mode not in STORAGE_MODES:
            raise ValueError(f"mode must be one of {STORAGE_MODES}")
        self.payloads = payloads
        self.mode = mode
        self.profile = [8] * len(payloads)
        self.device = torch.device("cpu")
        self.slot_planes: torch.Tensor | None = None
        self.slot_scales: torch.Tensor | None = None
        self.active: ActivePayload | None = None
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.events: list[dict] = []
        self.reset_stats()

    def prepare(self, device: str | torch.device) -> dict:
        """Place source storage or allocate the sole reusable block slot."""
        self.device = torch.device(device)
        if self.active is not None:
            raise RuntimeError("cannot prepare while a block is active")
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        copied = 0
        if self.mode == "resident":
            for payload in self.payloads:
                copied += payload.planes.numel() * payload.planes.element_size()
                copied += payload.scale_bytes
                payload.planes = payload.planes.to(self.device)
                payload.scales = payload.scales.to(self.device)
        else:
            if any(p.planes.device.type != "cpu" or p.scales.device.type != "cpu"
                   for p in self.payloads):
                raise RuntimeError("on-demand source payloads must remain on CPU")
            max_plane_bytes = max(p.planes.shape[1] for p in self.payloads)
            max_scales = max(p.scales.numel() for p in self.payloads)
            self.slot_planes = torch.empty((8, max_plane_bytes), dtype=torch.uint8,
                                           device=self.device)
            self.slot_scales = torch.empty(max_scales, dtype=torch.float32,
                                           device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "seconds": time.perf_counter() - started,
            "h2d_copied_bytes": copied if self.device.type == "cuda" else 0,
            "slot_capacity_bytes": self.slot_capacity_bytes,
        }

    @property
    def slot_capacity_bytes(self) -> int:
        tensors = (self.slot_planes, self.slot_scales)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def attach(self, model: nn.Module) -> None:
        if self.handles:
            raise RuntimeError("storage hooks are already attached")
        block_list = list(blocks(model))
        if len(block_list) != len(self.payloads):
            raise ValueError("block/payload count mismatch")
        for index, block in enumerate(block_list):
            def before(_module, _args, _kwargs, index=index):
                self.begin(index)

            def after(_module, _args, _kwargs, output, index=index):
                self.end(index)
                return output

            self.handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(block.register_forward_hook(
                after, with_kwargs=True, always_call=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = None

    def set_profile(self, profile: list[int]) -> None:
        if (len(profile) != len(self.payloads)
                or any(bits not in PRECISIONS for bits in profile)):
            raise ValueError("profile must contain one 4/6/8 choice per block")
        if self.active is not None:
            raise RuntimeError("cannot change profile during block execution")
        self.profile = list(profile)

    def reset_stats(self) -> None:
        if getattr(self, "active", None) is not None:
            raise RuntimeError("cannot reset statistics while a block is active")
        self.requests = 0
        self.loads = 0
        self.releases = 0
        self.view_closes = 0
        self.requested_plane_bits = 0
        self.requested_bytes = 0
        self.copied_bytes = 0
        self.transfer_seconds = 0.0
        self.max_active_slots = 0
        self.events = []

    def begin(self, index: int) -> None:
        if self.active is not None:
            raise RuntimeError("only one block payload may be active")
        payload = self.payloads[index]
        bits = self.profile[index]
        first = 8 - bits
        plane_bytes = payload.planes.shape[1]
        logical_plane_bits = sum(spec.elements for spec in payload.projections) * bits
        requested = (logical_plane_bits + 7) // 8 + payload.scale_bytes
        copied = 0
        transfer = 0.0
        if self.mode == "resident":
            if payload.planes.device != self.device or payload.scales.device != self.device:
                raise RuntimeError("resident payload is not on the compute device")
            selected = payload.planes[first:]
            scales = payload.scales
        else:
            if self.slot_planes is None or self.slot_scales is None:
                raise RuntimeError("on-demand slot is not prepared")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            # Row-wise copy keeps every physical transfer contiguous while moving
            # only selected high planes. The slot's unselected rows remain unused.
            for row in range(first, 8):
                self.slot_planes[row, :plane_bytes].copy_(payload.planes[row])
            self.slot_scales[:payload.scales.numel()].copy_(payload.scales)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            transfer = time.perf_counter() - started
            selected = self.slot_planes[first:, :plane_bytes]
            scales = self.slot_scales[:payload.scales.numel()]
            copied = bits * plane_bytes + payload.scale_bytes
            self.loads += 1
        self.active = ActivePayload(index, bits, selected, scales)
        self.requests += 1
        self.requested_plane_bits += logical_plane_bits
        self.requested_bytes += requested
        self.copied_bytes += copied
        self.transfer_seconds += transfer
        self.max_active_slots = max(self.max_active_slots, 1)
        self.events.append({
            "request": self.requests,
            "block": index,
            "kind": "attention" if index % 2 == 0 else "ffn",
            "bits": bits,
            "logical_plane_bits": logical_plane_bits,
            "requested_bytes": requested,
            "copied_bytes": copied,
            "transfer_seconds": transfer,
            "released": False,
        })

    def end(self, index: int) -> None:
        if self.active is None or self.active.block != index:
            raise RuntimeError("active block release mismatch")
        self.active = None
        self.view_closes += 1
        if self.mode == "ondemand_sync":
            self.releases += 1
        self.events[-1]["released"] = True

    def reconstruct(self, spec: ProjectionSpec) -> torch.Tensor:
        active = self.active
        if active is None:
            raise RuntimeError("projection executed without an active block payload")
        payload = self.payloads[active.block]
        if spec not in payload.projections:
            raise RuntimeError("projection does not belong to the active block")
        planes = active.planes[:, spec.packed_start:spec.packed_stop]
        scales = active.scales[spec.scale_start:spec.scale_stop].reshape(spec.scale_shape)
        return reconstruct_packed(planes, spec.q_shape, scales, active.bits, spec.dtype)

    def stats(self) -> dict:
        return {
            "requests": self.requests,
            "loads": self.loads,
            "releases": self.releases,
            "view_closes": self.view_closes,
            "requested_plane_bits": self.requested_plane_bits,
            "requested_bytes": self.requested_bytes,
            "copied_bytes": self.copied_bytes,
            "transfer_seconds": self.transfer_seconds,
            "max_active_slots": self.max_active_slots,
            "active_slots_at_end": int(self.active is not None),
            "slot_capacity_bytes": self.slot_capacity_bytes,
        }

    def inventory(self) -> dict:
        plane_bytes = sum(p.planes.numel() * p.planes.element_size() for p in self.payloads)
        scale_bytes = sum(p.scale_bytes for p in self.payloads)
        source_devices = sorted({str(p.planes.device) for p in self.payloads}
                                | {str(p.scales.device) for p in self.payloads})
        source_gpu_bytes = sum(
            p.planes.numel() * p.planes.element_size() + p.scale_bytes
            for p in self.payloads if p.planes.device.type == "cuda")
        return {
            "packed_plane_bytes": plane_bytes,
            "scale_bytes": scale_bytes,
            "source_storage_bytes": plane_bytes + scale_bytes,
            "source_devices": source_devices,
            "source_gpu_bytes": source_gpu_bytes,
            "slot_capacity_bytes": self.slot_capacity_bytes,
            "active_slot_bytes": 0 if self.active is None else (
                self.active.planes.numel() * self.active.planes.element_size()
                + self.active.scales.numel() * self.active.scales.element_size()),
        }


def convert_nested_model(model: nn.Module, mode: str) -> PackedStorage:
    """Pack every core ``NestedLinear`` by block and remove its q/scale buffers."""
    payloads: list[BlockPayload] = []
    packed_linears: list[PackedLinear] = []
    for index, block in enumerate(blocks(model)):
        originals = [(name, module) for name, module in block.named_children()
                     if isinstance(module, NestedLinear)]
        expected = 4 if index % 2 == 0 else 3
        if len(originals) != expected:
            raise ValueError("expected an integrated Qwen model with 4/3 block projections")
        plane_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        specs: list[ProjectionSpec] = []
        packed_offset = scale_offset = 0
        for name, linear in originals:
            planes = pack_twos_complement(linear.q)
            scales = linear.scale.detach().cpu().contiguous().reshape(-1)
            spec = ProjectionSpec(
                name=name,
                q_shape=tuple(linear.q.shape),
                packed_start=packed_offset,
                packed_stop=packed_offset + planes.shape[1],
                scale_start=scale_offset,
                scale_stop=scale_offset + scales.numel(),
                scale_shape=tuple(linear.scale.shape),
                dtype=linear.dtype,
            )
            replacement = PackedLinear(
                spec, linear.bias.detach().clone() if linear.bias is not None else None)
            setattr(block, name, replacement)
            packed_linears.append(replacement)
            plane_parts.append(planes)
            scale_parts.append(scales)
            specs.append(spec)
            packed_offset += planes.shape[1]
            scale_offset += scales.numel()
        payloads.append(BlockPayload(torch.cat(plane_parts, dim=1),
                                     torch.cat(scale_parts), tuple(specs)))
    storage = PackedStorage(payloads, mode)
    for linear in packed_linears:
        linear.bind(storage)
    storage.attach(model)
    if any(isinstance(module, NestedLinear) for module in model.modules()):
        raise AssertionError("original int8/scale modules remain installed")
    return storage
