"""A toy signed bit-plane candidate for QAQ reconstruction work.

The paper does not state how negative real weights, scales, or zero points are
stored. This module therefore implements one explicit candidate rather than
claiming to recover the authors' format:

* symmetric quantization with one scale per selected axis;
* one sign plane;
* ``bits - 1`` magnitude planes, ordered least-significant first;
* lower precision keeps the sign and the most-significant magnitude planes.

The full-width path is exactly reversible at the quantized-integer level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class QuantizedTensor:
    """Integer weights and the scale needed to recover approximate real values."""

    values: NDArray[np.int32]
    scale: NDArray[np.float64]
    bits: int
    reduce_axis: int | tuple[int, ...]


def _validate_bits(bits: int) -> None:
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")


def symmetric_quantize(
    values: ArrayLike,
    *,
    bits: int = 8,
    reduce_axis: int | tuple[int, ...] = -1,
) -> QuantizedTensor:
    """Quantize real values symmetrically while retaining a broadcastable scale."""

    _validate_bits(bits)
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")

    limit = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(array), axis=reduce_axis, keepdims=True)
    scale = np.where(maximum == 0.0, 1.0, maximum / limit).astype(np.float64)
    integers = np.rint(array / scale)
    integers = np.clip(integers, -limit, limit).astype(np.int32)
    return QuantizedTensor(integers, scale, bits, reduce_axis)


def symmetric_dequantize(quantized: QuantizedTensor) -> NDArray[np.float64]:
    """Recover real-valued approximations from a :class:`QuantizedTensor`."""

    return quantized.values.astype(np.float64) * quantized.scale


def to_sign_magnitude_planes(
    integers: ArrayLike,
    *,
    bits: int = 8,
) -> NDArray[np.uint8]:
    """Encode signed integers as magnitude planes followed by one sign plane."""

    _validate_bits(bits)
    raw = np.asarray(integers)
    if raw.size == 0:
        raise ValueError("integers must not be empty")
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw)):
        raise ValueError("integers must contain finite integer values")

    signed = raw.astype(np.int64)
    limit = (1 << (bits - 1)) - 1
    if np.any(signed < -limit) or np.any(signed > limit):
        raise ValueError(f"values must be within {-limit}..{limit} for {bits} bits")

    magnitude = np.abs(signed).astype(np.uint64)
    magnitude_planes = [
        ((magnitude >> offset) & 1).astype(np.uint8)
        for offset in range(bits - 1)
    ]
    sign_plane = (signed < 0).astype(np.uint8)
    return np.stack([*magnitude_planes, sign_plane], axis=0)


def from_sign_magnitude_planes(
    planes: ArrayLike,
    *,
    precision: int | None = None,
) -> NDArray[np.int32]:
    """Reconstruct integers using the sign and top ``precision - 1`` magnitude planes.

    ``precision`` counts the sign plane. For an 8-plane bundle, precision 4
    keeps the sign and magnitude positions 4, 5, and 6 while setting positions
    0 through 3 to zero.
    """

    encoded = np.asarray(planes, dtype=np.uint8)
    if encoded.ndim < 1 or encoded.shape[0] < 2:
        raise ValueError("planes must have a leading bit dimension of at least 2")
    if np.any((encoded != 0) & (encoded != 1)):
        raise ValueError("planes must be binary")

    bits = int(encoded.shape[0])
    _validate_bits(bits)
    selected = bits if precision is None else int(precision)
    if selected < 2 or selected > bits:
        raise ValueError(f"precision must be between 2 and {bits}")

    first_kept_magnitude = bits - selected
    magnitude = np.zeros(encoded.shape[1:], dtype=np.int64)
    for offset in range(first_kept_magnitude, bits - 1):
        magnitude += encoded[offset].astype(np.int64) << offset

    negative = encoded[-1].astype(bool)
    signed = np.where(negative, -magnitude, magnitude)
    return signed.astype(np.int32)

