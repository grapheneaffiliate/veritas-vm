"""Deterministic kernels with automatic trace recording.

Every kernel here:
  - takes plain ``np.ndarray`` inputs
  - performs IEEE-754 ops in a fixed, single-threaded reduction order
    so results are bit-exact across platforms and NumPy builds
  - if a trace is active (set via ``set_active_trace``), appends a
    ``KernelRecord`` capturing canonical hashes of inputs / weights /
    params / output

The kernel implementations are intentionally simple Python loops. This is
an honest design choice for the demo: it makes determinism trivial to
audit. A production implementation would replace these with deterministic
SIMD/GPU kernels emitting the same Merkle leaves.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import numpy as np

from .trace import ExecutionTrace

# Module-level "active" trace. Kernel calls outside any active trace are
# still computed correctly — they just aren't recorded.
_active_trace: Optional[ExecutionTrace] = None


def set_active_trace(trace: Optional[ExecutionTrace]) -> None:
    global _active_trace
    _active_trace = trace


def get_active_trace() -> Optional[ExecutionTrace]:
    return _active_trace


@contextmanager
def tracing(trace: ExecutionTrace):
    """Context manager: install ``trace`` as the active recorder for its body."""
    prev = _active_trace
    set_active_trace(trace)
    try:
        yield trace
    finally:
        set_active_trace(prev)


def _record(op: str, inputs, weights, params, output: np.ndarray) -> None:
    if _active_trace is not None:
        _active_trace.record(op, list(inputs), list(weights), dict(params), output)


# ---------------------------------------------------------------------------
# Linear algebra primitives — explicit reduction order, bit-exact.
# ---------------------------------------------------------------------------

def _det_dot1d(a: np.ndarray, b: np.ndarray) -> np.floating:
    """Sequential left-to-right dot product. Bit-exact across platforms."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"_det_dot1d shape mismatch: {a.shape} vs {b.shape}")
    s = a.dtype.type(0)
    for k in range(a.shape[0]):
        s = s + a[k] * b[k]
    return s


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Deterministic ``a @ b`` for 2D inputs."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul expects 2D inputs, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul inner-dim mismatch: {a.shape} @ {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"matmul dtype mismatch: {a.dtype} vs {b.dtype}")
    M, K = a.shape
    _, N = b.shape
    out = np.empty((M, N), dtype=a.dtype)
    for i in range(M):
        for j in range(N):
            out[i, j] = _det_dot1d(a[i], b[:, j])
    _record("matmul", [a, b], [], {}, out)
    return out


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add with broadcasting along leading axes (rows of ``a``)."""
    if a.dtype != b.dtype:
        raise ValueError(f"add dtype mismatch: {a.dtype} vs {b.dtype}")
    out = np.empty_like(a)
    if a.shape == b.shape:
        flat_a = a.reshape(-1)
        flat_b = b.reshape(-1)
        flat_o = out.reshape(-1)
        for i in range(flat_a.shape[0]):
            flat_o[i] = flat_a[i] + flat_b[i]
    elif a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0]:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out[i, j] = a[i, j] + b[j]
    else:
        raise ValueError(f"add: unsupported shapes {a.shape} and {b.shape}")
    _record("add", [a, b], [], {}, out)
    return out


def linear(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """y = x @ weight^T + bias.

    ``weight`` shape: ``(out_dim, in_dim)`` (PyTorch convention).
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"linear shapes: x={x.shape} weight={weight.shape}")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"linear inner-dim mismatch: x={x.shape} weight={weight.shape}")
    M = x.shape[0]
    out_dim, in_dim = weight.shape
    out = np.empty((M, out_dim), dtype=x.dtype)
    for i in range(M):
        for j in range(out_dim):
            out[i, j] = _det_dot1d(x[i], weight[j])
    if bias is not None:
        if bias.shape != (out_dim,):
            raise ValueError(f"linear bias shape: {bias.shape} vs out_dim {out_dim}")
        for i in range(M):
            for j in range(out_dim):
                out[i, j] = out[i, j] + bias[j]
        weights = [weight, bias]
    else:
        weights = [weight]
    _record("linear", [x], weights, {"has_bias": bias is not None}, out)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along one axis. Deterministic.

    Stability shift uses sequential max; sum uses sequential left-to-right add.
    """
    if axis < 0:
        axis = x.ndim + axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"softmax: bad axis {axis} for ndim {x.ndim}")
    out = np.empty_like(x)

    # Iterate over every "slice" along ``axis``.
    moved = np.moveaxis(x, axis, -1)
    out_moved = np.moveaxis(out, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    flat_out = out_moved.reshape(-1, moved.shape[-1])
    for i in range(flat.shape[0]):
        row = flat[i]
        # sequential max
        m = row[0]
        for k in range(1, row.shape[0]):
            if row[k] > m:
                m = row[k]
        # exp(row - m), then sequential sum
        ex = np.empty_like(row)
        for k in range(row.shape[0]):
            ex[k] = np.exp(row[k] - m)
        s = ex.dtype.type(0)
        for k in range(ex.shape[0]):
            s = s + ex[k]
        for k in range(ex.shape[0]):
            flat_out[i, k] = ex[k] / s
    _record("softmax", [x], [], {"axis": int(axis)}, out)
    return out


def layernorm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Per-row LayerNorm: ``(x - mean) / sqrt(var + eps) * weight + bias``.

    Mean and variance use sequential summation. Variance uses the unbiased=False
    form (sum of squares / n) consistent with PyTorch's ``nn.LayerNorm``.
    """
    if x.ndim != 2:
        raise ValueError(f"layernorm expects 2D input, got {x.shape}")
    M, D = x.shape
    if weight.shape != (D,) or bias.shape != (D,):
        raise ValueError(
            f"layernorm weight/bias shapes: {weight.shape}, {bias.shape}; expected ({D},)"
        )
    out = np.empty_like(x)
    for i in range(M):
        # sequential mean
        s = x.dtype.type(0)
        for k in range(D):
            s = s + x[i, k]
        mean = s / x.dtype.type(D)
        # sequential variance
        v = x.dtype.type(0)
        for k in range(D):
            d = x[i, k] - mean
            v = v + d * d
        var = v / x.dtype.type(D)
        inv = 1.0 / np.sqrt(var + x.dtype.type(eps))
        for k in range(D):
            out[i, k] = (x[i, k] - mean) * inv * weight[k] + bias[k]
    _record("layernorm", [x], [weight, bias], {"eps": float(eps)}, out)
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU: ``0.5 * x * (1 + erf(x / sqrt(2)))``.

    Uses ``math.erf`` per element to keep behaviour bit-exact (NumPy's
    ufunc may use SIMD-vectorized erf internally).
    """
    import math

    out = np.empty_like(x)
    flat_x = x.reshape(-1)
    flat_o = out.reshape(-1)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(flat_x.shape[0]):
        v = float(flat_x[i])
        flat_o[i] = x.dtype.type(0.5 * v * (1.0 + math.erf(v * inv_sqrt2)))
    _record("gelu", [x], [], {}, out)
    return out


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """Single-head scaled dot-product attention over a single sequence.

    Shapes: q, k, v all ``(seq_len, d_head)``. Returns ``(seq_len, d_head)``.
    """
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError(f"attention shape mismatch: q={q.shape} k={k.shape} v={v.shape}")
    if q.ndim != 2:
        raise ValueError(f"attention expects 2D, got {q.shape}")
    seq, d_head = q.shape
    scale = 1.0 / np.sqrt(q.dtype.type(d_head))

    # scores = q @ k^T, deterministic
    scores = np.empty((seq, seq), dtype=q.dtype)
    for i in range(seq):
        for j in range(seq):
            scores[i, j] = _det_dot1d(q[i], k[j]) * scale

    if causal:
        neg_inf = q.dtype.type(-1e30)
        for i in range(seq):
            for j in range(i + 1, seq):
                scores[i, j] = neg_inf

    # softmax row-wise (use the recorded softmax kernel)
    weights = softmax(scores, axis=-1)

    # out = weights @ v, deterministic
    out = np.empty((seq, d_head), dtype=q.dtype)
    for i in range(seq):
        for d in range(d_head):
            out[i, d] = _det_dot1d(weights[i], v[:, d])

    _record(
        "attention",
        [q, k, v],
        [],
        {"causal": bool(causal), "seq_len": int(seq), "d_head": int(d_head)},
        out,
    )
    return out
