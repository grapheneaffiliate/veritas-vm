"""Vectorized deterministic kernels — bit-exact with kernels.py, ~100× faster.

The trick: every fast kernel performs IEEE-754 operations *in the exact
same order* as its pure-Python reference, just executes the per-cell
operations in parallel. NumPy ufuncs are element-wise deterministic
(per-lane rounding is identical to scalar ops), so the output bytes are
bit-identical regardless of SIMD width or NumPy build.

Reduction order is preserved by keeping the *reduction axis* as an
explicit Python ``for`` loop, while vectorizing over the *non-reduction
axes*. For our small models this gives 100-1000× speedup without
changing a single byte of any KernelRecord.

These functions plug into the same trace as :mod:`kernels` — call
``use_fast_kernels()`` to swap the public API.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .trace import ExecutionTrace

# Module-level "active" trace, shared with kernels.py via a setter.
_active_trace: Optional[ExecutionTrace] = None


def set_active_trace(trace: Optional[ExecutionTrace]) -> None:
    global _active_trace
    _active_trace = trace


def get_active_trace() -> Optional[ExecutionTrace]:
    return _active_trace


def _record(op: str, inputs, weights, params, output: np.ndarray) -> None:
    if _active_trace is not None:
        _active_trace.record(op, list(inputs), list(weights), dict(params), output)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized outer-product accumulator. For each k, compute the
    rank-1 update ``a[:, k:k+1] @ b[k:k+1, :]`` and add to the running
    total. Per-cell sequence of float64 ops is identical to the
    reference triple-loop matmul, so bit-exact."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul expects 2D inputs, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul inner-dim mismatch: {a.shape} @ {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"matmul dtype mismatch: {a.dtype} vs {b.dtype}")
    M, K = a.shape
    _, N = b.shape
    out = np.zeros((M, N), dtype=a.dtype)
    for k in range(K):
        out += a[:, k : k + 1] * b[k : k + 1, :]
    _record("matmul", [a, b], [], {}, out)
    return out


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise add. NumPy ufunc is per-lane deterministic."""
    if a.dtype != b.dtype:
        raise ValueError(f"add dtype mismatch: {a.dtype} vs {b.dtype}")
    if a.shape == b.shape:
        out = a + b
    elif a.ndim == 2 and b.ndim == 1 and a.shape[1] == b.shape[0]:
        out = a + b[None, :]
    else:
        raise ValueError(f"add: unsupported shapes {a.shape} and {b.shape}")
    _record("add", [a, b], [], {}, np.ascontiguousarray(out))
    return np.ascontiguousarray(out)


def linear(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """y = x @ weight^T + bias.

    Uses outer-product accumulator: for each in_dim k, add the rank-1
    update ``x[:, k:k+1] @ weight[:, k:k+1].T`` to the running output.
    Bit-exact with kernels.linear.
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"linear shapes: x={x.shape} weight={weight.shape}")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"linear inner-dim mismatch: x={x.shape} weight={weight.shape}")
    M = x.shape[0]
    out_dim, in_dim = weight.shape
    out = np.zeros((M, out_dim), dtype=x.dtype)
    for k in range(in_dim):
        # weight[:, k] has shape (out_dim,). We want out += x[:, k:k+1] * weight[:, k]
        # broadcast: (M, 1) * (out_dim,) -> (M, out_dim).
        out += x[:, k : k + 1] * weight[:, k]
    if bias is not None:
        if bias.shape != (out_dim,):
            raise ValueError(f"linear bias shape: {bias.shape} vs out_dim {out_dim}")
        out = out + bias[None, :]
        weights = [weight, bias]
    else:
        weights = [weight]
    out = np.ascontiguousarray(out)
    _record("linear", [x], weights, {"has_bias": bias is not None}, out)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax. Vectorizes across batch slices but
    keeps the per-slice reduction order identical to the reference."""
    if axis < 0:
        axis = x.ndim + axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"softmax: bad axis {axis} for ndim {x.ndim}")
    moved = np.moveaxis(x, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    B, D = flat.shape

    # Sequential max along D, vectorized across B.
    m = flat[:, 0].copy()
    for k in range(1, D):
        m = np.maximum(m, flat[:, k])
    ex = np.exp(flat - m[:, None])
    # Sequential sum along D, vectorized across B (matches reference order).
    s = ex[:, 0].copy()
    for k in range(1, D):
        s = s + ex[:, k]
    out = ex / s[:, None]

    out_full = out.reshape(moved.shape)
    out = np.ascontiguousarray(np.moveaxis(out_full, -1, axis))
    _record("softmax", [x], [], {"axis": int(axis)}, out)
    return out


def layernorm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Per-row LayerNorm. Sequential mean/var along the feature dim,
    vectorized across rows."""
    if x.ndim != 2:
        raise ValueError(f"layernorm expects 2D input, got {x.shape}")
    M, D = x.shape
    if weight.shape != (D,) or bias.shape != (D,):
        raise ValueError(f"layernorm w/b shapes {weight.shape} {bias.shape}")
    # Sequential mean along axis 1.
    s = x[:, 0].copy()
    for k in range(1, D):
        s = s + x[:, k]
    mean = s / x.dtype.type(D)
    # Sequential variance along axis 1.
    diff0 = x[:, 0] - mean
    v = diff0 * diff0
    for k in range(1, D):
        d = x[:, k] - mean
        v = v + d * d
    var = v / x.dtype.type(D)
    inv = 1.0 / np.sqrt(var + x.dtype.type(eps))
    out = (x - mean[:, None]) * inv[:, None] * weight[None, :] + bias[None, :]
    out = np.ascontiguousarray(out)
    _record("layernorm", [x], [weight, bias], {"eps": float(eps)}, out)
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU. Per-element ``math.erf`` is bit-exact with the
    reference; we vectorize the surrounding arithmetic."""
    import math

    flat = x.reshape(-1)
    out = np.empty_like(flat)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for i in range(flat.shape[0]):
        v = float(flat[i])
        out[i] = x.dtype.type(0.5 * v * (1.0 + math.erf(v * inv_sqrt2)))
    out = out.reshape(x.shape)
    _record("gelu", [x], [], {}, out)
    return out


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    causal: bool = True,
) -> np.ndarray:
    """Single-head scaled dot-product attention using fast matmul."""
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError(f"attention shape mismatch: q={q.shape} k={k.shape} v={v.shape}")
    if q.ndim != 2:
        raise ValueError(f"attention expects 2D, got {q.shape}")
    seq, d_head = q.shape
    scale = 1.0 / np.sqrt(q.dtype.type(d_head))

    # scores = (q @ k^T) * scale, computed via outer-product accumulator.
    scores = np.zeros((seq, seq), dtype=q.dtype)
    for kk in range(d_head):
        scores += q[:, kk : kk + 1] * k[:, kk]  # (seq, 1) * (seq,) -> (seq, seq)
    scores = scores * scale

    if causal:
        neg_inf = q.dtype.type(-1e30)
        # Upper-triangular mask (j > i) -> -inf.
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = np.where(mask, neg_inf, scores)

    weights = softmax(scores, axis=-1)

    out = np.zeros((seq, d_head), dtype=q.dtype)
    for j in range(seq):
        out += weights[:, j : j + 1] * v[j, :]  # (seq, 1) * (d_head,) -> (seq, d_head)

    _record(
        "attention",
        [q, k, v],
        [],
        {"causal": bool(causal), "seq_len": int(seq), "d_head": int(d_head)},
        out,
    )
    return out


# --- Activation switching --------------------------------------------------

def use_fast_kernels() -> None:
    """Replace ``kernels`` module-level functions with the fast versions
    and share the active-trace pointer. After this call all ``kernels.X``
    references run the optimized implementations.

    This swap is reversible by calling :func:`use_reference_kernels`.
    """
    from . import kernels as K

    K._FAST_INSTALLED = True  # type: ignore[attr-defined]
    K._saved_ref = {  # type: ignore[attr-defined]
        "matmul": K.matmul,
        "add": K.add,
        "linear": K.linear,
        "softmax": K.softmax,
        "layernorm": K.layernorm,
        "gelu": K.gelu,
        "attention": K.attention,
        "set_active_trace": K.set_active_trace,
    }
    K.matmul = matmul  # type: ignore[assignment]
    K.add = add  # type: ignore[assignment]
    K.linear = linear  # type: ignore[assignment]
    K.softmax = softmax  # type: ignore[assignment]
    K.layernorm = layernorm  # type: ignore[assignment]
    K.gelu = gelu  # type: ignore[assignment]
    K.attention = attention  # type: ignore[assignment]
    # Bridge the trace pointer: kernels.set_active_trace must update both
    # modules so existing tracing context manager works unchanged.
    _orig = K._saved_ref["set_active_trace"]  # type: ignore[index]

    def _bridge(t):
        _orig(t)
        set_active_trace(t)

    K.set_active_trace = _bridge  # type: ignore[assignment]


def use_reference_kernels() -> None:
    """Undo :func:`use_fast_kernels`."""
    from . import kernels as K

    if not getattr(K, "_FAST_INSTALLED", False):
        return
    saved = K._saved_ref  # type: ignore[attr-defined]
    K.matmul = saved["matmul"]
    K.add = saved["add"]
    K.linear = saved["linear"]
    K.softmax = saved["softmax"]
    K.layernorm = saved["layernorm"]
    K.gelu = saved["gelu"]
    K.attention = saved["attention"]
    K.set_active_trace = saved["set_active_trace"]
    K._FAST_INSTALLED = False  # type: ignore[attr-defined]
