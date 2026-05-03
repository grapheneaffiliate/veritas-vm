"""Kernel correctness vs. NumPy reference; trace recording works."""

from __future__ import annotations

import numpy as np

from verifiable_inference import kernels as K
from verifiable_inference.trace import ExecutionTrace


def test_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 3)).astype(np.float64)
    b = rng.standard_normal((3, 5)).astype(np.float64)
    got = K.matmul(a, b)
    expect = a @ b
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-12)


def test_linear_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 6)).astype(np.float64)
    w = rng.standard_normal((3, 6)).astype(np.float64)
    bias = rng.standard_normal((3,)).astype(np.float64)
    got = K.linear(x, w, bias)
    expect = x @ w.T + bias
    np.testing.assert_allclose(got, expect, rtol=1e-12, atol=1e-12)


def test_softmax_normalizes():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 7)).astype(np.float64)
    got = K.softmax(x, axis=-1)
    np.testing.assert_allclose(got.sum(axis=-1), 1.0, rtol=1e-12, atol=1e-12)


def test_layernorm_zero_mean_unit_var():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 8)).astype(np.float64)
    w = np.ones(8, dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    out = K.layernorm(x, w, b, eps=1e-12)
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=1e-6)


def test_gelu_matches_reference():
    x = np.linspace(-3.0, 3.0, 7, dtype=np.float64)
    out = K.gelu(x)
    import math

    expected = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x])
    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=1e-14)


def test_attention_causal_mask_blocks_future():
    rng = np.random.default_rng(4)
    seq, d = 4, 3
    q = rng.standard_normal((seq, d)).astype(np.float64)
    k = rng.standard_normal((seq, d)).astype(np.float64)
    v = np.eye(seq, d, dtype=np.float64) if seq <= d else rng.standard_normal((seq, d))
    out = K.attention(q, k, v, causal=True)
    # First row is fully determined by v[0]; nothing from positions >= 1 can leak in.
    np.testing.assert_allclose(out[0], v[0], rtol=1e-12, atol=1e-12)


def test_kernels_record_when_tracing():
    trace = ExecutionTrace()
    a = np.ones((2, 2), dtype=np.float64)
    b = np.ones((2, 2), dtype=np.float64)
    with K.tracing(trace):
        K.matmul(a, b)
        K.add(a, b)
    assert len(trace.records) == 2
    assert trace.records[0].op == "matmul"
    assert trace.records[1].op == "add"


def test_kernels_silent_outside_tracing():
    a = np.ones((2, 2), dtype=np.float64)
    b = np.ones((2, 2), dtype=np.float64)
    K.matmul(a, b)  # no active trace, must not raise
