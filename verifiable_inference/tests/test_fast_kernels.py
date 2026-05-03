"""Fast kernels must be bit-exact with the reference kernels."""

from __future__ import annotations

import numpy as np
import pytest

from verifiable_inference import fast_kernels as F
from verifiable_inference import kernels as K
from verifiable_inference.canonical import hash_array
from verifiable_inference.fast_kernels import use_fast_kernels, use_reference_kernels
from verifiable_inference.model import ModelConfig, init_random_weights
from verifiable_inference.prover import Prover
from verifiable_inference.verifier import Verifier


@pytest.fixture
def reset_kernels():
    use_reference_kernels()
    yield
    use_reference_kernels()


def test_fast_matmul_bit_exact():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 5))
    b = rng.standard_normal((5, 7))
    ref = K.matmul(a, b)
    fast = F.matmul(a, b)
    assert hash_array(ref) == hash_array(fast)


def test_fast_linear_bit_exact():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 8))
    w = rng.standard_normal((4, 8))
    bias = rng.standard_normal((4,))
    ref = K.linear(x, w, bias)
    fast = F.linear(x, w, bias)
    assert hash_array(ref) == hash_array(fast)


def test_fast_softmax_bit_exact():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 16))
    assert hash_array(K.softmax(x)) == hash_array(F.softmax(x))


def test_fast_layernorm_bit_exact():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 8))
    w = rng.standard_normal((8,))
    b = rng.standard_normal((8,))
    assert hash_array(K.layernorm(x, w, b, eps=1e-5)) == hash_array(F.layernorm(x, w, b, eps=1e-5))


def test_fast_gelu_bit_exact():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 7))
    assert hash_array(K.gelu(x)) == hash_array(F.gelu(x))


def test_fast_attention_bit_exact():
    rng = np.random.default_rng(5)
    seq, d = 4, 6
    q = rng.standard_normal((seq, d))
    k = rng.standard_normal((seq, d))
    v = rng.standard_normal((seq, d))
    assert hash_array(K.attention(q, k, v, causal=True)) == hash_array(F.attention(q, k, v, causal=True))


def test_fast_certificate_bit_exact_with_reference(reset_kernels):
    """The certificate produced under fast kernels must equal the one
    produced under reference kernels — same bytes, same merkle root,
    same chain head, byte-identical full trace."""
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=2, max_seq_len=4)
    model = init_random_weights(cfg, seed=99)
    tokens = np.array([1, 2, 3, 0], dtype="<i8")

    use_reference_kernels()
    _, ref_cert = Prover(model).run(tokens, include_full_trace=True)

    use_fast_kernels()
    try:
        _, fast_cert = Prover(model).run(tokens, include_full_trace=True)
    finally:
        use_reference_kernels()

    assert ref_cert.merkle_root == fast_cert.merkle_root
    assert ref_cert.chain_head == fast_cert.chain_head
    assert ref_cert.output_logits_hash == fast_cert.output_logits_hash
    # Trace structure identical too.
    assert len(ref_cert.full_trace) == len(fast_cert.full_trace)
    for r, f in zip(ref_cert.full_trace, fast_cert.full_trace):
        assert r["op"] == f["op"]
        assert r["output_hash"] == f["output_hash"]


def test_fast_kernels_verifier_round_trip(reset_kernels):
    """Cert produced by fast kernels must pass full re-derivation
    verification using fast kernels."""
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=4)
    model = init_random_weights(cfg, seed=33)
    tokens = np.array([1, 2, 3, 0], dtype="<i8")

    use_fast_kernels()
    try:
        _, cert = Prover(model).run(tokens)
        Verifier(model).verify(cert).raise_if_failed()
    finally:
        use_reference_kernels()


def test_use_reference_is_idempotent():
    use_reference_kernels()
    use_reference_kernels()  # must not raise


def test_speed_smoke(reset_kernels):
    """Soft check that fast kernels finish in reasonable time on a model
    that is sluggish under the reference loops."""
    import time

    cfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2, max_seq_len=8, n_heads=4)
    model = init_random_weights(cfg, seed=2)
    tokens = np.array([1, 2, 3, 0, 1, 2, 3, 0], dtype="<i8")

    use_fast_kernels()
    try:
        t0 = time.perf_counter()
        _, cert = Prover(model).run(tokens, include_full_trace=False)
        dt = time.perf_counter() - t0
    finally:
        use_reference_kernels()
    # 2-second ceiling — generous, but the slow kernels would not finish.
    assert dt < 4.0, f"fast kernels too slow: {dt:.2f}s"
    assert cert.predicted_token is not None
