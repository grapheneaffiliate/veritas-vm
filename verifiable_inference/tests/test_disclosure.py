"""Selective disclosure: per-kernel inclusion proofs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from verifiable_inference.disclosure import (
    DisclosedKernel,
    compact_certificate,
    disclose_kernel,
    verify_disclosure,
)
from verifiable_inference.kernels import tracing
from verifiable_inference.model import ModelConfig, init_random_weights
from verifiable_inference.model import forward as model_forward
from verifiable_inference.prover import Prover
from verifiable_inference.trace import ExecutionTrace


def _trace_for(model, tokens):
    trace = ExecutionTrace()
    with tracing(trace):
        model_forward(tokens, model)
    return trace


def _tiny():
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=4)
    return cfg, init_random_weights(cfg, seed=7)


def test_disclose_each_kernel_verifies():
    _, model = _tiny()
    tokens = np.array([1, 2, 3, 0], dtype="<i8")
    trace = _trace_for(model, tokens)
    for i in range(len(trace.records)):
        disclosed = disclose_kernel(trace.records, i)
        assert verify_disclosure(disclosed)


def test_disclosure_binds_to_certificate_root():
    _, model = _tiny()
    tokens = np.array([1, 2, 3, 0], dtype="<i8")
    _, cert = Prover(model).run(tokens)
    trace = _trace_for(model, tokens)
    disclosed = disclose_kernel(trace.records, 2)
    expected = bytes.fromhex(cert.merkle_root)
    assert verify_disclosure(disclosed, expected_root=expected)


def test_disclosure_rejects_wrong_root():
    _, model = _tiny()
    trace = _trace_for(model, np.array([1, 2, 3, 0], dtype="<i8"))
    d = disclose_kernel(trace.records, 1)
    assert not verify_disclosure(d, expected_root=b"\x00" * 32)


def test_disclosure_rejects_tampered_record():
    _, model = _tiny()
    trace = _trace_for(model, np.array([1, 2, 3, 0], dtype="<i8"))
    d = disclose_kernel(trace.records, 1)
    # Mutate the disclosed record's op name — proof must reject.
    bad_record_dict = d.record.to_dict()
    bad_record_dict["op"] = "wrong_op"
    bad = DisclosedKernel.from_dict(
        {**d.to_dict(), "record": bad_record_dict},
    )
    assert not verify_disclosure(bad)


def test_disclosure_rejects_wrong_index_claim():
    _, model = _tiny()
    trace = _trace_for(model, np.array([1, 2, 3, 0], dtype="<i8"))
    d0 = disclose_kernel(trace.records, 0)
    d1 = disclose_kernel(trace.records, 1)
    # Try to claim record 1's content with record 0's proof.
    spliced = DisclosedKernel(record=d1.record, merkle_root=d0.merkle_root, proof=d0.proof)
    assert not verify_disclosure(spliced)


def test_disclosure_dict_round_trip():
    _, model = _tiny()
    trace = _trace_for(model, np.array([1, 2, 3, 0], dtype="<i8"))
    d = disclose_kernel(trace.records, 3)
    blob = json.dumps(d.to_dict())
    d2 = DisclosedKernel.from_dict(json.loads(blob))
    assert verify_disclosure(d2)


def test_compact_certificate_drops_full_trace():
    _, model = _tiny()
    _, cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"), include_full_trace=True
    )
    assert cert.full_trace is not None
    compact = compact_certificate(cert)
    assert compact.full_trace is None
    # Compact cert still has the Merkle root — disclosed kernels still bind.
    assert compact.merkle_root == cert.merkle_root
    assert compact.weight_root == cert.weight_root


def test_compact_certificate_with_disclosed_kernel():
    """The standard 'private trace' deployment: ship compact cert + one
    disclosed kernel for an audit query."""
    _, model = _tiny()
    tokens = np.array([1, 2, 3, 0], dtype="<i8")
    _, cert = Prover(model).run(tokens, include_full_trace=True)
    trace = _trace_for(model, tokens)

    compact = compact_certificate(cert)

    # Disclose only the very last kernel (the unembed) to an auditor.
    disclosed = disclose_kernel(trace.records, len(trace.records) - 1)

    # Auditor can verify the disclosed kernel binds to the compact cert.
    assert verify_disclosure(disclosed, expected_root=bytes.fromhex(compact.merkle_root))
    # And that the disclosed kernel's output matches the cert's output hash
    # (since it's the final kernel).
    assert disclosed.record.output_hash.hex() == compact.output_logits_hash
