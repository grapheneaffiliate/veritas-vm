"""Hash-chain integrity properties of ExecutionTrace."""

from __future__ import annotations

import numpy as np
import pytest

from verifiable_inference.trace import GENESIS_CHAIN_HASH, ExecutionTrace, KernelRecord


def _make_record(trace: ExecutionTrace, seq_marker: int) -> KernelRecord:
    a = np.full((2, 2), float(seq_marker), dtype=np.float64)
    b = np.full((2, 2), float(seq_marker + 1), dtype=np.float64)
    out = a + b
    return trace.record("op", [a], [b], {"marker": seq_marker}, out)


def test_chain_starts_at_genesis():
    t = ExecutionTrace()
    assert t.chain_head == GENESIS_CHAIN_HASH


def test_chain_advances_with_each_record():
    t = ExecutionTrace()
    seen = {t.chain_head}
    for i in range(5):
        _make_record(t, i)
        assert t.chain_head not in seen
        seen.add(t.chain_head)
    assert len(t.records) == 5


def test_replay_succeeds_on_clean_trace():
    t = ExecutionTrace()
    for i in range(4):
        _make_record(t, i)
    head = t.replay_chain()
    assert head == t.chain_head


def test_replay_detects_tampered_chain_hash():
    t = ExecutionTrace()
    for i in range(3):
        _make_record(t, i)
    # Mutate one record's chain_hash.
    bad = list(t.records)
    r = bad[1]
    bad[1] = KernelRecord(
        seq=r.seq,
        op=r.op,
        input_hashes=r.input_hashes,
        weight_hashes=r.weight_hashes,
        params=r.params,
        output_hash=r.output_hash,
        prev_chain_hash=r.prev_chain_hash,
        chain_hash=b"\xff" * 32,  # wrong
    )
    bad_trace = ExecutionTrace()
    bad_trace.records.extend(bad)
    with pytest.raises(ValueError):
        bad_trace.replay_chain()


def test_replay_detects_tampered_output_hash():
    t = ExecutionTrace()
    for i in range(3):
        _make_record(t, i)
    bad = list(t.records)
    r = bad[2]
    bad[2] = KernelRecord(
        seq=r.seq,
        op=r.op,
        input_hashes=r.input_hashes,
        weight_hashes=r.weight_hashes,
        params=r.params,
        output_hash=b"\x00" * 32,
        prev_chain_hash=r.prev_chain_hash,
        chain_hash=r.chain_hash,
    )
    bad_trace = ExecutionTrace()
    bad_trace.records.extend(bad)
    with pytest.raises(ValueError):
        bad_trace.replay_chain()


def test_record_dict_round_trip():
    t = ExecutionTrace()
    rec = _make_record(t, 42)
    rec2 = KernelRecord.from_dict(rec.to_dict())
    assert rec == rec2


def test_merkle_leaves_match_record_count():
    t = ExecutionTrace()
    for i in range(6):
        _make_record(t, i)
    leaves = t.merkle_leaves()
    assert len(leaves) == 6
    assert all(len(leaf) == 32 for leaf in leaves)
