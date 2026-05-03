"""Multi-token autoregressive generation: per-step certs chained into a transcript."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from verifiable_inference import signatures as ed25519
from verifiable_inference.model import ModelConfig, init_random_weights
from verifiable_inference.prover import Prover
from verifiable_inference.session import (
    GENESIS_SESSION_HASH,
    SessionTranscript,
    StepCertificate,
    generate,
    load_transcript,
    save_transcript,
    verify_step_inclusion,
    verify_transcript,
)


def _tiny():
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=8)
    return cfg, init_random_weights(cfg, seed=11)


def test_generate_three_tokens_produces_three_steps():
    _, model = _tiny()
    prover = Prover(model)
    prompt = np.array([1, 2, 3], dtype="<i8")
    t = generate(prover, prompt, max_new_tokens=3, include_full_trace=False)
    assert len(t.steps) == 3
    assert len(t.generated_tokens) == 3
    assert len(t.all_tokens) == 6


def test_session_chain_links_correct():
    _, model = _tiny()
    t = generate(Prover(model), np.array([0, 1], dtype="<i8"), max_new_tokens=4)
    parent = GENESIS_SESSION_HASH
    for step in t.steps:
        assert step.parent_session_hash == parent
        parent = step.session_hash
    assert t.final_session_hash == parent


def test_transcript_verify_succeeds_clean():
    _, model = _tiny()
    sk, pk = ed25519.generate_keypair(seed=b"\x33" * 32)
    t = generate(
        Prover(model),
        np.array([0, 1, 2], dtype="<i8"),
        max_new_tokens=3,
        sign_key=sk,
        sign_algo="ed25519",
    )
    assert verify_transcript(t, expected_weight_root=Prover(model).weight_root, public_key=pk)


def test_transcript_verify_rejects_wrong_weight_root():
    _, model = _tiny()
    t = generate(Prover(model), np.array([0, 1], dtype="<i8"), max_new_tokens=2)
    assert not verify_transcript(t, expected_weight_root="0" * 64)


def test_transcript_verify_rejects_tampered_step():
    _, model = _tiny()
    sk, _ = ed25519.generate_keypair(seed=b"\x33" * 32)
    t = generate(
        Prover(model),
        np.array([0, 1, 2], dtype="<i8"),
        max_new_tokens=3,
        sign_key=sk,
        sign_algo="ed25519",
    )
    # Mutate one step's predicted token AFTER signing — signature should reject.
    d = t.to_dict()
    d["steps"][1]["cert"]["output"]["predicted_token"] = (
        d["steps"][1]["cert"]["output"]["predicted_token"] + 1
    ) % 8
    bad = SessionTranscript.from_dict(d)
    assert not verify_transcript(bad)


def test_transcript_verify_rejects_dropped_step():
    _, model = _tiny()
    t = generate(Prover(model), np.array([0, 1, 2], dtype="<i8"), max_new_tokens=4)
    d = t.to_dict()
    d["steps"] = d["steps"][:-1]  # drop final step
    # Don't update transcript_root or final_session_hash — they must mismatch.
    bad = SessionTranscript.from_dict(d)
    assert not verify_transcript(bad)


def test_transcript_verify_rejects_swapped_steps():
    _, model = _tiny()
    t = generate(Prover(model), np.array([0, 1, 2], dtype="<i8"), max_new_tokens=4)
    d = t.to_dict()
    d["steps"][1], d["steps"][2] = d["steps"][2], d["steps"][1]
    bad = SessionTranscript.from_dict(d)
    assert not verify_transcript(bad)


def test_step_inclusion_proof():
    _, model = _tiny()
    t = generate(Prover(model), np.array([0, 1, 2], dtype="<i8"), max_new_tokens=5)
    for i in range(len(t.steps)):
        step, proof = t.step_proof(i)
        assert verify_step_inclusion(t.transcript_root, step, proof)


def test_step_inclusion_proof_wrong_root_rejected():
    _, model = _tiny()
    t = generate(Prover(model), np.array([0, 1, 2], dtype="<i8"), max_new_tokens=3)
    step, proof = t.step_proof(1)
    assert not verify_step_inclusion(b"\x00" * 32, step, proof)


def test_transcript_round_trip_json():
    _, model = _tiny()
    t = generate(Prover(model), np.array([0, 1, 2], dtype="<i8"), max_new_tokens=2)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "transcript.json")
        save_transcript(t, p)
        loaded = load_transcript(p)
        assert loaded.transcript_root == t.transcript_root
        assert loaded.final_session_hash == t.final_session_hash
        assert len(loaded.steps) == len(t.steps)
        assert verify_transcript(loaded)


def test_generation_is_deterministic():
    _, model = _tiny()
    prompt = np.array([0, 1, 2], dtype="<i8")
    t1 = generate(Prover(model), prompt, max_new_tokens=3)
    t2 = generate(Prover(model), prompt, max_new_tokens=3)
    assert t1.transcript_root == t2.transcript_root
    assert t1.final_session_hash == t2.final_session_hash
    assert t1.generated_tokens == t2.generated_tokens
