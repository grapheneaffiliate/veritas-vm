"""End-to-end prove-then-verify, plus systematic tampering detection."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from verifiable_inference.certificate import Certificate, load_certificate, save_certificate
from verifiable_inference.model import ModelConfig, init_random_weights
from verifiable_inference.prover import Prover
from verifiable_inference.verifier import VerificationError, Verifier, verify_certificate


def _tiny_model(seed: int = 7):
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=4)
    return cfg, init_random_weights(cfg, seed=seed)


def _input():
    return np.array([1, 2, 3, 0], dtype="<i8")


def test_prove_then_structural_verify():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())
    report = verify_certificate(cert)
    report.raise_if_failed()
    assert report.ok


def test_prove_then_full_rederivation_verify():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())
    report = Verifier(model).verify(cert)
    report.raise_if_failed()
    assert report.ok


def test_certificate_json_round_trip():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.json")
        save_certificate(cert, p)
        loaded = load_certificate(p)
        assert loaded.merkle_root == cert.merkle_root
        assert loaded.chain_head == cert.chain_head
        assert loaded.weight_root == cert.weight_root
        Verifier(model).verify(loaded).raise_if_failed()


def test_determinism_two_runs_identical():
    _, model = _tiny_model(seed=99)
    _, c1 = Prover(model).run(_input())
    _, c2 = Prover(model).run(_input())
    assert c1.merkle_root == c2.merkle_root
    assert c1.chain_head == c2.chain_head
    assert c1.output_logits_hash == c2.output_logits_hash
    assert c1.weight_root == c2.weight_root


def test_determinism_same_seed_different_models_identical_weight_root():
    _, m1 = _tiny_model(seed=11)
    _, m2 = _tiny_model(seed=11)
    p1 = Prover(m1)
    p2 = Prover(m2)
    assert p1.weight_root == p2.weight_root


def test_different_input_produces_different_certificate():
    _, model = _tiny_model()
    _, c1 = Prover(model).run(np.array([1, 2, 3, 0], dtype="<i8"))
    _, c2 = Prover(model).run(np.array([1, 2, 3, 4], dtype="<i8"))
    assert c1.merkle_root != c2.merkle_root
    assert c1.input_hash != c2.input_hash


def test_signature_round_trip():
    _, model = _tiny_model()
    key = b"super-secret"
    _, cert = Prover(model).run(_input(), sign_key=key, key_id="kid42")
    assert cert.signature["algo"] == "hmac-sha256"
    assert cert.signature["key_id"] == "kid42"
    Verifier(model).verify(cert, sign_key=key).raise_if_failed()


def test_signature_rejects_wrong_key():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input(), sign_key=b"correct-key")
    with pytest.raises(VerificationError) as ei:
        Verifier(model).verify(cert, sign_key=b"wrong-key").raise_if_failed()
    assert ei.value.code == "bad_signature"


# --- Tampering tests --------------------------------------------------------

def _tamper(cert: Certificate, fn) -> Certificate:
    d = cert.to_dict()
    fn(d)
    return Certificate.from_dict(d)


def test_tamper_output_hash_detected():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())
    bad = _tamper(cert, lambda d: d["output"].__setitem__("logits_hash", "0" * 64))
    with pytest.raises(VerificationError) as ei:
        Verifier(model).verify(bad).raise_if_failed()
    assert ei.value.code in {"output_hash_mismatch", "rerun_output_mismatch"}


def test_tamper_merkle_root_detected():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())
    bad = _tamper(cert, lambda d: d["trace"].__setitem__("merkle_root", "0" * 64))
    with pytest.raises(VerificationError) as ei:
        Verifier(model).verify(bad).raise_if_failed()
    assert ei.value.code == "merkle_root_mismatch"


def test_tamper_chain_head_detected():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())
    bad = _tamper(cert, lambda d: d["trace"].__setitem__("chain_head", "0" * 64))
    with pytest.raises(VerificationError) as ei:
        Verifier(model).verify(bad).raise_if_failed()
    assert ei.value.code == "chain_head_mismatch"


def test_tamper_drop_kernel_detected():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())

    def fn(d):
        d["full_trace"] = d["full_trace"][:-1]
        d["trace"]["n_kernels"] -= 1

    bad = _tamper(cert, fn)
    with pytest.raises(VerificationError):
        Verifier(model).verify(bad).raise_if_failed()


def test_tamper_swap_two_records_detected():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())

    def fn(d):
        ft = d["full_trace"]
        if len(ft) >= 2:
            ft[-1], ft[-2] = ft[-2], ft[-1]

    bad = _tamper(cert, fn)
    with pytest.raises(VerificationError):
        Verifier(model).verify(bad).raise_if_failed()


def test_tamper_input_token_detected():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())

    def fn(d):
        d["input"]["tokens"][0] = (d["input"]["tokens"][0] + 1) % 8

    bad = _tamper(cert, fn)
    with pytest.raises(VerificationError):
        Verifier(model).verify(bad).raise_if_failed()


def test_tamper_weight_detected_against_original_cert():
    _, model = _tiny_model()
    _, cert = Prover(model).run(_input())

    # Build a model with a single weight perturbed.
    cfg, bad_model = _tiny_model()
    bad_model.tok_embed[0, 0] = bad_model.tok_embed[0, 0] + 1e-12

    with pytest.raises(VerificationError) as ei:
        Verifier(bad_model).verify(cert).raise_if_failed()
    assert ei.value.code == "weight_root_mismatch"
