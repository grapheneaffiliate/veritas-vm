"""Ed25519 signing/verification, integration with Certificate."""

from __future__ import annotations

import numpy as np
import pytest

from verifiable_inference import signatures as ed25519
from verifiable_inference.certificate import Certificate
from verifiable_inference.model import ModelConfig, init_random_weights
from verifiable_inference.prover import Prover
from verifiable_inference.verifier import VerificationError, Verifier


def test_ed25519_keypair_roundtrip():
    sk, pk = ed25519.generate_keypair(seed=b"\x42" * 32)
    assert len(sk) == 32 and len(pk) == 32
    msg = b"hello verifiable world"
    sig = ed25519.sign(msg, sk)
    assert len(sig) == 64
    assert ed25519.verify(msg, sig, pk)


def test_ed25519_rejects_wrong_message():
    sk, pk = ed25519.generate_keypair(seed=b"\x01" * 32)
    sig = ed25519.sign(b"original", sk)
    assert not ed25519.verify(b"tampered", sig, pk)


def test_ed25519_rejects_wrong_pubkey():
    sk1, pk1 = ed25519.generate_keypair(seed=b"\x01" * 32)
    _, pk2 = ed25519.generate_keypair(seed=b"\x02" * 32)
    sig = ed25519.sign(b"msg", sk1)
    assert ed25519.verify(b"msg", sig, pk1)
    assert not ed25519.verify(b"msg", sig, pk2)


def test_ed25519_rejects_garbage_signature():
    _, pk = ed25519.generate_keypair(seed=b"\x03" * 32)
    assert not ed25519.verify(b"msg", b"\x00" * 64, pk)
    assert not ed25519.verify(b"msg", b"too short", pk)


def test_ed25519_deterministic():
    """Ed25519 (per RFC 8032) is deterministic — same key + msg → same sig."""
    sk = b"\x7f" * 32
    sig1 = ed25519.sign(b"abc", sk)
    sig2 = ed25519.sign(b"abc", sk)
    assert sig1 == sig2


def _tiny():
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=4)
    return cfg, init_random_weights(cfg, seed=7)


def test_certificate_ed25519_signature():
    _, model = _tiny()
    sk, pk = ed25519.generate_keypair(seed=b"\x10" * 32)
    _, cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"),
        sign_key=sk,
        sign_algo="ed25519",
        key_id="model-vendor-key-1",
    )
    assert cert.signature["algo"] == "ed25519"
    assert cert.signature["public_key"] == pk.hex()
    # Verify without supplying any key (cert carries the pubkey).
    Verifier(model).verify(cert).raise_if_failed()
    # Pinning the correct key works.
    Verifier(model).verify(cert, sign_key=pk).raise_if_failed()


def test_certificate_ed25519_pinning_rejects_wrong_key():
    _, model = _tiny()
    sk, _ = ed25519.generate_keypair(seed=b"\x11" * 32)
    _, other_pk = ed25519.generate_keypair(seed=b"\x99" * 32)
    _, cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"),
        sign_key=sk,
        sign_algo="ed25519",
    )
    with pytest.raises(VerificationError) as ei:
        Verifier(model).verify(cert, sign_key=other_pk).raise_if_failed()
    assert ei.value.code == "bad_signature"


def test_certificate_tamper_invalidates_ed25519_signature():
    _, model = _tiny()
    sk, _ = ed25519.generate_keypair(seed=b"\x22" * 32)
    _, cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"),
        sign_key=sk,
        sign_algo="ed25519",
    )
    d = cert.to_dict()
    d["output"]["predicted_token"] = (d["output"]["predicted_token"] + 1) % 8
    bad = Certificate.from_dict(d)
    # Even before structural verification fires, the signature must reject.
    assert not bad.verify_signature()
