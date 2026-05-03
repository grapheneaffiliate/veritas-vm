"""Model registry: pin weight_root → publisher-signed metadata."""

from __future__ import annotations

import numpy as np
import pytest

from verifiable_inference import signatures as ed25519
from verifiable_inference.model import ModelConfig, init_random_weights
from verifiable_inference.prover import Prover
from verifiable_inference.registry import (
    Registry,
    RegistryEntry,
    verify_certificate_against_registry,
)


def _entry_for(model, model_pk: bytes, *, attest_with: bytes | None = None) -> RegistryEntry:
    p = Prover(model)
    e = RegistryEntry(
        weight_root=p.weight_root,
        config=model.config.to_dict(),
        public_key=model_pk.hex(),
        name="demo-tinygpt",
        version="0.1.0",
        license="Apache-2.0",
    )
    if attest_with is not None:
        e.attest_ed25519(attest_with)
    return e


def _build_world(seed: int = 1):
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=4)
    model = init_random_weights(cfg, seed=seed)
    model_sk, model_pk = ed25519.generate_keypair(seed=b"\x10" * 32)
    pub_sk, pub_pk = ed25519.generate_keypair(seed=b"\x20" * 32)
    reg = Registry()
    reg.register(_entry_for(model, model_pk, attest_with=pub_sk))
    return model, model_sk, model_pk, pub_pk, reg


def test_attestation_round_trip():
    pub_sk, pub_pk = ed25519.generate_keypair(seed=b"\x05" * 32)
    cfg = ModelConfig(vocab_size=4, d_model=2, n_layers=1, max_seq_len=2)
    model = init_random_weights(cfg, seed=2)
    _, model_pk = ed25519.generate_keypair(seed=b"\x06" * 32)
    e = _entry_for(model, model_pk, attest_with=pub_sk)
    assert e.verify_attestation()
    assert e.verify_attestation(publisher_pk=pub_pk)


def test_attestation_rejects_wrong_publisher_pk():
    pub_sk, _ = ed25519.generate_keypair(seed=b"\x07" * 32)
    _, other_pk = ed25519.generate_keypair(seed=b"\x08" * 32)
    cfg = ModelConfig(vocab_size=4, d_model=2, n_layers=1, max_seq_len=2)
    model = init_random_weights(cfg, seed=3)
    _, model_pk = ed25519.generate_keypair(seed=b"\x09" * 32)
    e = _entry_for(model, model_pk, attest_with=pub_sk)
    assert not e.verify_attestation(publisher_pk=other_pk)


def test_registry_json_round_trip():
    model, _, model_pk, _, reg = _build_world()
    s = reg.to_json()
    reg2 = Registry.from_json(s)
    e = reg2.lookup(Prover(model).weight_root)
    assert e is not None
    assert e.public_key == model_pk.hex()
    assert e.verify_attestation()


def test_full_verification_flow_succeeds():
    model, model_sk, model_pk, pub_pk, reg = _build_world()
    _, cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"),
        sign_key=model_sk,
        sign_algo="ed25519",
    )
    ok, reason = verify_certificate_against_registry(
        cert, reg, trusted_publisher_pks=[pub_pk]
    )
    assert ok, reason


def test_full_verification_rejects_wrong_pubkey():
    model, model_sk, _model_pk, pub_pk, reg = _build_world()
    # Sign cert with a different model key — registry pin must reject.
    rogue_sk, _ = ed25519.generate_keypair(seed=b"\xaa" * 32)
    _, cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"),
        sign_key=rogue_sk,
        sign_algo="ed25519",
    )
    ok, reason = verify_certificate_against_registry(
        cert, reg, trusted_publisher_pks=[pub_pk]
    )
    assert not ok
    assert reason == "cert_pubkey_does_not_match_registry"


def test_full_verification_rejects_unknown_weight_root():
    cfg = ModelConfig(vocab_size=4, d_model=2, n_layers=1, max_seq_len=2)
    model = init_random_weights(cfg, seed=99)
    sk, _ = ed25519.generate_keypair(seed=b"\xbb" * 32)
    _, cert = Prover(model).run(np.array([0, 1], dtype="<i8"), sign_key=sk, sign_algo="ed25519")
    empty = Registry()
    ok, reason = verify_certificate_against_registry(cert, empty)
    assert not ok
    assert reason == "weight_root_not_in_registry"


def test_full_verification_rejects_untrusted_publisher():
    model, model_sk, _model_pk, _pub_pk, reg = _build_world()
    _, cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"),
        sign_key=model_sk,
        sign_algo="ed25519",
    )
    _, evil_pk = ed25519.generate_keypair(seed=b"\xcc" * 32)
    ok, reason = verify_certificate_against_registry(
        cert, reg, trusted_publisher_pks=[evil_pk]
    )
    assert not ok
    assert reason == "registry_publisher_untrusted"


def test_unsigned_cert_rejected():
    model, _, _, _, reg = _build_world()
    _, cert = Prover(model).run(np.array([1, 2, 3, 0], dtype="<i8"))
    ok, reason = verify_certificate_against_registry(cert, reg)
    assert not ok
    assert reason == "cert_not_ed25519_signed"


def test_double_register_rejected():
    model, _, model_pk, _, reg = _build_world()
    with pytest.raises(ValueError):
        reg.register(_entry_for(model, model_pk))
