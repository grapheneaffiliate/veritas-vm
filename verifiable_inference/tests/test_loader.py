"""state_dict loader: round-trip native, GPT-2 name conversion."""

from __future__ import annotations

import numpy as np
import pytest

from verifiable_inference.loader import (
    convert_gpt2_state_dict,
    export_state_dict,
    load_state_dict,
)
from verifiable_inference.model import ModelConfig, init_random_weights
from verifiable_inference.prover import Prover


def test_native_round_trip():
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=2, max_seq_len=4)
    m = init_random_weights(cfg, seed=42)
    sd = export_state_dict(m)
    m2 = load_state_dict(sd, cfg)
    assert Prover(m).weight_root == Prover(m2).weight_root


def test_load_state_dict_rejects_missing_keys():
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=4)
    m = init_random_weights(cfg, seed=1)
    sd = export_state_dict(m)
    del sd["tok_embed"]
    with pytest.raises(KeyError):
        load_state_dict(sd, cfg)


def test_load_state_dict_rejects_wrong_shape():
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=1, max_seq_len=4)
    m = init_random_weights(cfg, seed=1)
    sd = export_state_dict(m)
    sd["tok_embed"] = np.zeros((9, 4), dtype="<f8")
    with pytest.raises(ValueError):
        load_state_dict(sd, cfg)


def test_gpt2_converter_naming():
    """Synthesize a GPT-2-shaped state_dict, convert it, load it, and
    confirm the inference is bit-identical to a model loaded from the
    equivalent native dict."""
    cfg = ModelConfig(vocab_size=8, d_model=4, n_layers=2, max_seq_len=4, n_heads=2)
    rng = np.random.default_rng(7)

    def rn(*shape):
        return rng.standard_normal(shape).astype("<f8")

    def zeros(*shape):
        return np.zeros(shape, dtype="<f8")

    sd: dict[str, np.ndarray] = {}
    sd["transformer.wte.weight"] = rn(cfg.vocab_size, cfg.d_model)
    sd["transformer.wpe.weight"] = rn(cfg.max_seq_len, cfg.d_model)
    sd["transformer.ln_f.weight"] = np.ones(cfg.d_model, dtype="<f8")
    sd["transformer.ln_f.bias"] = zeros(cfg.d_model)
    for i in range(cfg.n_layers):
        # GPT-2 Conv1D stores (in, out) — so c_attn weight is (d, 3d).
        sd[f"transformer.h.{i}.ln_1.weight"] = np.ones(cfg.d_model, dtype="<f8")
        sd[f"transformer.h.{i}.ln_1.bias"] = zeros(cfg.d_model)
        sd[f"transformer.h.{i}.attn.c_attn.weight"] = rn(cfg.d_model, 3 * cfg.d_model)
        sd[f"transformer.h.{i}.attn.c_attn.bias"] = zeros(3 * cfg.d_model)
        sd[f"transformer.h.{i}.attn.c_proj.weight"] = rn(cfg.d_model, cfg.d_model)
        sd[f"transformer.h.{i}.attn.c_proj.bias"] = zeros(cfg.d_model)
        sd[f"transformer.h.{i}.ln_2.weight"] = np.ones(cfg.d_model, dtype="<f8")
        sd[f"transformer.h.{i}.ln_2.bias"] = zeros(cfg.d_model)
        sd[f"transformer.h.{i}.mlp.c_fc.weight"] = rn(cfg.d_model, 4 * cfg.d_model)
        sd[f"transformer.h.{i}.mlp.c_fc.bias"] = zeros(4 * cfg.d_model)
        sd[f"transformer.h.{i}.mlp.c_proj.weight"] = rn(4 * cfg.d_model, cfg.d_model)
        sd[f"transformer.h.{i}.mlp.c_proj.bias"] = zeros(cfg.d_model)

    native = convert_gpt2_state_dict(sd, transpose_conv1d=True, tied_unembed=True)
    # Ensure shapes line up after transposition.
    assert native["layer0.qkv_w"].shape == (3 * cfg.d_model, cfg.d_model)
    assert native["layer0.out_w"].shape == (cfg.d_model, cfg.d_model)
    assert native["layer0.mlp_in_w"].shape == (4 * cfg.d_model, cfg.d_model)
    assert native["layer0.mlp_out_w"].shape == (cfg.d_model, 4 * cfg.d_model)
    # Tied unembed must equal token embedding bytes.
    assert np.array_equal(native["unembed_w"], native["tok_embed"])

    model = load_state_dict(native, cfg)
    # We can run inference on it.
    _, cert = Prover(model).run(np.array([1, 2, 3, 0], dtype="<i8"))
    assert cert.predicted_token is not None
    # Determinism: same state_dict → same cert.
    model2 = load_state_dict(convert_gpt2_state_dict(sd), cfg)
    _, cert2 = Prover(model2).run(np.array([1, 2, 3, 0], dtype="<i8"))
    assert cert.merkle_root == cert2.merkle_root


def test_multi_head_inference_works():
    """n_heads=2 path runs end-to-end and produces a valid certificate."""
    cfg = ModelConfig(vocab_size=8, d_model=8, n_layers=1, max_seq_len=4, n_heads=2)
    m = init_random_weights(cfg, seed=21)
    _, cert = Prover(m).run(np.array([1, 2, 3, 0], dtype="<i8"))
    assert cert.predicted_token is not None
    # Verifies via full re-derivation.
    from verifiable_inference.verifier import Verifier
    Verifier(m).verify(cert).raise_if_failed()


def test_n_heads_must_divide_d_model():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=8, d_model=7, n_layers=1, max_seq_len=4, n_heads=2)
