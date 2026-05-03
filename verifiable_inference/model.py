"""Tiny GPT-style transformer that runs on the deterministic kernels.

Single-head attention, MLP with GELU, pre-LN. Small enough for the
pure-Python deterministic kernels to run a forward pass in well under a
second. Intended as a *complete worked example* of an end-to-end
deterministic inference, not a competitive language model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels as K


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    max_seq_len: int
    n_heads: int = 1
    dtype: str = "<f8"  # canonical little-endian float64

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return {
            "vocab_size": int(self.vocab_size),
            "d_model": int(self.d_model),
            "n_layers": int(self.n_layers),
            "max_seq_len": int(self.max_seq_len),
            "n_heads": int(self.n_heads),
            "dtype": str(self.dtype),
        }

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        return ModelConfig(
            vocab_size=int(d["vocab_size"]),
            d_model=int(d["d_model"]),
            n_layers=int(d["n_layers"]),
            max_seq_len=int(d["max_seq_len"]),
            n_heads=int(d.get("n_heads", 1)),
            dtype=str(d.get("dtype", "<f8")),
        )


@dataclass
class LayerWeights:
    ln1_w: np.ndarray
    ln1_b: np.ndarray
    qkv_w: np.ndarray  # (3*d_model, d_model)
    qkv_b: np.ndarray  # (3*d_model,)
    out_w: np.ndarray  # (d_model, d_model)
    out_b: np.ndarray  # (d_model,)
    ln2_w: np.ndarray
    ln2_b: np.ndarray
    mlp_in_w: np.ndarray  # (4*d_model, d_model)
    mlp_in_b: np.ndarray  # (4*d_model,)
    mlp_out_w: np.ndarray  # (d_model, 4*d_model)
    mlp_out_b: np.ndarray  # (d_model,)


@dataclass
class ModelWeights:
    config: ModelConfig
    tok_embed: np.ndarray  # (vocab, d_model)
    pos_embed: np.ndarray  # (max_seq, d_model)
    layers: list[LayerWeights]
    ln_f_w: np.ndarray
    ln_f_b: np.ndarray
    unembed_w: np.ndarray  # (vocab, d_model)
    unembed_b: np.ndarray  # (vocab,)

    def all_arrays(self) -> list[tuple[str, np.ndarray]]:
        """Flat (name, array) list in a deterministic order — used for hashing."""
        out: list[tuple[str, np.ndarray]] = []
        out.append(("tok_embed", self.tok_embed))
        out.append(("pos_embed", self.pos_embed))
        for i, layer in enumerate(self.layers):
            out.append((f"layer{i}.ln1_w", layer.ln1_w))
            out.append((f"layer{i}.ln1_b", layer.ln1_b))
            out.append((f"layer{i}.qkv_w", layer.qkv_w))
            out.append((f"layer{i}.qkv_b", layer.qkv_b))
            out.append((f"layer{i}.out_w", layer.out_w))
            out.append((f"layer{i}.out_b", layer.out_b))
            out.append((f"layer{i}.ln2_w", layer.ln2_w))
            out.append((f"layer{i}.ln2_b", layer.ln2_b))
            out.append((f"layer{i}.mlp_in_w", layer.mlp_in_w))
            out.append((f"layer{i}.mlp_in_b", layer.mlp_in_b))
            out.append((f"layer{i}.mlp_out_w", layer.mlp_out_w))
            out.append((f"layer{i}.mlp_out_b", layer.mlp_out_b))
        out.append(("ln_f_w", self.ln_f_w))
        out.append(("ln_f_b", self.ln_f_b))
        out.append(("unembed_w", self.unembed_w))
        out.append(("unembed_b", self.unembed_b))
        return out


def init_random_weights(config: ModelConfig, seed: int = 0) -> ModelWeights:
    """Reproducible random init. Two calls with the same config + seed give
    bit-identical weights."""
    rng = np.random.default_rng(seed)
    dt = np.dtype(config.dtype)

    def rn(shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(dt)

    def zeros(shape):
        return np.zeros(shape, dtype=dt)

    def ones(shape):
        return np.ones(shape, dtype=dt)

    layers = []
    for _ in range(config.n_layers):
        layers.append(
            LayerWeights(
                ln1_w=ones((config.d_model,)),
                ln1_b=zeros((config.d_model,)),
                qkv_w=rn((3 * config.d_model, config.d_model)),
                qkv_b=zeros((3 * config.d_model,)),
                out_w=rn((config.d_model, config.d_model)),
                out_b=zeros((config.d_model,)),
                ln2_w=ones((config.d_model,)),
                ln2_b=zeros((config.d_model,)),
                mlp_in_w=rn((4 * config.d_model, config.d_model)),
                mlp_in_b=zeros((4 * config.d_model,)),
                mlp_out_w=rn((config.d_model, 4 * config.d_model)),
                mlp_out_b=zeros((config.d_model,)),
            )
        )

    return ModelWeights(
        config=config,
        tok_embed=rn((config.vocab_size, config.d_model)),
        pos_embed=rn((config.max_seq_len, config.d_model)),
        layers=layers,
        ln_f_w=ones((config.d_model,)),
        ln_f_b=zeros((config.d_model,)),
        unembed_w=rn((config.vocab_size, config.d_model)),
        unembed_b=zeros((config.vocab_size,)),
    )


def _embed(tokens: np.ndarray, model: ModelWeights) -> np.ndarray:
    """Token + position embedding lookup. Recorded as a single 'embed' op."""
    seq = tokens.shape[0]
    if seq > model.config.max_seq_len:
        raise ValueError(f"sequence length {seq} > max {model.config.max_seq_len}")
    dt = np.dtype(model.config.dtype)
    out = np.empty((seq, model.config.d_model), dtype=dt)
    for i in range(seq):
        tok = int(tokens[i])
        if tok < 0 or tok >= model.config.vocab_size:
            raise ValueError(f"token {tok} out of vocab {model.config.vocab_size}")
        for d in range(model.config.d_model):
            out[i, d] = model.tok_embed[tok, d] + model.pos_embed[i, d]
    K._record(
        "embed",
        [tokens.astype("<i8")],
        [model.tok_embed, model.pos_embed],
        {"seq_len": int(seq)},
        out,
    )
    return out


def _attention_block(
    x: np.ndarray, layer: LayerWeights, d_model: int, n_heads: int
) -> np.ndarray:
    """Pre-LN multi-head self-attention block. ``n_heads == 1`` recovers
    the single-head path bit-for-bit."""
    norm = K.layernorm(x, layer.ln1_w, layer.ln1_b)
    qkv = K.linear(norm, layer.qkv_w, layer.qkv_b)
    q_full = qkv[:, 0:d_model]
    k_full = qkv[:, d_model : 2 * d_model]
    v_full = qkv[:, 2 * d_model : 3 * d_model]

    seq = x.shape[0]
    d_head = d_model // n_heads
    head_outs: list[np.ndarray] = []
    for h in range(n_heads):
        s = h * d_head
        e = s + d_head
        q_h = np.ascontiguousarray(q_full[:, s:e])
        k_h = np.ascontiguousarray(k_full[:, s:e])
        v_h = np.ascontiguousarray(v_full[:, s:e])
        head_outs.append(K.attention(q_h, k_h, v_h, causal=True))
    if n_heads == 1:
        attn_out = head_outs[0]
    else:
        attn_out = np.empty((seq, d_model), dtype=x.dtype)
        for h, ho in enumerate(head_outs):
            attn_out[:, h * d_head : (h + 1) * d_head] = ho
    proj = K.linear(attn_out, layer.out_w, layer.out_b)
    return K.add(x, proj)


def _mlp_block(x: np.ndarray, layer: LayerWeights) -> np.ndarray:
    norm = K.layernorm(x, layer.ln2_w, layer.ln2_b)
    h = K.linear(norm, layer.mlp_in_w, layer.mlp_in_b)
    h = K.gelu(h)
    h = K.linear(h, layer.mlp_out_w, layer.mlp_out_b)
    return K.add(x, h)


def forward(tokens: np.ndarray, model: ModelWeights) -> np.ndarray:
    """Run a forward pass. Returns logits of shape ``(seq_len, vocab_size)``.

    If a trace is active (see ``kernels.tracing``), every kernel call is
    recorded.
    """
    if tokens.ndim != 1:
        raise ValueError(f"forward expects 1D token array, got {tokens.shape}")
    x = _embed(tokens, model)
    for layer in model.layers:
        x = _attention_block(x, layer, model.config.d_model, model.config.n_heads)
        x = _mlp_block(x, layer)
    x = K.layernorm(x, model.ln_f_w, model.ln_f_b)
    logits = K.linear(x, model.unembed_w, model.unembed_b)
    return logits


def greedy_next_token(logits: np.ndarray) -> int:
    """Argmax over the last-token row. Tie-break: smallest token id wins
    (deterministic across NumPy versions)."""
    last = logits[-1]
    best_idx = 0
    best_val = last[0]
    for j in range(1, last.shape[0]):
        if last[j] > best_val:
            best_val = last[j]
            best_idx = j
    return int(best_idx)
