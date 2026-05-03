"""Load a state-dict-style weight blob into ModelWeights.

Two entry points:

  ``load_state_dict(state_dict, config)``
      Native format. Keys follow the names used by :class:`ModelWeights`:

          tok_embed, pos_embed, ln_f_w, ln_f_b, unembed_w, unembed_b
          layer{i}.ln1_w, layer{i}.ln1_b
          layer{i}.qkv_w, layer{i}.qkv_b
          layer{i}.out_w, layer{i}.out_b
          layer{i}.ln2_w, layer{i}.ln2_b
          layer{i}.mlp_in_w, layer{i}.mlp_in_b
          layer{i}.mlp_out_w, layer{i}.mlp_out_b

      Each value must be a ``np.ndarray`` of the right shape; missing
      keys raise ``KeyError`` and shape mismatches raise ``ValueError``.

  ``convert_gpt2_state_dict(sd)``
      Maps a HuggingFace / minGPT / nanoGPT GPT-2 state_dict into the
      native format above. Handles:
        - ``transformer.wte.weight``, ``transformer.wpe.weight``
        - ``transformer.h.{i}.ln_1.{weight,bias}``
        - ``transformer.h.{i}.attn.c_attn.{weight,bias}``
        - ``transformer.h.{i}.attn.c_proj.{weight,bias}``
        - ``transformer.h.{i}.ln_2.{weight,bias}``
        - ``transformer.h.{i}.mlp.c_fc.{weight,bias}``
        - ``transformer.h.{i}.mlp.c_proj.{weight,bias}``
        - ``transformer.ln_f.{weight,bias}``
        - ``lm_head.weight`` (or tied to ``wte`` if absent)

      GPT-2's `Conv1D` stores weights transposed vs. nn.Linear; a
      ``transpose_conv1d`` flag controls that. Default: True (matches HF).

The loader does NOT touch torch — pass it numpy arrays. To convert from
PyTorch::

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
"""

from __future__ import annotations

import re

import numpy as np

from .model import LayerWeights, ModelConfig, ModelWeights


def _to_canonical_dtype(arr: np.ndarray, target: str) -> np.ndarray:
    """Cast ``arr`` to the model's canonical dtype (default '<f8')."""
    return np.ascontiguousarray(arr.astype(np.dtype(target)))


def _require(sd: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in sd:
        raise KeyError(f"missing weight: {key!r}")
    return sd[key]


def _check_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise ValueError(
            f"weight {name!r} has shape {arr.shape}, expected {expected}"
        )


def load_state_dict(sd: dict[str, np.ndarray], config: ModelConfig) -> ModelWeights:
    """Build ``ModelWeights`` from a native-named state dict."""
    dt = config.dtype

    def fetch(key: str, expected: tuple[int, ...]) -> np.ndarray:
        arr = _to_canonical_dtype(_require(sd, key), dt)
        _check_shape(key, arr, expected)
        return arr

    layers: list[LayerWeights] = []
    for i in range(config.n_layers):
        layers.append(
            LayerWeights(
                ln1_w=fetch(f"layer{i}.ln1_w", (config.d_model,)),
                ln1_b=fetch(f"layer{i}.ln1_b", (config.d_model,)),
                qkv_w=fetch(f"layer{i}.qkv_w", (3 * config.d_model, config.d_model)),
                qkv_b=fetch(f"layer{i}.qkv_b", (3 * config.d_model,)),
                out_w=fetch(f"layer{i}.out_w", (config.d_model, config.d_model)),
                out_b=fetch(f"layer{i}.out_b", (config.d_model,)),
                ln2_w=fetch(f"layer{i}.ln2_w", (config.d_model,)),
                ln2_b=fetch(f"layer{i}.ln2_b", (config.d_model,)),
                mlp_in_w=fetch(f"layer{i}.mlp_in_w", (4 * config.d_model, config.d_model)),
                mlp_in_b=fetch(f"layer{i}.mlp_in_b", (4 * config.d_model,)),
                mlp_out_w=fetch(f"layer{i}.mlp_out_w", (config.d_model, 4 * config.d_model)),
                mlp_out_b=fetch(f"layer{i}.mlp_out_b", (config.d_model,)),
            )
        )

    return ModelWeights(
        config=config,
        tok_embed=fetch("tok_embed", (config.vocab_size, config.d_model)),
        pos_embed=fetch("pos_embed", (config.max_seq_len, config.d_model)),
        layers=layers,
        ln_f_w=fetch("ln_f_w", (config.d_model,)),
        ln_f_b=fetch("ln_f_b", (config.d_model,)),
        unembed_w=fetch("unembed_w", (config.vocab_size, config.d_model)),
        unembed_b=fetch("unembed_b", (config.vocab_size,)),
    )


# --- GPT-2 / HF naming converter --------------------------------------------

_GPT2_LAYER_RE = re.compile(r"transformer\.h\.(\d+)\.")


def _maybe_transpose_conv1d(arr: np.ndarray, do_transpose: bool) -> np.ndarray:
    """GPT-2's Conv1D stores weight as (in, out); nn.Linear is (out, in)."""
    return arr.T if (do_transpose and arr.ndim == 2) else arr


def convert_gpt2_state_dict(
    sd: dict[str, np.ndarray],
    *,
    transpose_conv1d: bool = True,
    tied_unembed: bool = True,
) -> dict[str, np.ndarray]:
    """Translate GPT-2 weight names into our native names.

    Returns a fresh dict; the input is not modified. Untouched keys are
    dropped (LM head, dropout, attention bias buffers, etc.).
    """
    out: dict[str, np.ndarray] = {}

    # Embeddings.
    if "transformer.wte.weight" in sd:
        out["tok_embed"] = sd["transformer.wte.weight"]
    elif "wte.weight" in sd:
        out["tok_embed"] = sd["wte.weight"]
    if "transformer.wpe.weight" in sd:
        out["pos_embed"] = sd["transformer.wpe.weight"]
    elif "wpe.weight" in sd:
        out["pos_embed"] = sd["wpe.weight"]

    # Final layer norm.
    if "transformer.ln_f.weight" in sd:
        out["ln_f_w"] = sd["transformer.ln_f.weight"]
    if "transformer.ln_f.bias" in sd:
        out["ln_f_b"] = sd["transformer.ln_f.bias"]

    # LM head — tied to wte by default in GPT-2.
    if "lm_head.weight" in sd:
        out["unembed_w"] = sd["lm_head.weight"]
    elif tied_unembed and "tok_embed" in out:
        out["unembed_w"] = out["tok_embed"]
    if "unembed_w" in out:
        out["unembed_b"] = np.zeros((out["unembed_w"].shape[0],), dtype=out["unembed_w"].dtype)

    # Per-layer weights.
    layers_seen: set[int] = set()
    for k in sd:
        m = _GPT2_LAYER_RE.match(k)
        if m:
            layers_seen.add(int(m.group(1)))

    for i in sorted(layers_seen):
        prefix = f"transformer.h.{i}."

        def g(suffix: str, pref: str = prefix) -> np.ndarray | None:
            return sd.get(pref + suffix)

        # ln_1
        if g("ln_1.weight") is not None:
            out[f"layer{i}.ln1_w"] = g("ln_1.weight")
        if g("ln_1.bias") is not None:
            out[f"layer{i}.ln1_b"] = g("ln_1.bias")
        # ln_2
        if g("ln_2.weight") is not None:
            out[f"layer{i}.ln2_w"] = g("ln_2.weight")
        if g("ln_2.bias") is not None:
            out[f"layer{i}.ln2_b"] = g("ln_2.bias")
        # attn.c_attn  (GPT-2 fuses qkv into one matrix)
        if g("attn.c_attn.weight") is not None:
            out[f"layer{i}.qkv_w"] = _maybe_transpose_conv1d(
                g("attn.c_attn.weight"), transpose_conv1d
            )
        if g("attn.c_attn.bias") is not None:
            out[f"layer{i}.qkv_b"] = g("attn.c_attn.bias")
        # attn.c_proj
        if g("attn.c_proj.weight") is not None:
            out[f"layer{i}.out_w"] = _maybe_transpose_conv1d(
                g("attn.c_proj.weight"), transpose_conv1d
            )
        if g("attn.c_proj.bias") is not None:
            out[f"layer{i}.out_b"] = g("attn.c_proj.bias")
        # mlp.c_fc
        if g("mlp.c_fc.weight") is not None:
            out[f"layer{i}.mlp_in_w"] = _maybe_transpose_conv1d(
                g("mlp.c_fc.weight"), transpose_conv1d
            )
        if g("mlp.c_fc.bias") is not None:
            out[f"layer{i}.mlp_in_b"] = g("mlp.c_fc.bias")
        # mlp.c_proj
        if g("mlp.c_proj.weight") is not None:
            out[f"layer{i}.mlp_out_w"] = _maybe_transpose_conv1d(
                g("mlp.c_proj.weight"), transpose_conv1d
            )
        if g("mlp.c_proj.bias") is not None:
            out[f"layer{i}.mlp_out_b"] = g("mlp.c_proj.bias")

    return out


def export_state_dict(model: ModelWeights) -> dict[str, np.ndarray]:
    """Inverse of :func:`load_state_dict` — produces the canonical
    flat-name mapping. Useful for round-trip tests."""
    return {name: arr for name, arr in model.all_arrays()}
