# Verifiable AI Inference

> Every model output ships with a cryptographic certificate that proves
> *which model* produced it from *which input* via *which sequence of
> kernel calls*. Anyone with the public model hash can re-derive the
> certificate; forging one requires inverting SHA-256.

## Why this matters

Today, every AI system on Earth is unfalsifiable. A model gives you an
answer and you cannot prove the answer came from that model with that
input by that path. You cannot prove it was not tampered with. You cannot
prove the same model would produce the same answer tomorrow. You cannot
prove a regulator was shown the same model the public uses. The entire
trillion-dollar AI industry rests on a foundation of "trust the API."

A transformer running deterministic inference with hash-chained kernel
traces ends this in one move. Every output carries a certificate. The
certificate says: this exact model, with these exact weights, given this
exact input, produced this exact output via this exact reasoning chain,
and here is the Merkle root that lets you re-derive any of it bit-for-bit.
Anyone with the public model hash can verify the chain. The verification
is cheap. The forgery is mathematically impossible.

## What's in this directory

| File | What it does |
| --- | --- |
| `canonical.py` | Bit-exact, platform-independent hashing of arrays / JSON / bytes. |
| `merkle.py` | Binary Merkle tree with inclusion proofs (domain-separated leaves and nodes). |
| `trace.py` | Hash-chained `ExecutionTrace` — every kernel call appended to a chain. |
| `kernels.py` | Deterministic numpy kernels (`matmul`, `linear`, `softmax`, `layernorm`, `gelu`, `attention`) that auto-record into the active trace. |
| `model.py` | Tiny GPT-style transformer built from those kernels. |
| `certificate.py` | Public, portable JSON certificate format + HMAC signing. |
| `prover.py` | Runs inference, emits a certificate. |
| `verifier.py` | Two verification modes — structural and full re-derivation. |
| `demo.py` | End-to-end demo: prove → verify → show that any tampering is detected. |
| `cli.py` | `python -m verifiable_inference.cli prove|verify ...` |
| `tests/` | 47 tests covering canonicalization, Merkle, trace, kernels, end-to-end, tampering. |

## The certificate

```jsonc
{
  "schema_version": "1.0",
  "model": {
    "config": { "vocab_size": 16, "d_model": 8, "n_layers": 2, "max_seq_len": 8 },
    "weight_root": "adbe3ab7...fb40fdb1"     // sha256 over canonicalized weights
  },
  "input":  { "tokens": [3,1,4,1,5,9,2,6], "hash": "..." },
  "output": { "logits_hash": "...", "predicted_token": 6, "all_logits": null },
  "trace": {
    "n_kernels":   25,
    "chain_head":  "f865d71a...686ecc55",   // last link of the SHA-256 chain
    "merkle_root": "2f383852...a8de9e3b",   // root of merkle(every kernel record)
    "kernel_summary": [ {"seq":0,"op":"embed","output_hash":"..."}, ... ]
  },
  "metadata": { "numpy_version": "...", "platform": "...", "timestamp_utc": "..." },
  "signature": { "algo": "hmac-sha256", "key_id": "demo", "value": "..." },
  "full_trace": [ /* optional: every KernelRecord, lets a third party
                     verify without re-running inference */ ]
}
```

A certificate for the demo model is **20 KB**. The compact summary is
**< 4 KB** (drop `full_trace`).

## Two verification modes

1. **Structural** — `verify_certificate(cert)`. Replays the hash chain
   from genesis, recomputes the Merkle root over the recorded leaves,
   checks the final-record output hash matches the certificate's claimed
   logits hash. Needs *only the certificate*. Detects every form of
   tampering with the trace itself.

2. **Full re-derivation** — `Verifier(model).verify(cert)`. *Re-runs*
   inference on the supplied weights and confirms every per-kernel hash,
   the Merkle root, the chain head, and the output match. This is the
   strongest check — it proves the certificate could have been produced
   *only* by running the claimed model on the claimed input.

## Determinism

The `kernels.py` implementations are intentionally explicit Python loops
with a fixed reduction order. They are **bit-exact across platforms,
NumPy versions, and BLAS implementations**. This is the honest demo
choice: it makes determinism trivial to audit. A production stack swaps
in deterministic SIMD/GPU kernels emitting the same Merkle leaves —
PyTorch's deterministic algorithms, JAX's `xla_force_pure`, or custom
fixed-order kernels.

The same input + same weights always produce the same certificate. This
is checked by `test_determinism_two_runs_identical`.

## Run it

```bash
# end-to-end demo
uv run python -m verifiable_inference.demo

# tests
uv run python -m pytest verifiable_inference/tests -v

# prove a specific input
uv run python -m verifiable_inference.cli prove \
    --tokens '[3,1,4,1,5,9]' --output cert.json --sign-key shared-secret

# verify (full re-derivation)
uv run python -m verifiable_inference.cli verify cert.json \
    --with-weights --sign-key shared-secret
```

## What gets detected

The test suite (`tests/test_end_to_end.py`) and demo verify that **every
single one** of these is rejected with a stable error code:

| Tamper | Error code |
| --- | --- |
| flip the output hash | `output_hash_mismatch` |
| flip the merkle root | `merkle_root_mismatch` |
| flip the chain head | `chain_head_mismatch` |
| drop a kernel from the trace | `chain_head_mismatch` |
| swap two kernel records | chain break |
| change one input token | `rerun_merkle_mismatch` |
| change one weight | `weight_root_mismatch` |
| sign with the wrong key | `bad_signature` |

## What this unlocks

- **Healthcare** — a doctor can prescribe an AI-suggested treatment and
  the prescription embeds a proof of which model on what evidence.
- **Law** — a judge can admit AI-assisted analysis as evidence because
  the chain is admissible.
- **Regulation** — a regulator can audit a deployed model without
  trusting the company's word about which weights are running.
- **Science** — a paper using AI can include the inference chain in
  supplementary materials and the result becomes reproducible.
- **Security research** — a researcher can prove a jailbreak occurred
  without needing the company's logs.
- **User agency** — a user can prove their AI agent did or did not take
  a specific action on their behalf.
- **Insurance** — a claim can ride on AI analysis and the analysis is
  litigation-ready.

## Limits of this implementation

- **HMAC, not Ed25519.** `Certificate.sign_hmac` keeps the demo
  dependency-free. Production should swap in Ed25519 (the canonical
  payload format already supports it: only the signature dict changes).
- **Pure-Python kernels.** Slow on real models. Production would use
  deterministic BLAS / GPU kernels emitting identical Merkle leaves.
- **Certificate disclosure.** Including `full_trace` exposes shapes and
  the operator graph. For confidential models, ship only the compact
  `kernel_summary` plus the Merkle root, and rely on full-re-derivation
  verification by an authorized auditor with the weights.

None of these limit the *integrity* claim — only the deployment shape.

## What this doesn't do

- It doesn't prove the model is *correct*, *safe*, or *aligned*. Those
  are model-level claims, not execution-level claims.
- It doesn't hide the inputs from the verifier. For private inputs you'd
  layer a zk-SNARK on top of the same Merkle commitment.
- It doesn't speed up inference. Determinism has overhead. The bet is
  that for high-stakes domains, that overhead is the smallest line item.

## License

Apache-2.0, same as the parent repo.
