# Verifiable AI Inference

> Every model output ships with a cryptographic certificate that proves
> *which model* produced it, from *which input*, via *which sequence of
> kernel calls*. Anyone with the public model hash can re-derive the
> certificate; forging one requires inverting SHA-256 or breaking
> Curve25519.

This is a complete, dependency-light implementation of the layer that
has been missing from the AI stack. Every component — canonical
hashing, hash-chained kernel traces, Merkle commitments, certificates,
public-key signatures, autoregressive sessions, model registries,
selective disclosure, an HTTP API — is implemented, tested, and
specified.

| | |
| --- | --- |
| **Lines of code**  | ~2700 (excluding tests + spec) |
| **Tests**          | 171 (unit + integration + adversarial) |
| **Dependencies**   | `numpy` only |
| **Spec**           | [`SPEC.md`](SPEC.md) — wire-protocol level |
| **Demo**           | `python -m verifiable_inference.full_demo` |
| **Server**         | `python -m verifiable_inference.server` |
| **Benchmarks**     | `python -m verifiable_inference.benchmarks` |

## Why this matters

Today, every AI system on Earth is unfalsifiable. A model gives you an
answer and you cannot prove the answer came from that model with that
input by that path. You cannot prove it was not tampered with. You
cannot prove the same model would produce the same answer tomorrow. You
cannot prove a regulator was shown the same model the public uses. The
entire trillion-dollar AI industry rests on a foundation of
"trust the API."

A transformer running deterministic inference with hash-chained kernel
traces ends this. Every output carries a certificate. The certificate
says: this exact model, with these exact weights, given this exact
input, produced this exact output via this exact reasoning chain, and
here is the Merkle root that lets you re-derive any of it bit-for-bit.

## Architecture

```
                 ┌─────────────────┐
        ┌────────│   Registry      │   weight_root → publisher-signed
        │        │  (transparency  │   (config, public_key, license, ...)
        │        │   log)          │
        │        └─────────────────┘
        │ lookup            ▲
        │                   │ attest (publisher Ed25519)
        ▼                   │
┌─────────────────┐   ┌─────────────────┐
│   Verifier      │←──│   Prover        │   sign cert (model Ed25519)
│ structural      │   │ run inference   │   ┌──────────────────────┐
│ + re-derivation │   │ (deterministic) │←──│  ModelWeights        │
│ + signature     │   │                 │   │  (canonical hash =   │
│ + registry      │   │                 │   │   weight_root)       │
└────────▲────────┘   └────────┬────────┘   └──────────────────────┘
         │                     │
         │  Certificate        │  KernelRecord per op:
         │  ┌──────────────┐   │   ┌──────────────────────┐
         │  │ model_hash   │   ▼   │ chain_hash           │
         │  │ input_hash   │ ┌─────│ prev_chain_hash      │
         │  │ output_hash  │ │trace│ input_hashes         │
         │  │ chain_head   │ │     │ weight_hashes        │
         │  │ merkle_root  │←┤     │ params               │
         │  │ signature    │ │     │ output_hash          │
         │  │ kernel_summ. │ │     └──────────────────────┘
         │  │ full_trace?  │ │     │ │ │ │ │ │ │ │ │ │ │ │
         │  └──────────────┘ │     ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
         │                   └─►   binary Merkle tree
         │                          ┌──────────┐
         └──────────────────────────│ root     │
                                    └──────────┘

                            Sessions (autoregressive)
              GENESIS_SESSION = bytes(32)
              session[i] = H( session[i-1] || step_payload[i] )
              transcript_root = MerkleRoot([H(step_payload[i])])
```

## What's in this directory

### Core integrity primitives

| File | What it does |
| --- | --- |
| [`canonical.py`](canonical.py)   | Bit-exact, platform-independent hashing of arrays / JSON / bytes (with domain-separation magic prefixes). |
| [`merkle.py`](merkle.py)         | Domain-separated binary Merkle tree + inclusion proofs. |
| [`trace.py`](trace.py)           | Hash-chained `ExecutionTrace` (genesis = 32 zero bytes; each record links to the previous chain head). |
| [`signatures.py`](signatures.py) | Pure-Python Ed25519 (RFC 8032). 32-byte secret/public keys; deterministic signing. |
| [`certificate.py`](certificate.py) | Portable JSON cert; HMAC-SHA256 + Ed25519 signing. |
| [`disclosure.py`](disclosure.py)   | `DisclosedKernel` + selective-disclosure inclusion proofs. |
| [`session.py`](session.py)       | Autoregressive generation: per-step certs linked by session hash, transcript-level Merkle root. |
| [`registry.py`](registry.py)     | Model registry: pin `weight_root` → publisher-signed `(config, public_key, license, ...)`. |

### Compute

| File | What it does |
| --- | --- |
| [`kernels.py`](kernels.py)       | Reference deterministic kernels (`matmul`, `linear`, `softmax`, `layernorm`, `gelu`, `attention`); pure-Python loops, bit-exact across platforms. |
| [`fast_kernels.py`](fast_kernels.py) | Vectorized versions; bit-identical hashes to `kernels.py`, ~100× faster. Toggle with `use_fast_kernels()`. |
| [`model.py`](model.py)           | Tiny pre-LN GPT-style transformer. Multi-head attention. |
| [`loader.py`](loader.py)         | `load_state_dict()` (native names) + `convert_gpt2_state_dict()` (HF / minGPT / nanoGPT). |

### Application surface

| File | What it does |
| --- | --- |
| [`prover.py`](prover.py)        | `Prover.run()` → `(logits, Certificate)`. |
| [`verifier.py`](verifier.py)    | Two verification modes (structural / full re-derivation), stable error codes. |
| [`server.py`](server.py)        | Stdlib HTTP server: `/prove`, `/generate`, `/verify`, `/verify/full`, `/verify/transcript`, `/health`, `/model`. |
| [`client.py`](client.py)        | Stdlib HTTP client wrapping the same endpoints. |
| [`cli.py`](cli.py)              | `python -m verifiable_inference.cli prove|verify ...` |
| [`demo.py`](demo.py)            | Original end-to-end prove → verify → tamper-detect demo. |
| [`full_demo.py`](full_demo.py)  | Full-stack walkthrough: registry + sessions + selective disclosure + tamper detection at every layer. |
| [`benchmarks.py`](benchmarks.py) | Prove/verify time + cert size at varying model sizes. |

### Specification

| File | |
| --- | --- |
| [`SPEC.md`](SPEC.md) | Wire-protocol-level specification: canonical encodings, hash chain, Merkle construction, certificate JSON, signature algorithms, sessions, registry, all three verification protocols, threat model, security reductions. |

## The certificate

```jsonc
{
  "schema_version": "1.0",
  "model": {
    "config":       { "vocab_size": 32, "d_model": 16, "n_layers": 2, "n_heads": 2, "max_seq_len": 16, "dtype": "<f8" },
    "weight_root":  "eb96a607...4c182801"
  },
  "input":  { "tokens": [1,4,1,5,9,2,6], "hash": "..." },
  "output": { "logits_hash": "...", "predicted_token": 27, "all_logits": null },
  "trace": {
    "n_kernels":   29,
    "chain_head":  "f865d71a...686ecc55",   // last link of the SHA-256 chain
    "merkle_root": "70337a1c...cc2f0dd8",   // root of merkle(every kernel record)
    "kernel_summary": [ {"seq":0,"op":"embed","output_hash":"..."}, ... ]
  },
  "metadata":  { "numpy_version": "...", "platform": "...", "timestamp_utc": "..." },
  "signature": { "algo": "ed25519", "key_id": "model-vendor-key-1",
                 "public_key": "55154f42...ca554207", "value": "..." },
  "full_trace": [ /* one entry per kernel; optional */ ]
}
```

Self-contained certificate ≈ 20 KB for a 25-kernel inference; compact
form (no `full_trace`) ≈ 4 KB.

## Three verification modes

```python
from verifiable_inference import (
    Verifier, Prover, Registry,
    verify_certificate, verify_certificate_against_registry,
)

# 1) Structural — needs only the cert (with full_trace).
report = verify_certificate(cert)
report.raise_if_failed()

# 2) Full re-derivation — re-runs inference on the supplied weights,
#    checks every per-kernel hash, the merkle root, the chain head,
#    and the output match.
report = Verifier(model).verify(cert)

# 3) Registry-bound — looks the cert's weight_root up in a public
#    registry, confirms the signing pubkey matches, and verifies the
#    signature.
ok, reason = verify_certificate_against_registry(
    cert, registry, trusted_publisher_pks=[trusted_pk]
)
```

## Sessions (autoregressive generation)

```python
from verifiable_inference import Prover, generate, verify_transcript

prover = Prover(model)
transcript = generate(
    prover, prompt_tokens, max_new_tokens=10,
    sign_key=model_secret_key, sign_algo="ed25519",
)
# transcript.transcript_root commits to the entire generation
# transcript.final_session_hash is the chain head over all steps
assert verify_transcript(transcript, public_key=model_pub_key)
```

Each step is its own signed certificate; steps link by hashing
`parent_session_hash || step_payload`. A single corrupted step breaks
the chain. The transcript also has a Merkle root over per-step hashes
for O(log n) inclusion proofs.

## Selective disclosure

A regulator wants to audit one suspicious kernel call out of 10,000
without seeing the rest of the trace:

```python
from verifiable_inference import (
    compact_certificate, disclose_kernel, verify_disclosure,
)

# Prover publishes the compact cert (no full_trace).
public = compact_certificate(cert)

# Regulator asks for kernel #7,341. Prover discloses just that record
# plus an O(log n) proof.
disclosed = disclose_kernel(prover_trace_records, 7341)

# Regulator confirms the disclosed kernel was at position 7341 of the
# trace whose merkle_root is on the certificate.
assert verify_disclosure(disclosed, expected_root=bytes.fromhex(public.merkle_root))
```

## Run it

```bash
# 1) Originally tiny demo (47 tests of integrity primitives + tampering)
uv run python -m verifiable_inference.demo

# 2) Full-stack demo (multi-head model + registry + sessions + disclosure)
uv run python -m verifiable_inference.full_demo

# 3) HTTP server + client
uv run python -m verifiable_inference.server &
uv run python -c "
from verifiable_inference.client import Client
from verifiable_inference.session import verify_transcript
c = Client('http://127.0.0.1:8765')
print(c.health())
t = c.generate([1,2,3], max_new_tokens=4)
print('verified:', verify_transcript(t))
print(t.generated_tokens)"

# 4) Test suite (171 tests)
uv run python -m pytest verifiable_inference/tests -v

# 5) Benchmarks
uv run python -m verifiable_inference.benchmarks
```

## Benchmarks

Per-inference timings on a single core (fast kernels):

| config (d_model × layers × heads × seq) | n_kernels | cert bytes | prove ms | verify ms |
| --- | ---: | ---: | ---: | ---: |
|  4 × 1 × 1 × 4    |  14 |  9.8 KB |   4 |  2 |
|  8 × 2 × 1 × 8    |  25 | 16.7 KB |   2 |  2 |
| 16 × 2 × 2 × 16   |  29 | 19.2 KB |   3 |  4 |
| 32 × 4 × 4 × 16   |  71 | 45.2 KB |  11 | 12 |
| 64 × 4 × 8 × 32   | 103 | 64.8 KB |  40 | 43 |

Verify time is dominated by re-running inference (full re-derivation
mode); structural-only verification is roughly 5× faster again.

## What gets detected

The 64 adversarial fuzz tests in
[`tests/test_adversarial.py`](tests/test_adversarial.py) random-mutate
every signed/structural field and assert verifiers reject every
mutation. Coverage:

| Tamper | Detected by | Error code |
| --- | --- | --- |
| flip output hash         | structural / re-derivation | `output_hash_mismatch` / `rerun_output_mismatch` |
| flip merkle root         | structural                 | `merkle_root_mismatch` |
| flip chain head          | structural                 | `chain_head_mismatch` |
| flip any record's hash   | structural                 | `chain_break` |
| drop a kernel            | structural                 | `chain_head_mismatch` |
| swap two records         | structural                 | `chain_break` |
| change one input token   | re-derivation              | `rerun_merkle_mismatch` / signature failure |
| change one weight        | re-derivation              | `weight_root_mismatch` |
| sign with the wrong key  | signature                  | `bad_signature` |
| forge a registry entry   | registry attestation       | `registry_publisher_untrusted` |
| splice cert pubkey       | registry                   | `cert_pubkey_does_not_match_registry` |
| drop/swap a session step | transcript chain           | (verify_transcript returns False) |
| 40 random nibble flips   | one of the above           | always |

## What this unlocks

| Domain | What changes |
| --- | --- |
| **Healthcare** | A doctor can prescribe an AI-suggested treatment and the prescription embeds a proof of which model on what evidence. |
| **Law** | A judge can admit AI-assisted analysis as evidence because the chain is admissible. |
| **Regulation** | A regulator can audit a deployed model without trusting the company's word about which weights are running. |
| **Science** | A paper using AI can include the inference chain in supplementary materials and the result becomes reproducible. |
| **Security research** | A researcher can prove a jailbreak occurred without needing the company's logs. |
| **User agency** | A user can prove their AI agent did or did not take a specific action on their behalf. |
| **Insurance** | A claim can ride on AI analysis and the analysis is litigation-ready. |

## Limits

- **Verifier-visible inputs.** The default protocol exposes input
  tokens to the verifier. For private inputs, layer a zk-SNARK on top
  of the same Merkle commitment.
- **Model misbehaviour.** The certificate proves the computation
  happened, not that the model is correct, safe, or aligned.
- **Cross-architecture float determinism.** The kernels are bit-exact
  across IEEE-754 platforms; non-IEEE float environments may diverge.

None of these limit the **integrity** claim — only the deployment shape.

## License

Apache-2.0.
