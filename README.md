# Veritas-VM

> **Verifiable AI inference.** Every model output ships with a cryptographic
> certificate that proves *which model* produced it, from *which input*, via
> *which sequence of kernel calls*. Anyone with the public model hash can
> re-derive the certificate; forging one requires inverting SHA-256 or
> breaking Curve25519.

```
┌─────────────────┐        ┌─────────────────┐
│   Verifier      │ ◄────  │   Prover        │
│ structural      │  cert  │ deterministic   │
│ + re-derivation │        │ inference       │
│ + signature     │        │ + hash-chain    │
│ + registry      │        │ + Merkle        │
└─────────────────┘        └─────────────────┘
```

Veritas-VM is a working Python reference implementation of a protocol that
is itself still maturing. The reference implementation produces and verifies
real certificates today; the broader protocol — cross-language verifier, C
kernels compiled to WASM, formal security proofs — is design-stage.

## Quick start

```bash
pip install -e .

python -m verifiable_inference.full_demo        # full-stack walkthrough
python -m verifiable_inference.demo             # original tamper-detect demo
python -m verifiable_inference.server &         # HTTP prove/verify endpoints
python -m verifiable_inference.benchmarks       # prove/verify time + cert size
python -m pytest verifiable_inference/tests     # 171 tests
```

## What it does

The Python reference implementation under [`verifiable_inference/`](verifiable_inference/)
is the part that runs today. It is a single package, `numpy` is the only
runtime dependency, and 171 tests cover every integrity layer.

| Component | What it does | Module |
| --- | --- | --- |
| **Canonical hashing**     | Deterministic byte-level encoding of tensors and structs so the same value always hashes to the same digest, on any machine. | [`canonical.py`](verifiable_inference/canonical.py) |
| **Kernel trace**          | Each kernel call (matmul, attention, layernorm, …) is hash-chained: `h_n = H(h_{n-1} ‖ op ‖ inputs ‖ output)`. | [`trace.py`](verifiable_inference/trace.py), [`kernels.py`](verifiable_inference/kernels.py) |
| **Fast kernels**          | Bit-exact, ~100× faster numpy kernels for the same operations the reference kernels implement. | [`fast_kernels.py`](verifiable_inference/fast_kernels.py) |
| **Merkle commitment**     | Binary Merkle tree over the trace; the root commits to the entire computation. | [`merkle.py`](verifiable_inference/merkle.py) |
| **Deterministic transformer** | Reference transformer wired through the traced kernels; same weights + same input → same output, byte-for-byte. | [`model.py`](verifiable_inference/model.py) |
| **Certificate**           | The wire format: input hash, weight root, kernel-trace root, output, transcript metadata. | [`certificate.py`](verifiable_inference/certificate.py) |
| **Ed25519 signatures**    | Publisher signs certificates so a verifier can attribute them to a specific identity. | [`signatures.py`](verifiable_inference/signatures.py) |
| **Autoregressive sessions** | Per-token certificates linked into a session chain so multi-token generations carry one continuous proof. | [`session.py`](verifiable_inference/session.py) |
| **Model registry**        | Mapping from `weight_root` to publisher-signed metadata (config, public key, license). | [`registry.py`](verifiable_inference/registry.py) |
| **Selective disclosure**  | Inclusion proofs over the Merkle root that reveal one slice of the trace without disclosing the rest. | [`disclosure.py`](verifiable_inference/disclosure.py) |
| **HTTP server + client**  | `prove` and `verify` endpoints over plain HTTP for integration outside Python. | [`server.py`](verifiable_inference/server.py), [`client.py`](verifiable_inference/client.py) |
| **GPT-2 loader**          | Reads a GPT-2-style `state_dict` into the deterministic transformer so real weights can be certified. | [`loader.py`](verifiable_inference/loader.py) |
| **Wire-protocol spec**    | Byte-level specification of every hashed structure. | [`SPEC.md`](verifiable_inference/SPEC.md) |

## What it is not

- **Not a model.** Veritas-VM does not train, fine-tune, or alter any model. It runs an existing transformer through a deterministic, traced execution path.
- **Not a logging wrapper.** A log can be edited; a Merkle root cannot. Certificates are cryptographic commitments, not records.
- **Not a blockchain.** There is no consensus, no global ledger, no token. The registry is a transparency log; nothing requires distributed agreement.
- **Not magic interpretability.** A certificate proves *how* an output was computed, not *why* the model believes it. Determinism is not alignment.
- **Not a substitute for a threat model.** See [`THREAT-MODEL.md`](THREAT-MODEL.md) for what this construction does and does not defend against.

## Status

### Working today

Everything below is exercised by the test suite (`python -m pytest verifiable_inference/tests`, 171 tests including adversarial fuzz):

- Deterministic transformer with bit-exact reference and fast kernels
- Hash-chained kernel trace and binary Merkle commitment
- Certificate generation, structural verification, and re-derivation
- Ed25519 signing and verification
- Autoregressive sessions with per-token linked certificates
- Model registry with publisher-signed metadata
- Selective disclosure via Merkle inclusion proofs
- HTTP prove/verify server and client
- GPT-2-style `state_dict` loader
- Wire-protocol specification at the byte level ([`SPEC.md`](verifiable_inference/SPEC.md))

### Spec / design stage

These are part of the protocol vision but are not implemented in this repository yet:

- **C decision kernels compiled to WASM.** A small, auditable kernel set as the canonical implementation, with the Python kernels as a reference.
- **Cross-language verifier.** A standalone verifier in a memory-safe systems language (e.g. Rust) that depends only on a hash and a signature primitive.
- **Formal security proofs.** Machine-checked proofs of soundness for the certificate construction.
- **Scale integration.** Deterministic inference at production sizes, with the trace overhead amortized across batched workloads.

A reader should be able to tell within thirty seconds which side of this line any given claim falls on. If something is not in the "Working today" list, it is design-stage and should not be relied on.

## Repository layout

```
veritas-vm/
├── README.md              ← you are here
├── LICENSE                ← Apache-2.0
├── THREAT-MODEL.md        ← what the construction defends against, what it does not
├── pyproject.toml
└── verifiable_inference/  ← the Python reference implementation
    ├── README.md          ← package-level tour
    ├── SPEC.md            ← byte-level wire-protocol specification
    ├── canonical.py       ← deterministic hashing of tensors and structs
    ├── kernels.py         ← reference kernels (clear, slow)
    ├── fast_kernels.py    ← bit-exact fast kernels (~100× faster)
    ├── trace.py           ← hash-chained kernel trace
    ├── merkle.py          ← binary Merkle tree
    ├── model.py           ← deterministic transformer
    ├── certificate.py     ← certificate format
    ├── signatures.py      ← Ed25519
    ├── session.py         ← autoregressive session chain
    ├── registry.py        ← weight_root → publisher metadata
    ├── disclosure.py      ← selective disclosure proofs
    ├── prover.py          ← prover entry point
    ├── verifier.py        ← verifier entry point
    ├── server.py          ← HTTP prove/verify server
    ├── client.py          ← HTTP client
    ├── loader.py          ← GPT-2-style state_dict loader
    ├── demo.py            ← original tamper-detect demo
    ├── full_demo.py       ← full-stack walkthrough
    ├── benchmarks.py      ← prove/verify time + cert size
    ├── cli.py             ← `veritas-prove` entry point
    └── tests/             ← 171 tests
```

## Contributing

Veritas-VM is in the phase where outside contribution makes the most difference: the reference implementation works, the spec is written down, and the next moves — verifier-in-another-language, formal proofs, scale integration — are tractable, well-scoped pieces of work.

**Where to plug in.** The "Spec / design stage" list above is the contributor map. Each item is a self-contained project that can land independently of the others. The verifier-in-Rust effort in particular is high-value and standable on its own: it does not require touching the Python prover, and a small auditable verifier is what makes the "anyone can verify" claim credible.

**Before opening a PR.** Read [`verifiable_inference/SPEC.md`](verifiable_inference/SPEC.md) for the wire format and [`THREAT-MODEL.md`](THREAT-MODEL.md) for what the construction is supposed to guarantee. Run the test suite locally; new functionality should come with tests, and any change that affects a hashed structure must update the spec in the same PR.

**Issues and discussion.** Open an issue at [github.com/grapheneaffiliate/veritas-vm/issues](https://github.com/grapheneaffiliate/veritas-vm/issues) for bugs, spec ambiguities, or scoping questions on a design-stage component before writing code against it.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
