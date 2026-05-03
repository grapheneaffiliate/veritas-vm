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

The package lives under [`verifiable_inference/`](verifiable_inference/) —
that directory has the full README, the formal wire-protocol spec
([`SPEC.md`](verifiable_inference/SPEC.md)), the demos, the benchmarks,
and 171 tests covering every integrity layer.

## Quick start

```bash
pip install -e .

python -m verifiable_inference.full_demo        # full-stack walkthrough
python -m verifiable_inference.demo             # original tamper-detect demo
python -m verifiable_inference.server &         # HTTP prove/verify endpoints
python -m verifiable_inference.benchmarks       # prove/verify time + cert size
python -m pytest verifiable_inference/tests     # 171 tests
```

## What's inside

| | |
| --- | --- |
| **Lines of code**  | ~2700 (excluding tests + spec) |
| **Tests**          | 171 (unit + integration + adversarial fuzz) |
| **Dependencies**   | `numpy` only |
| **Spec**           | [`verifiable_inference/SPEC.md`](verifiable_inference/SPEC.md) — wire-protocol level |

Components: canonical hashing, hash-chained kernel trace, binary Merkle
tree, deterministic transformer (reference + ~100×-faster bit-exact
kernels), certificate format, Ed25519 signatures, autoregressive
sessions with linked per-token certificates, model registry, selective
disclosure via inclusion proofs, HTTP server + client, GPT-2-style
state_dict loader. See [`verifiable_inference/README.md`](verifiable_inference/README.md)
for the full tour.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
