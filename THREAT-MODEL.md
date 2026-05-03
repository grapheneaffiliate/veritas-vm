# Threat Model

This document states what the Veritas-VM construction is designed to defend
against, what it explicitly does not defend against, and the assumptions a
verifier must accept for the guarantees to hold. It is a stub; it will grow
as the protocol matures and as formal proofs replace prose arguments.

A claim that is not on the "defends against" list below should be treated as
out of scope, even if it sounds adjacent.

## What the construction defends against

- **Output forgery.** An attacker cannot produce a valid certificate for an
  `(input, output)` pair that the registered model would not produce, without
  inverting SHA-256 or breaking Ed25519. The certificate binds the output to
  the kernel-trace root, the trace root to the weight root, and the weight
  root to the publisher's signing key.
- **Silent model swap.** A server cannot quietly answer with model B while
  claiming to run model A. The certificate names a specific `weight_root`;
  the registry binds that root to publisher-signed metadata; a verifier that
  checks both detects the substitution.
- **Inference tampering.** A prover that perturbs intermediate activations,
  skips a kernel, reorders operations, or substitutes a kernel implementation
  produces a trace whose Merkle root no longer matches a re-derivation from
  the same inputs and weights. The verifier rejects.
- **Supply-chain substitution of weights.** Replacing the weight file after
  publication changes the `weight_root`. Certificates issued under the old
  root remain verifiable; certificates issued under the new root surface as
  a new entry in the registry rather than silently shadowing the old one.
- **Selective-disclosure forgery.** A disclosed slice of a trace is bound to
  the same Merkle root as the rest of the computation. A prover cannot reveal
  a slice that did not actually occur in the committed execution.

## What the construction does not defend against

- **Harmful but deterministic outputs.** A certificate proves *how* an output
  was computed, not *whether* it is safe, true, or aligned. A model whose
  deterministic outputs are themselves harmful will produce valid
  certificates for those harmful outputs.
- **Side-channel attacks against the prover.** Timing, power, cache, and
  memory side channels on the machine running the prover are out of scope.
  The certificate commits to the result of computation, not to the physical
  conditions under which it was computed.
- **Compromise of the publisher's signing key.** If the Ed25519 private key
  is exfiltrated, the attacker can sign certificates indistinguishable from
  the legitimate publisher's. Key custody is the publisher's responsibility.
- **Compromise of the registry.** If the registry is replaced or its entries
  are altered, a verifier resolving `weight_root` through it can be misled
  about which publisher controls which model. A future transparency-log
  construction is intended to harden this; today, the registry is trusted.
- **Hash-function or signature-scheme breaks.** A practical preimage or
  collision attack on SHA-256, or a key-recovery attack on Ed25519, breaks
  the construction. The protocol is designed to allow primitive substitution,
  but no such migration is in place today.
- **Denial of service.** Nothing in the certificate prevents a prover from
  refusing to serve, returning errors, or rate-limiting. Liveness is not a
  property this construction provides.
- **Model extraction or weight confidentiality.** Certificates commit to the
  weight root, not to the weights themselves. The protocol does not protect
  weights from extraction via repeated queries, and selective disclosure is
  designed to *reveal* trace slices, not to hide them.

## Trust assumptions

A verifier accepting a Veritas-VM certificate is implicitly relying on:

1. **Cryptographic primitives.** SHA-256 is collision-resistant; Ed25519 is
   unforgeable under chosen-message attack.
2. **Registry integrity.** The mapping from `weight_root` to publisher
   metadata that the verifier consults has not been tampered with.
3. **Signing-key custody.** The publisher's Ed25519 private key has not been
   exfiltrated.
4. **Deterministic execution.** The reference implementation produces
   bit-exact results across the platforms a verifier and prover are expected
   to share. The fast kernels are tested for bit-exact equivalence with the
   reference kernels; deviations would be a bug, not a permitted variance.
5. **Specification fidelity.** The byte-level layout of every hashed
   structure is as specified in [`verifiable_inference/SPEC.md`](verifiable_inference/SPEC.md).
   A verifier built against a different layout will compute different roots.

## Out of scope for this stub

The following will be addressed as the protocol matures and are intentionally
not covered here:

- Formal, machine-checked proofs of soundness for the certificate
  construction.
- A transparency-log construction for the registry.
- A specification for primitive migration (e.g. SHA-256 → SHA-3, Ed25519 →
  a post-quantum scheme).
- A specification for the C-kernel-to-WASM canonical kernel set and the
  bit-exactness contract between it and the Python reference.

Until those land, treat this document as a scope statement, not a security
proof.
