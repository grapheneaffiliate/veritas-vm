# Verifiable AI Inference — Specification v1.0

This document specifies the certificate format, the verification
protocol, and the security guarantees of the verifiable_inference
stack. It is sufficient to build an interoperable verifier from
scratch in any language.

## 1. Notation

- `H(x)` : SHA-256 of byte string `x`. 32 bytes.
- `||`   : byte concatenation.
- `LE_u32(n)`, `LE_u64(n)` : little-endian 4- and 8-byte unsigned encodings of `n`.
- `hex(b)` : lowercase hex string of byte string `b`.
- `enc_int(s, n)` : `s.to_bytes(n, "little")`.

All hashes in this spec are SHA-256. The format is hash-agile (the
schema_version field gates upgrades) but v1.0 fixes SHA-256 everywhere.

## 2. Canonical encodings

Every hash in the protocol is computed over a *canonical* byte
representation. Two semantically-equal inputs MUST produce identical
canonical bytes regardless of platform / float endianness / dict order.

### 2.1 Domain separation prefixes

| Encoding kind | Magic prefix |
| --- | --- |
| ndarray       | `00 56 41 49 01 41 52 52 00`   ("\0VAI\1ARR\0")  |
| opaque bytes  | `00 56 41 49 01 42 59 54 00`   ("\0VAI\1BYT\0")  |
| canonical JSON| `00 56 41 49 01 4A 53 4E 00`   ("\0VAI\1JSN\0")  |

### 2.2 ndarray

```
canonical(arr) =
    MAGIC_ARRAY ||
    LE_u8(len(dtype_str)) || dtype_str ||      // ASCII dtype string e.g. "<f8"
    LE_u32(ndim) ||
    LE_u64(shape[0]) || ... || LE_u64(shape[ndim-1]) ||
    raw_bytes_in_C_order_little_endian
```

Allowlisted dtype strings: `<f8 <f4 <f2 <i8 <i4 <i2 <i1 <u8 <u4 <u2 <u1 |b1`.
Non-little-endian arrays MUST be byte-swapped before encoding.
Object/string dtypes MUST be rejected.

### 2.3 opaque bytes

```
canonical(b) = MAGIC_BYTES || LE_u64(len(b)) || b
```

### 2.4 JSON

```
canonical(obj) = MAGIC_JSON || LE_u64(len(payload)) || payload
```

where `payload` is `obj` encoded with: ASCII-only, sorted keys, no
whitespace separators, no NaN / Infinity. JSON booleans, numbers, strings,
arrays, and objects only.

## 3. Execution trace

### 3.1 KernelRecord

A KernelRecord is a sequence of operation evidence:

```
record = {
  seq:              uint                              // 0-based index in trace
  op:               string                            // kernel name (matmul, attention, ...)
  input_hashes:     [bytes32, ...]                    // canonical-array hashes of inputs
  weight_hashes:    [bytes32, ...]                    // canonical-array hashes of weights
  params:           json-canonical scalar dict        // op parameters (eps, axis, causal, ...)
  output_hash:      bytes32                           // canonical-array hash of output
  prev_chain_hash:  bytes32                           // chain head before this record
  chain_hash:       bytes32                           // chain head after this record
}
```

### 3.2 Canonical record payload

The bytes used both for chain linking and for Merkle leaf computation:

```
record_payload(rec) = canonical_json({
  "seq":             rec.seq,
  "op":              rec.op,
  "input_hashes":    [hex(h) for h in rec.input_hashes],
  "weight_hashes":   [hex(h) for h in rec.weight_hashes],
  "params":          rec.params,
  "output_hash":     hex(rec.output_hash),
  "prev_chain_hash": hex(rec.prev_chain_hash),
})
```

The `chain_hash` field is *excluded* from the payload — it is computed
from the payload, not signed by it.

### 3.3 Hash chain

```
GENESIS = bytes(32)                       // 32 zero bytes
chain[0]   = H_bytes(GENESIS  || record_payload(records[0]))
chain[i+1] = H_bytes(chain[i] || record_payload(records[i+1]))
```

where `H_bytes(x) = SHA256(MAGIC_BYTES || LE_u64(len(x)) || x)`.

For each record `i`: `records[i].chain_hash == chain[i]`.
For each record `i > 0`: `records[i].prev_chain_hash == chain[i-1]`.
For record 0: `records[0].prev_chain_hash == GENESIS`.

### 3.4 Merkle root

```
leaf[i] = H_bytes(record_payload(records[i]))
root    = MerkleRoot(leaves)
```

Merkle tree construction:
- Leaf-tag: `H(0x00 || leaf_bytes)`
- Node-tag: `H(0x01 || left || right)`
- Odd levels duplicate the last node before pairing.

## 4. Certificate

```
{
  "schema_version": "1.0",
  "model": {
    "config":      <json-canonical config dict>,
    "weight_root": <hex sha256 over canonicalized weights>
  },
  "input": {
    "tokens": [int, ...],
    "hash":   <hex sha256 of canonical_array(int64 token vector)>
  },
  "output": {
    "logits_hash":     <hex sha256 of canonical_array(logits)>,
    "predicted_token": int | null,
    "all_logits":      [[float]] | null      // optional disclosure
  },
  "trace": {
    "n_kernels":       int,
    "chain_head":      <hex chain hash after final record>,
    "merkle_root":     <hex merkle root over leaves>,
    "kernel_summary":  [{seq, op, output_hash}, ...]
  },
  "metadata": {
    "numpy_version":  string,
    "python_version": string,
    "platform":       string,
    "machine":        string,
    "tracer_version": string,
    "timestamp_utc":  string
  },
  "signature": <see §5>,
  "full_trace": [<KernelRecord dict>, ...]   // optional; see §6
}
```

### 4.1 weight_root

```
weight_root = hex(H_bytes(
    canonical_json(config)
 || canonical_json({"name": name_0, "shape": shape_0}) || canonical_array_hash(arr_0)
 || canonical_json({"name": name_1, "shape": shape_1}) || canonical_array_hash(arr_1)
 || ...
))
```

Names and shapes are taken from `ModelWeights.all_arrays()` in declaration order.

### 4.2 input_hash

```
tokens_int64 = numpy.asarray(tokens, dtype="<i8")
input_hash   = hex(canonical_array_hash(tokens_int64))
```

### 4.3 output

```
predicted_token = argmax(logits[-1])     // ties broken by smallest index
output_hash     = hex(canonical_array_hash(logits))
```

## 5. Signatures

### 5.1 Signed payload

The bytes covered by the signature are:

```
canonical_json(certificate)        with the "signature" field removed
```

### 5.2 Algorithms

| algo          | signature dict fields                                                    |
| ------------- | ------------------------------------------------------------------------ |
| `none`        | `{"algo": "none", "key_id": null, "value": null}`                        |
| `hmac-sha256` | `{"algo": "hmac-sha256", "key_id": str, "value": hex}`                   |
| `ed25519`     | `{"algo": "ed25519", "key_id": str, "public_key": hex32, "value": hex64}` |

`hmac-sha256` is symmetric; verifier needs the secret. `ed25519` is
asymmetric per RFC 8032, deterministic; the certificate self-contains
the public key (so a verifier needs only the registry to know which key
to trust).

## 6. Trace inclusion mode

The `full_trace` field is OPTIONAL. Two deployment shapes:

1. **Self-contained certificate.** `full_trace` present.
   Verifier replays the chain and rebuilds the Merkle root from the
   records alone. No network access, no model weights required for
   structural verification.

2. **Compact certificate + selective disclosure.** `full_trace` absent.
   The merkle_root and chain_head still bind the certificate to a
   specific computation. To audit individual kernels the prover emits
   `DisclosedKernel` documents (§7) on demand.

## 7. Selective disclosure (DisclosedKernel)

```
{
  "record": <KernelRecord dict>,
  "merkle_root": <hex>,
  "proof": {
    "leaf_index": int,
    "leaf_hash":  <hex>,
    "siblings":   [<hex>, ...],
    "directions": [bool, ...]      // True iff sibling is on the right
  }
}
```

Verification:
1. `record.seq == proof.leaf_index`
2. `H_bytes(record_payload(record))` equals `proof.leaf_hash` after
   leaf-tag hashing (i.e. the merkle proof verifies it as a leaf input).
3. Recompute the merkle path from `proof.siblings` + `proof.directions`,
   compare to the disclosed `merkle_root`.
4. (Optional) compare `merkle_root` to a previously-trusted certificate.

## 8. Sessions (autoregressive generation)

A SessionTranscript binds N per-token Certificates into a chain:

```
GENESIS_SESSION = bytes(32)
step_payload[i] = signed_payload_bytes(cert_i)
session[i]      = H_bytes(parent || step_payload[i])
            where parent = GENESIS_SESSION (i=0) or session[i-1] (i>0)
transcript_root = MerkleRoot([H_bytes(step_payload[i]) for i in 0..N-1])
final_session_hash = session[N-1]
```

The transcript document carries:

```
{
  "schema":               "vai-session/1.0",
  "model_weight_root":    hex,
  "prompt_tokens":        [int, ...],
  "steps":                [{step_index, parent_session_hash, session_hash, cert}, ...],
  "final_session_hash":   hex,
  "transcript_root":      hex
}
```

## 9. Registry

A registry maps `weight_root → RegistryEntry`. An entry carries:

```
{
  "schema":         "vai-registry-entry/1.0",
  "weight_root":    hex,
  "config":         <model config>,
  "public_key":     hex32                    // signing key the prover uses
  "name":           str,
  "version":        str,
  "license":        str,
  "training_data":  str,
  "model_card_url": str | null,
  "description":    str,
  "attestation":    {"algo": "ed25519", "publisher_public_key": hex32, "value": hex64}
}
```

The publisher signs the canonical_json of the entry minus its own
attestation field. Verifiers maintain a list of trusted publisher keys.

## 10. Verification protocols

### 10.1 Structural verification (cert only)

Inputs: certificate.
Steps:
1. `schema_version == "1.0"`.
2. If `full_trace` absent: ABORT — structural verification not possible.
3. Replay hash chain from GENESIS, checking each record's
   `prev_chain_hash` and `chain_hash`. Final equals `chain_head`.
4. Build Merkle tree from `merkle_leaves(records)` and compare to
   `merkle_root`.
5. `records[-1].output_hash == output.logits_hash`.
6. `len(kernel_summary) == n_kernels` and each entry agrees with the
   corresponding record.

### 10.2 Full re-derivation (cert + weights)

Inputs: certificate, ModelWeights.
Steps:
1. Run §10.1.
2. `weight_root(weights) == cert.model.weight_root`.
3. `weights.config == cert.model.config`.
4. `canonical_array_hash(input_tokens) == cert.input.hash`.
5. Run inference on `weights` with `cert.input.tokens` under a fresh
   ExecutionTrace.
6. Recomputed merkle_root == `cert.trace.merkle_root`.
7. Recomputed chain_head == `cert.trace.chain_head`.
8. `canonical_array_hash(re-run_logits) == cert.output.logits_hash`.
9. `argmax(logits[-1]) == cert.output.predicted_token` (if specified).

### 10.3 Registry-bound verification

1. Lookup `cert.model.weight_root` in registry. ABORT if absent.
2. Verify entry's attestation (if a trusted-publisher set is provided,
   the publisher key must be in the set).
3. `cert.signature.public_key == entry.public_key`.
4. `cert.verify_signature()`.
5. Run §10.1 or §10.2.

## 11. Threat model

### 11.1 Adversary capabilities

The protocol is robust against an adversary who can:
- Read every public certificate and registry entry.
- See the model weights (white-box prover).
- Execute arbitrary code on the prover.
- Tamper with certificates in transit.
- Issue forged registry entries (but cannot forge a valid attestation
  without the publisher's secret key).

### 11.2 Security goals

| Goal | Achieved by |
| --- | --- |
| **Output integrity** — the certificate's claimed output is the bit-exact result of running the claimed model on the claimed input | hash chain + Merkle root + final-record output hash check (§10.1, §10.2) |
| **Computation integrity** — the certificate's listed kernel sequence is the exact one that produced the output | per-kernel hash linkage + Merkle commitment |
| **Model integrity** — the certificate is bound to a specific model | weight_root canonical hash; verifier compares against locally-loaded weights |
| **Non-repudiation** — a model owner cannot deny having signed a certificate | Ed25519 signature over the canonical signed payload |
| **Public verifiability** — any third party can verify without a shared secret | Ed25519 public key in the certificate, optionally pinned via registry |
| **Selective disclosure** — a prover can prove a single kernel without revealing the whole trace | Merkle inclusion proof (§7) |
| **Session integrity** — multi-token generation cannot have steps inserted/dropped/swapped | session hash chain + transcript root (§8) |

### 11.3 Out of scope (NOT defended against)

- **Model misbehaviour.** The certificate proves the computation
  happened, not that the model is correct, safe, or aligned. A model
  trained to lie still produces honest certificates of its lies.
- **Input privacy from the verifier.** The default protocol exposes
  input tokens to the verifier. Use a zero-knowledge wrapper if input
  privacy is required.
- **Denial of service.** A prover may refuse to produce a certificate
  or refuse to disclose a kernel. The registry can attest to which
  inputs SHOULD be served, but not enforce service.
- **Side channels.** Timing, power, or memory side channels in the
  prover are not addressed here.
- **Cross-architecture floating-point determinism beyond IEEE-754.**
  The reference and fast kernels are bit-exact across IEEE-754 platforms;
  exotic float environments (e.g. flush-to-zero, alternate rounding modes)
  may diverge.

### 11.4 Security reductions

- **Chain forgery → SHA-256 second-preimage.** Forging a record at
  position `i` with a different payload but the same `chain_hash`
  requires finding a second preimage of SHA-256.
- **Merkle forgery → SHA-256 collision.** Producing a different leaf
  set with the same Merkle root requires SHA-256 collisions in the
  domain-separated leaf/node hashes.
- **Certificate forgery (Ed25519) → discrete log.** Forging an Ed25519
  signature without the secret key reduces to the elliptic curve
  discrete log problem on Curve25519 (≥ 128 bits of security).
- **Registry forgery → publisher key compromise.** Without compromising
  a trusted publisher's key, an adversary cannot insert a new entry.

## 12. Versioning

`schema_version` is the wire-protocol version of the Certificate
document. v1.0 is the version this spec describes. Future versions MUST
be backwards-incompatible if any of:
- the canonical encoding rules change
- the chain or Merkle construction changes
- the signed-payload set changes
- the signature algorithms change

The signature dict's `algo` field is independent of `schema_version` —
new signature algorithms can be added without bumping the schema.
