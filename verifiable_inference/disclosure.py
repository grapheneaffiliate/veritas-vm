"""Selective disclosure: prove that a *specific* kernel call ran without
revealing the rest of the trace.

Use case: a regulator audits one suspicious step in a 10,000-kernel
inference. The model vendor publishes a compact certificate (containing
the Merkle root but not the full trace) and a per-kernel inclusion proof
plus that one ``KernelRecord``. The regulator verifies that record was
genuinely part of the inference whose root is on the certificate, with
O(log n) bytes of proof material — no other internal state leaks.

Threat model: the prover can choose *which* records to disclose, but
cannot disclose a record that wasn't in the original trace, and cannot
disclose a record at the wrong position. Both are prevented by the
domain-separated Merkle construction in :mod:`merkle`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import hash_bytes
from .merkle import MerkleProof, MerkleTree, verify_merkle_proof
from .trace import KernelRecord


@dataclass(frozen=True)
class DisclosedKernel:
    """A single kernel record + the proof that it was at position
    ``record.seq`` in the trace whose Merkle root is ``merkle_root``."""

    record: KernelRecord
    merkle_root: bytes
    proof: MerkleProof

    def to_dict(self) -> dict:
        return {
            "record": self.record.to_dict(),
            "merkle_root": self.merkle_root.hex(),
            "proof": {
                "leaf_index": self.proof.leaf_index,
                "leaf_hash": self.proof.leaf_hash.hex(),
                "siblings": [s.hex() for s in self.proof.siblings],
                "directions": list(self.proof.directions),
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "DisclosedKernel":
        rec = KernelRecord.from_dict(d["record"])
        proof = MerkleProof(
            leaf_index=int(d["proof"]["leaf_index"]),
            leaf_hash=bytes.fromhex(d["proof"]["leaf_hash"]),
            siblings=[bytes.fromhex(s) for s in d["proof"]["siblings"]],
            directions=[bool(x) for x in d["proof"]["directions"]],
        )
        return DisclosedKernel(
            record=rec,
            merkle_root=bytes.fromhex(d["merkle_root"]),
            proof=proof,
        )


def disclose_kernel(
    records: list[KernelRecord],
    seq: int,
) -> DisclosedKernel:
    """Build a ``DisclosedKernel`` for the kernel at position ``seq`` in
    ``records``. The full record list is needed to compute the proof
    (the Merkle tree spans all records) but only the disclosed leaf and
    its proof are exposed to the verifier."""
    if seq < 0 or seq >= len(records):
        raise IndexError(seq)
    leaves = [hash_bytes(r.canonical_payload()) for r in records]
    tree = MerkleTree(leaves)
    proof = tree.proof(seq)
    return DisclosedKernel(record=records[seq], merkle_root=tree.root, proof=proof)


def verify_disclosure(
    disclosed: DisclosedKernel,
    *,
    expected_root: bytes | None = None,
) -> bool:
    """Independently verify a disclosed kernel.

    If ``expected_root`` is supplied (e.g. read from a certificate the
    verifier already trusts), the disclosed root must match it.
    Otherwise the function only checks the proof is internally consistent.

    The Merkle leaf for a kernel record is ``hash_bytes(canonical_payload)``
    — i.e. trace records are double-hashed (once for canonical-form
    binding, once by the Merkle leaf-tag) so an internal-node hash can
    never collide with a leaf hash.
    """
    if expected_root is not None and expected_root != disclosed.merkle_root:
        return False
    if disclosed.record.seq != disclosed.proof.leaf_index:
        return False
    leaf_input = hash_bytes(disclosed.record.canonical_payload())
    return verify_merkle_proof(leaf_input, disclosed.proof, disclosed.merkle_root)


def compact_certificate(cert) -> "object":
    """Return a copy of ``cert`` with ``full_trace`` stripped — the
    canonical "publishable" form. Verifiers paired with disclosed
    kernels only need the merkle root and the chain head."""
    from copy import deepcopy

    from .certificate import Certificate

    d = cert.to_dict()
    d.pop("full_trace", None)
    return Certificate.from_dict(d)
