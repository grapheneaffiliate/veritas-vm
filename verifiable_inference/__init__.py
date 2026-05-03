"""Verifiable AI inference: deterministic execution + hash-chained kernel traces.

Every model output ships with a cryptographic certificate that proves:
- which model (by weight hash) ran
- on which input (by input hash)
- via which sequence of kernel calls (Merkle root over the trace)
- producing which output (by output hash)

The certificate is independently verifiable: anyone with the public model
weights and the certificate can re-run the inference and check that the
recomputed Merkle root matches. Forgery requires inverting SHA-256.
"""

from .canonical import canonical_array_bytes, hash_array, hash_bytes, hash_json
from .certificate import Certificate, load_certificate, save_certificate
from .merkle import MerkleTree, verify_merkle_proof
from .prover import Prover
from .trace import ExecutionTrace, KernelRecord
from .verifier import VerificationError, Verifier, verify_certificate

__all__ = [
    "Certificate",
    "ExecutionTrace",
    "KernelRecord",
    "MerkleTree",
    "Prover",
    "VerificationError",
    "Verifier",
    "canonical_array_bytes",
    "hash_array",
    "hash_bytes",
    "hash_json",
    "load_certificate",
    "save_certificate",
    "verify_certificate",
    "verify_merkle_proof",
]

__version__ = "0.1.0"
