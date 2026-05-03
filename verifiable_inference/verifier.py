"""Verifier: independently checks a certificate.

Two verification modes:

1. ``verify_certificate(cert)`` — *structural* verification. Needs only
   the certificate itself (when ``full_trace`` is included). Checks:
   - schema_version
   - hash chain replays correctly from genesis
   - chain_head matches the last record's chain_hash
   - Merkle root matches the tree built from the recorded leaves
   - the final-record output_hash matches the claimed logits hash

2. ``Verifier(model).verify(cert)`` — *full re-derivation*. Re-runs
   inference from scratch using the supplied weights and confirms every
   per-kernel hash, the merkle root, and the final output match. This is
   the strongest check: it proves the certificate could have been produced
   *only* by running the claimed model on the claimed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .canonical import hash_array
from .certificate import Certificate, hash_token_input
from .kernels import tracing
from .merkle import MerkleTree
from .model import ModelWeights, forward, greedy_next_token
from .prover import compute_weight_root
from .trace import ExecutionTrace, KernelRecord


class VerificationError(Exception):
    """Raised when a certificate fails any verification check.

    ``code`` is a short stable identifier for the failure reason — useful
    for machine-readable audit pipelines."""

    def __init__(self, message: str, code: str = "verification_failed") -> None:
        super().__init__(message)
        self.code = code


@dataclass
class VerificationReport:
    ok: bool
    checks: dict
    code: Optional[str] = None
    message: Optional[str] = None

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise VerificationError(self.message or "verification failed", self.code or "unknown")


def _verify_structural(cert: Certificate) -> VerificationReport:
    checks: dict = {}
    if cert.schema_version != "1.0":
        return VerificationReport(False, checks, "bad_schema",
                                   f"unsupported schema_version {cert.schema_version!r}")
    checks["schema_version"] = True

    if cert.full_trace is None:
        return VerificationReport(
            False, checks, "no_full_trace",
            "certificate has no full_trace; structural verification requires it",
        )

    records = [KernelRecord.from_dict(r) for r in cert.full_trace]
    if len(records) != cert.n_kernels:
        return VerificationReport(
            False, checks, "trace_count_mismatch",
            f"n_kernels {cert.n_kernels} does not match len(full_trace) {len(records)}",
        )
    checks["trace_count"] = True

    # Replay the chain — also checks per-record chain_hash + prev_chain_hash linkage.
    trace = ExecutionTrace()
    trace.records.extend(records)
    try:
        replayed = trace.replay_chain()
    except ValueError as e:
        return VerificationReport(False, checks, "chain_break", str(e))
    if replayed.hex() != cert.chain_head:
        return VerificationReport(
            False, checks, "chain_head_mismatch",
            f"replayed chain_head {replayed.hex()} != certificate chain_head {cert.chain_head}",
        )
    checks["chain_replay"] = True

    # Re-build Merkle tree.
    leaves = trace.merkle_leaves()
    if not leaves:
        return VerificationReport(False, checks, "empty_trace", "trace has no records")
    merkle = MerkleTree(leaves)
    if merkle.root.hex() != cert.merkle_root:
        return VerificationReport(
            False, checks, "merkle_root_mismatch",
            f"recomputed merkle_root {merkle.root.hex()} != certificate {cert.merkle_root}",
        )
    checks["merkle_root"] = True

    # Final record must produce the certificate's output logits.
    final = records[-1]
    if final.output_hash.hex() != cert.output_logits_hash:
        return VerificationReport(
            False, checks, "output_hash_mismatch",
            f"final record output {final.output_hash.hex()} != certificate {cert.output_logits_hash}",
        )
    checks["output_hash"] = True

    # Spot-check kernel_summary if present.
    if cert.kernel_summary:
        if len(cert.kernel_summary) != len(records):
            return VerificationReport(
                False, checks, "summary_length_mismatch",
                f"kernel_summary length {len(cert.kernel_summary)} != records {len(records)}",
            )
        for s, r in zip(cert.kernel_summary, records):
            if s["seq"] != r.seq or s["op"] != r.op or s["output_hash"] != r.output_hash.hex():
                return VerificationReport(
                    False, checks, "summary_mismatch",
                    f"kernel_summary entry {s['seq']} disagrees with full_trace",
                )
        checks["kernel_summary"] = True

    return VerificationReport(True, checks)


def verify_certificate(
    cert: Certificate,
    *,
    sign_key: Optional[bytes] = None,
) -> VerificationReport:
    """Run structural verification. If ``sign_key`` is supplied, also check
    the HMAC signature."""
    report = _verify_structural(cert)
    if not report.ok:
        return report

    if sign_key is not None:
        if not cert.verify_signature(sign_key):
            return VerificationReport(
                False, report.checks, "bad_signature", "signature verification failed",
            )
        report.checks["signature"] = True
    elif cert.signature.get("algo") not in (None, "none"):
        # Signature present but no key supplied — note it but don't fail.
        report.checks["signature"] = "present_but_unverified"

    return report


class Verifier:
    """Full re-derivation verifier — needs the actual model weights."""

    def __init__(self, model: ModelWeights) -> None:
        self.model = model
        self._weight_root = compute_weight_root(model)

    def verify(
        self,
        cert: Certificate,
        *,
        sign_key: Optional[bytes] = None,
    ) -> VerificationReport:
        # Start with structural verification.
        report = verify_certificate(cert, sign_key=sign_key)
        if not report.ok:
            return report

        # Weight root binding.
        if cert.weight_root != self._weight_root:
            return VerificationReport(
                False, report.checks, "weight_root_mismatch",
                f"certificate weight_root {cert.weight_root} != local model {self._weight_root}",
            )
        report.checks["weight_root"] = True

        # Config binding.
        if cert.model_config != self.model.config.to_dict():
            return VerificationReport(
                False, report.checks, "config_mismatch",
                "certificate model_config does not match local model config",
            )
        report.checks["model_config"] = True

        # Re-run inference under a fresh trace.
        tokens = np.asarray(cert.input_tokens, dtype="<i8")
        if hash_token_input(tokens) != cert.input_hash:
            return VerificationReport(
                False, report.checks, "input_hash_mismatch",
                "input_hash does not match canonical hash of input_tokens",
            )
        report.checks["input_hash"] = True

        trace = ExecutionTrace()
        with tracing(trace):
            logits = forward(tokens, self.model)

        # Merkle root from re-run must match.
        merkle = MerkleTree(trace.merkle_leaves())
        if merkle.root.hex() != cert.merkle_root:
            return VerificationReport(
                False, report.checks, "rerun_merkle_mismatch",
                "merkle_root from local re-run does not match certificate",
            )
        report.checks["rerun_merkle_root"] = True

        if trace.chain_head.hex() != cert.chain_head:
            return VerificationReport(
                False, report.checks, "rerun_chain_mismatch",
                "chain_head from local re-run does not match certificate",
            )
        report.checks["rerun_chain_head"] = True

        # Output match.
        if hash_array(logits).hex() != cert.output_logits_hash:
            return VerificationReport(
                False, report.checks, "rerun_output_mismatch",
                "logits hash from local re-run does not match certificate",
            )
        report.checks["rerun_output_hash"] = True

        if cert.predicted_token is not None:
            if greedy_next_token(logits) != cert.predicted_token:
                return VerificationReport(
                    False, report.checks, "predicted_token_mismatch",
                    "predicted_token from local re-run disagrees",
                )
            report.checks["predicted_token"] = True

        return report
