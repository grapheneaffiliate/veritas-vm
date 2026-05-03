"""Certificate format: the public, portable proof of an inference.

A certificate is a small JSON document. It does not contain weights —
just hashes — so it's safe to publish even when the model is private.

Structure (top level keys):
  schema_version    : str
  model             : { config: {...}, weight_root: hex }
  input             : { tokens: [int], hash: hex }
  output            : { logits_hash: hex, predicted_token: int|null,
                        all_logits: [[float]] | None }
  trace             : { n_kernels: int,
                        chain_head: hex,
                        merkle_root: hex,
                        kernel_summary: [ {seq, op, output_hash}, ... ] }
  metadata          : { numpy_version, python_version, platform,
                        tracer_version, timestamp_utc }
  signature         : { algo: "hmac-sha256"|"none", key_id: str|null,
                        value: hex|null }
  full_trace        : optional — the full list of KernelRecord dicts. When
                      present the certificate is *self-contained* (re-runnable
                      without re-running inference); otherwise the verifier
                      must re-run the prover.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import platform as platform_mod
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .canonical import canonical_json_bytes, hash_array
from .trace import KernelRecord

SCHEMA_VERSION = "1.0"


@dataclass
class Certificate:
    model_config: dict
    weight_root: str
    input_tokens: list[int]
    input_hash: str
    output_logits_hash: str
    predicted_token: Optional[int]
    all_logits: Optional[list[list[float]]]
    n_kernels: int
    chain_head: str
    merkle_root: str
    kernel_summary: list[dict]
    metadata: dict = field(default_factory=dict)
    signature: dict = field(default_factory=lambda: {"algo": "none", "key_id": None, "value": None})
    full_trace: Optional[list[dict]] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model": {"config": self.model_config, "weight_root": self.weight_root},
            "input": {"tokens": list(self.input_tokens), "hash": self.input_hash},
            "output": {
                "logits_hash": self.output_logits_hash,
                "predicted_token": self.predicted_token,
                "all_logits": self.all_logits,
            },
            "trace": {
                "n_kernels": self.n_kernels,
                "chain_head": self.chain_head,
                "merkle_root": self.merkle_root,
                "kernel_summary": self.kernel_summary,
            },
            "metadata": self.metadata,
            "signature": self.signature,
        }
        if self.full_trace is not None:
            d["full_trace"] = self.full_trace
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Certificate":
        ver = d.get("schema_version", "0")
        if ver != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {ver!r}")
        return Certificate(
            schema_version=ver,
            model_config=dict(d["model"]["config"]),
            weight_root=str(d["model"]["weight_root"]),
            input_tokens=list(d["input"]["tokens"]),
            input_hash=str(d["input"]["hash"]),
            output_logits_hash=str(d["output"]["logits_hash"]),
            predicted_token=d["output"].get("predicted_token"),
            all_logits=d["output"].get("all_logits"),
            n_kernels=int(d["trace"]["n_kernels"]),
            chain_head=str(d["trace"]["chain_head"]),
            merkle_root=str(d["trace"]["merkle_root"]),
            kernel_summary=list(d["trace"]["kernel_summary"]),
            metadata=dict(d.get("metadata", {})),
            signature=dict(d.get("signature", {"algo": "none", "key_id": None, "value": None})),
            full_trace=d.get("full_trace"),
        )

    def signed_payload_bytes(self) -> bytes:
        """Bytes covered by ``signature``. Excludes the signature itself."""
        payload = self.to_dict()
        payload.pop("signature", None)
        return canonical_json_bytes(payload)

    def sign_hmac(self, key: bytes, key_id: str = "default") -> None:
        """Mutate ``self.signature`` to a HMAC-SHA256 over the canonical payload.

        HMAC was chosen to keep the demo dependency-free; production users
        should swap in Ed25519 by writing a new ``sign_*`` method and a
        matching verifier in :func:`verify_signature`.
        """
        mac = hmac.new(key, self.signed_payload_bytes(), hashlib.sha256).digest()
        self.signature = {"algo": "hmac-sha256", "key_id": key_id, "value": mac.hex()}

    def verify_signature(self, key: bytes) -> bool:
        algo = self.signature.get("algo")
        if algo == "none":
            return True
        if algo != "hmac-sha256":
            raise ValueError(f"unsupported signature algo: {algo!r}")
        expected = hmac.new(key, self.signed_payload_bytes(), hashlib.sha256).digest()
        got = bytes.fromhex(self.signature["value"])
        return hmac.compare_digest(expected, got)


def save_certificate(cert: Certificate, path: str) -> None:
    """Write ``cert`` to ``path`` as pretty JSON."""
    with open(path, "w", encoding="ascii") as f:
        json.dump(cert.to_dict(), f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def load_certificate(path: str) -> Certificate:
    with open(path, encoding="ascii") as f:
        return Certificate.from_dict(json.load(f))


def standard_metadata() -> dict[str, Any]:
    """Best-effort environment fingerprint. None of these fields are
    security-relevant — they're for human auditors triaging mismatches."""
    return {
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform_mod.platform(),
        "machine": platform_mod.machine(),
        "tracer_version": "verifiable_inference/0.1.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def hash_token_input(tokens: np.ndarray) -> str:
    """Canonical hash of an int64 token vector — used as the public input id."""
    if tokens.dtype != np.dtype("<i8"):
        tokens = tokens.astype("<i8")
    return hash_array(tokens).hex()


def kernel_summary_from_records(records: list[KernelRecord]) -> list[dict]:
    """Compact public-facing trace digest. Holds enough to spot-check what
    ops ran and in what order, without revealing tensor contents."""
    return [
        {"seq": r.seq, "op": r.op, "output_hash": r.output_hash.hex()}
        for r in records
    ]
