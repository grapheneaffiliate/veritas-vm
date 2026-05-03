"""Prover: runs deterministic inference and emits a certificate."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .canonical import canonical_json_bytes, hash_array, hash_bytes
from .certificate import (
    Certificate,
    hash_token_input,
    kernel_summary_from_records,
    standard_metadata,
)
from .kernels import tracing
from .merkle import MerkleTree
from .model import ModelWeights, forward, greedy_next_token
from .trace import ExecutionTrace


def compute_weight_root(model: ModelWeights) -> str:
    """Hex SHA-256 over the canonical encoding of (config, all named weights).

    Two ``ModelWeights`` instances with identical config and identical tensor
    contents produce identical weight roots."""
    parts = []
    parts.append(canonical_json_bytes(model.config.to_dict()))
    for name, arr in model.all_arrays():
        parts.append(canonical_json_bytes({"name": name, "shape": list(arr.shape)}))
        parts.append(hash_array(arr))
    return hash_bytes(b"".join(parts)).hex()


class Prover:
    """Wrap a ``ModelWeights`` and produce certified inferences."""

    def __init__(self, model: ModelWeights) -> None:
        self.model = model
        self._weight_root = compute_weight_root(model)

    @property
    def weight_root(self) -> str:
        return self._weight_root

    def run(
        self,
        tokens: np.ndarray,
        *,
        include_full_trace: bool = True,
        include_logits: bool = False,
        sign_key: Optional[bytes] = None,
        key_id: str = "default",
    ) -> tuple[np.ndarray, Certificate]:
        """Execute inference under a fresh trace; return ``(logits, certificate)``."""
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be 1D, got shape {tokens.shape}")
        if tokens.dtype != np.dtype("<i8"):
            tokens = tokens.astype("<i8")

        trace = ExecutionTrace()
        with tracing(trace):
            logits = forward(tokens, self.model)

        leaves = trace.merkle_leaves()
        merkle = MerkleTree(leaves)

        kernel_summary = kernel_summary_from_records(trace.records)
        all_logits = logits.tolist() if include_logits else None
        full_trace = (
            [r.to_dict() for r in trace.records] if include_full_trace else None
        )

        cert = Certificate(
            model_config=self.model.config.to_dict(),
            weight_root=self._weight_root,
            input_tokens=[int(t) for t in tokens],
            input_hash=hash_token_input(tokens),
            output_logits_hash=hash_array(logits).hex(),
            predicted_token=greedy_next_token(logits),
            all_logits=all_logits,
            n_kernels=len(trace.records),
            chain_head=trace.chain_head.hex(),
            merkle_root=merkle.root.hex(),
            kernel_summary=kernel_summary,
            metadata=standard_metadata(),
            full_trace=full_trace,
        )
        if sign_key is not None:
            cert.sign_hmac(sign_key, key_id=key_id)

        return logits, cert
