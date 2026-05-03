"""Hash-chained execution trace.

Each kernel invocation produces a ``KernelRecord`` containing canonical
hashes of its inputs, weights, parameters, and output. Records are linked
by ``chain_hash``: each record's chain hash equals SHA-256(prev_chain_hash
|| canonical_record_bytes), so any tampering with a single record breaks
the chain from that point on.

Two layers of integrity:
  1. Hash chain — sequential, O(1) per record, detects any past tampering.
  2. Merkle root over the chain — supports inclusion proofs for any single
     kernel call without revealing the rest of the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canonical import canonical_json_bytes, hash_array, hash_bytes


def _canonical_param(value: Any) -> Any:
    """Coerce kernel parameters to JSON-canonical types."""
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        # Reject non-finite floats: they break canonical JSON.
        if not np.isfinite(value):
            raise ValueError(f"non-finite float in kernel param: {value}")
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical_param(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical_param(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        if not np.isfinite(f):
            raise ValueError(f"non-finite float in kernel param: {f}")
        return f
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"unsupported param type: {type(value).__name__}")


@dataclass(frozen=True)
class KernelRecord:
    """One entry in the execution trace.

    All ``*_hash`` fields are 32-byte SHA-256 digests. ``params`` holds
    scalar/string/list parameters (e.g. axis, eps) — never tensors.
    """

    seq: int
    op: str
    input_hashes: tuple[bytes, ...]
    weight_hashes: tuple[bytes, ...]
    params: dict[str, Any]
    output_hash: bytes
    prev_chain_hash: bytes
    chain_hash: bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "op": self.op,
            "input_hashes": [h.hex() for h in self.input_hashes],
            "weight_hashes": [h.hex() for h in self.weight_hashes],
            "params": self.params,
            "output_hash": self.output_hash.hex(),
            "prev_chain_hash": self.prev_chain_hash.hex(),
            "chain_hash": self.chain_hash.hex(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "KernelRecord":
        return KernelRecord(
            seq=int(d["seq"]),
            op=str(d["op"]),
            input_hashes=tuple(bytes.fromhex(h) for h in d["input_hashes"]),
            weight_hashes=tuple(bytes.fromhex(h) for h in d["weight_hashes"]),
            params=dict(d["params"]),
            output_hash=bytes.fromhex(d["output_hash"]),
            prev_chain_hash=bytes.fromhex(d["prev_chain_hash"]),
            chain_hash=bytes.fromhex(d["chain_hash"]),
        )

    def canonical_payload(self) -> bytes:
        """Bytes used both as the chain-link preimage and as the Merkle leaf."""
        return canonical_json_bytes(
            {
                "seq": self.seq,
                "op": self.op,
                "input_hashes": [h.hex() for h in self.input_hashes],
                "weight_hashes": [h.hex() for h in self.weight_hashes],
                "params": self.params,
                "output_hash": self.output_hash.hex(),
                "prev_chain_hash": self.prev_chain_hash.hex(),
            }
        )


# 32 zero bytes — defined chain head before any record is appended.
GENESIS_CHAIN_HASH: bytes = b"\x00" * 32


@dataclass
class ExecutionTrace:
    """Mutable hash-chained recorder.

    Use ``record(...)`` while running inference; freeze it by reading
    ``records`` and ``chain_head`` at the end.
    """

    records: list[KernelRecord] = field(default_factory=list)
    _chain_head: bytes = GENESIS_CHAIN_HASH

    @property
    def chain_head(self) -> bytes:
        return self._chain_head

    def __len__(self) -> int:
        return len(self.records)

    def record(
        self,
        op: str,
        inputs: list[np.ndarray],
        weights: list[np.ndarray],
        params: dict[str, Any],
        output: np.ndarray,
    ) -> KernelRecord:
        """Append a kernel record. Returns the record so callers can inspect it."""
        in_hashes = tuple(hash_array(x) for x in inputs)
        w_hashes = tuple(hash_array(w) for w in weights)
        out_hash = hash_array(output)
        canonical_params = _canonical_param(params) if params else {}

        seq = len(self.records)
        # Build the record with a placeholder chain_hash so we can hash its
        # canonical payload, then insert the real chain_hash.
        prelim = KernelRecord(
            seq=seq,
            op=op,
            input_hashes=in_hashes,
            weight_hashes=w_hashes,
            params=canonical_params,
            output_hash=out_hash,
            prev_chain_hash=self._chain_head,
            chain_hash=b"",  # filled below
        )
        link_input = self._chain_head + prelim.canonical_payload()
        new_chain = hash_bytes(link_input)
        rec = KernelRecord(
            seq=seq,
            op=op,
            input_hashes=in_hashes,
            weight_hashes=w_hashes,
            params=canonical_params,
            output_hash=out_hash,
            prev_chain_hash=self._chain_head,
            chain_hash=new_chain,
        )
        self.records.append(rec)
        self._chain_head = new_chain
        return rec

    def merkle_leaves(self) -> list[bytes]:
        """Per-record canonical payload hashes — the input to ``MerkleTree``."""
        return [hash_bytes(r.canonical_payload()) for r in self.records]

    def replay_chain(self) -> bytes:
        """Independently recompute the chain head from the stored records,
        without trusting their ``chain_hash`` fields. Used by the verifier."""
        cur = GENESIS_CHAIN_HASH
        for rec in self.records:
            if rec.prev_chain_hash != cur:
                raise ValueError(
                    f"chain break at record {rec.seq}: "
                    f"prev_chain_hash mismatch (expected {cur.hex()}, "
                    f"got {rec.prev_chain_hash.hex()})"
                )
            payload = rec.canonical_payload()
            cur = hash_bytes(cur + payload)
            if cur != rec.chain_hash:
                raise ValueError(
                    f"chain break at record {rec.seq}: "
                    f"chain_hash mismatch (expected {cur.hex()}, "
                    f"got {rec.chain_hash.hex()})"
                )
        return cur
