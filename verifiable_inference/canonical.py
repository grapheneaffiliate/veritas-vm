"""Canonical, deterministic byte/hash encodings.

Two arrays with the same logical content must produce identical bytes here
regardless of memory layout, endianness of the host, or NumPy version. This
is the foundation that everything else (trace, Merkle, certificate) relies on.

Every function is pure and side-effect free.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import numpy as np

# Magic bytes prefixed to every canonical encoding so a hash collision
# between (e.g.) an array and a JSON object is impossible by construction.
_MAGIC_ARRAY = b"\x00VAI\x01ARR\x00"
_MAGIC_BYTES = b"\x00VAI\x01BYT\x00"
_MAGIC_JSON = b"\x00VAI\x01JSN\x00"

# Allowlist of dtypes we accept. Object/string arrays are rejected because
# their byte layout is platform/version dependent.
_ALLOWED_DTYPES = frozenset(
    [
        "<f8", "<f4", "<f2",
        "<i8", "<i4", "<i2", "<i1",
        "<u8", "<u4", "<u2", "<u1",
        "|b1",
    ]
)


def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
    """Return a little-endian, contiguous view with an allowlisted dtype."""
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    elif arr.dtype.byteorder == "=":
        # Native — force to little-endian explicitly.
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    dt = arr.dtype.str
    if dt not in _ALLOWED_DTYPES:
        raise ValueError(
            f"dtype {dt!r} not in canonical allowlist; got from array of shape {arr.shape}"
        )
    return np.ascontiguousarray(arr)


def canonical_array_bytes(arr: np.ndarray) -> bytes:
    """Encode ``arr`` to a canonical byte string.

    Layout: MAGIC || dtype_str (4 bytes ascii) || ndim (uint32 LE)
            || shape[0..ndim] (uint64 LE each) || raw little-endian payload.
    """
    arr = _normalize_dtype(arr)
    parts = [_MAGIC_ARRAY]
    dtype_str = arr.dtype.str.encode("ascii")
    if len(dtype_str) != 3 and len(dtype_str) != 4:
        # Defensive — every dtype in the allowlist is 3 or 4 chars.
        raise ValueError(f"unexpected dtype string length: {dtype_str!r}")
    parts.append(struct.pack("<B", len(dtype_str)))
    parts.append(dtype_str)
    parts.append(struct.pack("<I", arr.ndim))
    for d in arr.shape:
        parts.append(struct.pack("<Q", int(d)))
    parts.append(arr.tobytes(order="C"))
    return b"".join(parts)


def hash_array(arr: np.ndarray) -> bytes:
    """SHA-256 of the canonical encoding of ``arr``."""
    return hashlib.sha256(canonical_array_bytes(arr)).digest()


def hash_bytes(data: bytes) -> bytes:
    """SHA-256 of an opaque byte string, with a magic prefix to prevent
    cross-domain collisions with array/JSON hashes."""
    h = hashlib.sha256()
    h.update(_MAGIC_BYTES)
    h.update(struct.pack("<Q", len(data)))
    h.update(data)
    return h.digest()


def canonical_json_bytes(obj: Any) -> bytes:
    """Encode JSON-serializable object deterministically.

    Sorts keys, no whitespace, ensures ASCII-safe output. Two semantically
    equal objects produce identical bytes.
    """
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _MAGIC_JSON + struct.pack("<Q", len(payload)) + payload


def hash_json(obj: Any) -> bytes:
    """SHA-256 of the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json_bytes(obj)).digest()


def hex(digest: bytes) -> str:
    """Convenience: 64-char lowercase hex of a 32-byte digest."""
    return digest.hex()


def from_hex(s: str) -> bytes:
    """Inverse of ``hex``."""
    b = bytes.fromhex(s)
    if len(b) != 32:
        raise ValueError(f"expected 32-byte digest, got {len(b)}")
    return b
