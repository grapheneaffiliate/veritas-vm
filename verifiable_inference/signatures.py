"""Pure-Python Ed25519 signing/verification.

Public-key signatures are the right primitive for verifiable inference:
the model owner publishes a 32-byte verification key once; anyone in the
world can then verify any certificate that key signed without ever
holding a secret. (HMAC, by contrast, requires the verifier to share the
prover's secret — which means the verifier could forge certificates.)

This is a from-RFC-8032 reference implementation. It is correct but not
fast. Production deployments should swap in libsodium / cryptography /
PyNaCl by replacing the three top-level functions:

    derive_public_key(secret_key) -> bytes
    sign(message, secret_key) -> bytes
    verify(message, signature, public_key) -> bool

The certificate format and verification protocol do not change.

References:
- RFC 8032 §5.1 (Ed25519)
- Bernstein et al., "High-speed high-security signatures" (2011)
"""

from __future__ import annotations

import hashlib
import os

# Curve constants — Ed25519 (Curve25519 in twisted-Edwards form).
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493  # group order
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _modinv(x: int) -> int:
    return pow(x, _P - 2, _P)


def _x_recover(y: int) -> int:
    xx = (y * y - 1) * _modinv(_D * y * y + 1) % _P
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P != 0:
        x = (x * _I) % _P
    if x % 2 != 0:
        x = _P - x
    return x


_BY = (4 * _modinv(5)) % _P
_BX = _x_recover(_BY)
_B = (_BX % _P, _BY % _P)


def _edwards_add(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    denom_a = _modinv(1 + _D * x1 * x2 * y1 * y2)
    denom_b = _modinv(1 - _D * x1 * x2 * y1 * y2)
    x3 = (x1 * y2 + x2 * y1) * denom_a
    y3 = (y1 * y2 + x1 * x2) * denom_b
    return (x3 % _P, y3 % _P)


def _scalar_mult(point, e: int):
    """Iterative double-and-add. Avoids Python's 1000-deep recursion limit
    and is constant-time-ish for our purposes (signatures already require
    secrecy of the private key, which we ensure via os.urandom)."""
    result = (0, 1)  # identity element of the curve
    addend = point
    while e > 0:
        if e & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        e >>= 1
    return result


def _bit(h: bytes, i: int) -> int:
    return (h[i // 8] >> (i % 8)) & 1


def _encode_int(y: int) -> bytes:
    return y.to_bytes(32, "little")


def _encode_point(point) -> bytes:
    x, y = point
    bits = [(y >> i) & 1 for i in range(255)] + [x & 1]
    out = bytearray(32)
    for i in range(32):
        out[i] = sum(bits[i * 8 + j] << j for j in range(8))
    return bytes(out)


def _decode_int(s: bytes) -> int:
    return int.from_bytes(s, "little")


def _decode_point(s: bytes):
    if len(s) != 32:
        raise ValueError("ed25519 point must be 32 bytes")
    y = _decode_int(s) & ((1 << 255) - 1)
    x = _x_recover(y)
    if x & 1 != _bit(s, 255):
        x = _P - x
    point = (x, y)
    if not _is_on_curve(point):
        raise ValueError("ed25519 point not on curve")
    return point


def _is_on_curve(point) -> bool:
    x, y = point
    return (-x * x + y * y - 1 - _D * x * x * y * y) % _P == 0


def _hash_int(*chunks: bytes) -> int:
    h = hashlib.sha512()
    for c in chunks:
        h.update(c)
    return int.from_bytes(h.digest(), "little")


# --- Public API -------------------------------------------------------------


def generate_keypair(seed: bytes | None = None) -> tuple[bytes, bytes]:
    """Generate (secret_key, public_key). 32 bytes each.

    If ``seed`` is None, uses ``os.urandom(32)``.
    The Ed25519 secret key IS the seed — the implementation derives the
    actual scalar internally.
    """
    sk = seed if seed is not None else os.urandom(32)
    if len(sk) != 32:
        raise ValueError(f"ed25519 secret key must be 32 bytes, got {len(sk)}")
    return sk, derive_public_key(sk)


def derive_public_key(secret_key: bytes) -> bytes:
    if len(secret_key) != 32:
        raise ValueError("secret_key must be 32 bytes")
    h = _sha512(secret_key)
    a = (1 << 254) + sum((1 << i) * _bit(h, i) for i in range(3, 254))
    return _encode_point(_scalar_mult(_B, a))


def sign(message: bytes, secret_key: bytes) -> bytes:
    if len(secret_key) != 32:
        raise ValueError("secret_key must be 32 bytes")
    h = _sha512(secret_key)
    a = (1 << 254) + sum((1 << i) * _bit(h, i) for i in range(3, 254))
    pk = _encode_point(_scalar_mult(_B, a))
    r = _hash_int(h[32:64], message)
    R = _scalar_mult(_B, r)
    encR = _encode_point(R)
    k = _hash_int(encR, pk, message)
    s = (r + k * a) % _L
    return encR + _encode_int(s)


def verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Constant-time-ish signature verification. Returns True/False; never
    raises on malformed inputs (returns False instead) so callers can rely
    on a single boolean."""
    if len(signature) != 64 or len(public_key) != 32:
        return False
    try:
        R = _decode_point(signature[:32])
        A = _decode_point(public_key)
    except ValueError:
        return False
    s = _decode_int(signature[32:])
    if s >= _L:
        return False  # malleability rejection
    k = _hash_int(signature[:32], public_key, message)
    lhs = _scalar_mult(_B, s)
    rhs = _edwards_add(R, _scalar_mult(A, k))
    return lhs == rhs
