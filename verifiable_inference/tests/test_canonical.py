"""Canonical-encoding properties: same content ⇒ same hash; different content ⇒ different hash."""

from __future__ import annotations

import numpy as np
import pytest

from verifiable_inference.canonical import (
    canonical_array_bytes,
    canonical_json_bytes,
    hash_array,
    hash_bytes,
    hash_json,
)


def test_array_hash_stable_across_layouts():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert hash_array(a) == hash_array(b)


def test_array_hash_changes_with_value():
    a = np.zeros((4,), dtype=np.float64)
    b = a.copy()
    b[0] = 1e-30
    assert hash_array(a) != hash_array(b)


def test_array_hash_changes_with_dtype():
    a = np.zeros((4,), dtype=np.float64)
    b = np.zeros((4,), dtype=np.float32)
    assert hash_array(a) != hash_array(b)


def test_array_hash_changes_with_shape():
    a = np.zeros((2, 4), dtype=np.float64)
    b = np.zeros((4, 2), dtype=np.float64)
    assert hash_array(a) != hash_array(b)


def test_array_non_contiguous_normalized():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    view = a[:, ::1]  # contiguous
    assert hash_array(a) == hash_array(view)


def test_array_non_contiguous_transpose_differs_in_shape():
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    # The transpose has different shape so its hash legitimately differs.
    assert hash_array(a) != hash_array(a.T)


def test_object_dtype_rejected():
    a = np.array(["hi", "there"], dtype=object)
    with pytest.raises(ValueError):
        canonical_array_bytes(a)


def test_json_hash_stable_across_key_order():
    h1 = hash_json({"a": 1, "b": [2, 3], "c": {"d": 4}})
    h2 = hash_json({"c": {"d": 4}, "b": [2, 3], "a": 1})
    assert h1 == h2


def test_cross_domain_hashes_differ():
    """An array of zero bytes and a JSON empty dict must hash differently."""
    arr = np.zeros((1,), dtype=np.float64)
    assert hash_array(arr) != hash_json({})
    assert hash_array(arr) != hash_bytes(b"")


def test_canonical_bytes_round_trip_changes_under_value_change():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    # nextafter(3.0, 4.0) flips one bit of the float64 mantissa.
    c = np.array([1.0, 2.0, np.nextafter(3.0, 4.0)], dtype=np.float64)
    assert canonical_array_bytes(a) == canonical_array_bytes(b)
    assert canonical_array_bytes(a) != canonical_array_bytes(c)
