"""Merkle tree: roots are stable, proofs verify, single-bit changes break verification."""

from __future__ import annotations

import hashlib

import pytest

from verifiable_inference.merkle import MerkleTree, verify_merkle_proof


def _h(s: str) -> bytes:
    return hashlib.sha256(s.encode()).digest()


def test_single_leaf_root():
    t = MerkleTree([_h("a")])
    assert t.n_leaves == 1
    assert isinstance(t.root, bytes) and len(t.root) == 32


def test_proofs_verify_for_all_leaves():
    leaves = [_h(f"leaf-{i}") for i in range(7)]  # odd count exercises duplicate-last
    tree = MerkleTree(leaves)
    for i, payload in enumerate(leaves):
        proof = tree.proof(i)
        assert verify_merkle_proof(payload, proof, tree.root)


def test_wrong_payload_rejected():
    leaves = [_h(f"leaf-{i}") for i in range(8)]
    tree = MerkleTree(leaves)
    proof = tree.proof(3)
    assert not verify_merkle_proof(_h("not-leaf-3"), proof, tree.root)


def test_swapped_position_rejected():
    leaves = [_h(f"leaf-{i}") for i in range(8)]
    tree = MerkleTree(leaves)
    proof = tree.proof(3)
    # Try to claim leaf 4's payload is at position 3.
    assert not verify_merkle_proof(leaves[4], proof, tree.root)


def test_leaf_change_changes_root():
    leaves = [_h(f"leaf-{i}") for i in range(8)]
    t1 = MerkleTree(leaves)
    leaves2 = leaves.copy()
    leaves2[5] = _h("tampered")
    t2 = MerkleTree(leaves2)
    assert t1.root != t2.root


def test_empty_rejected():
    with pytest.raises(ValueError):
        MerkleTree([])


def test_bad_leaf_size_rejected():
    with pytest.raises(ValueError):
        MerkleTree([b"\x00" * 31])
