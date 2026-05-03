"""Binary Merkle tree over arbitrary 32-byte leaf digests.

Used to commit to the entire kernel-trace in a single 32-byte root that
also supports O(log n) inclusion proofs. The construction is the standard
"duplicate-last-on-odd" variant; leaves and internal nodes are domain-
separated so a leaf hash can never collide with an internal-node hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

_LEAF_TAG = b"\x00"
_NODE_TAG = b"\x01"


def _hash_leaf(payload: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(_LEAF_TAG)
    h.update(payload)
    return h.digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(_NODE_TAG)
    h.update(left)
    h.update(right)
    return h.digest()


@dataclass(frozen=True)
class MerkleProof:
    """Inclusion proof for a single leaf.

    ``siblings[i]`` is the sibling node at level ``i`` (level 0 = leaf).
    ``directions[i]`` is True if the sibling is on the right (i.e. our node
    is on the left) and False otherwise.
    """

    leaf_index: int
    leaf_hash: bytes
    siblings: list[bytes]
    directions: list[bool]


class MerkleTree:
    """Immutable binary Merkle tree.

    Construction takes O(n) hashes; the root and any inclusion proof are
    available in O(1) and O(log n) respectively after construction.
    """

    def __init__(self, leaves: list[bytes]) -> None:
        if not leaves:
            raise ValueError("Merkle tree requires at least one leaf")
        for i, leaf in enumerate(leaves):
            if not isinstance(leaf, (bytes, bytearray)):
                raise TypeError(f"leaf {i} is not bytes")
            if len(leaf) != 32:
                raise ValueError(f"leaf {i} has length {len(leaf)}, expected 32")
        leaf_hashes = [_hash_leaf(bytes(leaf)) for leaf in leaves]
        levels: list[list[bytes]] = [leaf_hashes]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]
                nxt.append(_hash_node(left, right))
            levels.append(nxt)
        self._levels = levels
        self._n_leaves = len(leaves)

    @property
    def root(self) -> bytes:
        return self._levels[-1][0]

    @property
    def n_leaves(self) -> int:
        return self._n_leaves

    def leaf_hash(self, index: int) -> bytes:
        if index < 0 or index >= self._n_leaves:
            raise IndexError(index)
        return self._levels[0][index]

    def proof(self, index: int) -> MerkleProof:
        if index < 0 or index >= self._n_leaves:
            raise IndexError(index)
        siblings: list[bytes] = []
        directions: list[bool] = []
        cur = index
        for level in self._levels[:-1]:
            if cur % 2 == 0:
                # we're on the left; sibling is on the right (or duplicate of self)
                sib_idx = cur + 1 if cur + 1 < len(level) else cur
                siblings.append(level[sib_idx])
                directions.append(True)
            else:
                siblings.append(level[cur - 1])
                directions.append(False)
            cur //= 2
        return MerkleProof(
            leaf_index=index,
            leaf_hash=self._levels[0][index],
            siblings=siblings,
            directions=directions,
        )


def verify_merkle_proof(leaf_payload: bytes, proof: MerkleProof, root: bytes) -> bool:
    """Independently verify ``leaf_payload`` is at ``proof.leaf_index`` of a tree
    with the given ``root``."""
    expected_leaf = _hash_leaf(leaf_payload)
    if expected_leaf != proof.leaf_hash:
        return False
    cur = expected_leaf
    for sib, sib_on_right in zip(proof.siblings, proof.directions):
        cur = _hash_node(cur, sib) if sib_on_right else _hash_node(sib, cur)
    return cur == root
