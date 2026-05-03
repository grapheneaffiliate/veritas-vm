"""Verifiable autoregressive generation: linked per-token certificates.

A real LLM generates many tokens. Each token requires its own forward
pass over the growing prompt. We emit one ``Certificate`` per generation
step and link them into a *session chain*: each step's certificate
includes ``parent_session_hash`` = hash of the previous step's signed
payload. The session has a single 32-byte ``transcript_root`` (Merkle
root over per-step certificate hashes) that commits to the entire
generated sequence.

Properties:
- Tampering with any step breaks the link to its successor (transcript
  root no longer matches).
- A verifier can confirm a single step's authenticity given just that
  step's certificate plus its inclusion proof in the transcript.
- The session is replay-able token-by-token: re-running with the same
  prompt + same model + same step index produces the same certificate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

from .canonical import canonical_json_bytes, hash_bytes
from .certificate import Certificate, save_certificate
from .merkle import MerkleProof, MerkleTree, verify_merkle_proof
from .model import ModelWeights, greedy_next_token
from .prover import Prover

GENESIS_SESSION_HASH: bytes = b"\x00" * 32


def _step_payload_bytes(cert: Certificate) -> bytes:
    """Bytes that uniquely identify a step. Used as the Merkle leaf and
    as the chain-link preimage for the next step."""
    return cert.signed_payload_bytes()


def _step_hash(cert: Certificate) -> bytes:
    return hash_bytes(_step_payload_bytes(cert))


@dataclass
class StepCertificate:
    """One generation step. Wraps a per-token ``Certificate`` plus the
    session-chain link from the previous step."""

    step_index: int
    parent_session_hash: bytes  # 32 zero bytes for the first step
    session_hash: bytes  # = hash_bytes(parent_session_hash || step_payload)
    cert: Certificate

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "parent_session_hash": self.parent_session_hash.hex(),
            "session_hash": self.session_hash.hex(),
            "cert": self.cert.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StepCertificate":
        return StepCertificate(
            step_index=int(d["step_index"]),
            parent_session_hash=bytes.fromhex(d["parent_session_hash"]),
            session_hash=bytes.fromhex(d["session_hash"]),
            cert=Certificate.from_dict(d["cert"]),
        )


@dataclass
class SessionTranscript:
    """A complete autoregressive generation. Verifiers should treat a
    transcript as the unit of audit — it captures the full prompt + every
    generated token + the entire kernel trail."""

    model_weight_root: str
    prompt_tokens: list[int]
    steps: list[StepCertificate] = field(default_factory=list)
    final_session_hash: bytes = GENESIS_SESSION_HASH
    transcript_root: bytes = b""  # set when finalize() is called

    @property
    def generated_tokens(self) -> list[int]:
        toks = []
        for s in self.steps:
            t = s.cert.predicted_token
            if t is not None:
                toks.append(int(t))
        return toks

    @property
    def all_tokens(self) -> list[int]:
        return list(self.prompt_tokens) + self.generated_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "vai-session/1.0",
            "model_weight_root": self.model_weight_root,
            "prompt_tokens": list(self.prompt_tokens),
            "steps": [s.to_dict() for s in self.steps],
            "final_session_hash": self.final_session_hash.hex(),
            "transcript_root": self.transcript_root.hex(),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SessionTranscript":
        return SessionTranscript(
            model_weight_root=str(d["model_weight_root"]),
            prompt_tokens=list(d["prompt_tokens"]),
            steps=[StepCertificate.from_dict(s) for s in d["steps"]],
            final_session_hash=bytes.fromhex(d["final_session_hash"]),
            transcript_root=bytes.fromhex(d["transcript_root"]),
        )

    def step_proof(self, step_index: int) -> tuple[StepCertificate, MerkleProof]:
        """Inclusion proof for one step in the transcript Merkle tree."""
        if not self.transcript_root:
            raise ValueError("transcript not finalized")
        leaves = [_step_hash(s.cert) for s in self.steps]
        tree = MerkleTree(leaves)
        return self.steps[step_index], tree.proof(step_index)


def generate(
    prover: Prover,
    prompt: np.ndarray,
    *,
    max_new_tokens: int,
    sign_key: Optional[bytes] = None,
    sign_algo: str = "ed25519",
    key_id: str = "default",
    eos_token: Optional[int] = None,
    include_full_trace: bool = True,
    on_step=None,
) -> SessionTranscript:
    """Run autoregressive greedy generation. Returns the full transcript.

    Each step's certificate is signed (if ``sign_key`` is provided) and
    chained to the previous step. The final transcript has a Merkle root
    that commits to the entire generation."""
    if prompt.ndim != 1:
        raise ValueError("prompt must be 1D")
    if prompt.dtype != np.dtype("<i8"):
        prompt = prompt.astype("<i8")

    transcript = SessionTranscript(
        model_weight_root=prover.weight_root,
        prompt_tokens=[int(t) for t in prompt],
    )
    parent = GENESIS_SESSION_HASH
    cur = prompt.copy()

    for step_index in range(max_new_tokens):
        if cur.shape[0] > prover.model.config.max_seq_len:
            # Slide the window — keep the last max_seq_len tokens.
            cur = cur[-prover.model.config.max_seq_len :].copy()
        _, cert = prover.run(
            cur,
            include_full_trace=include_full_trace,
            sign_key=sign_key,
            sign_algo=sign_algo,
            key_id=key_id,
        )
        # Bind the step into the session chain by hashing parent || step_payload.
        step_payload = _step_payload_bytes(cert)
        new_session_hash = hash_bytes(parent + step_payload)

        step = StepCertificate(
            step_index=step_index,
            parent_session_hash=parent,
            session_hash=new_session_hash,
            cert=cert,
        )
        transcript.steps.append(step)
        parent = new_session_hash

        next_tok = cert.predicted_token
        if next_tok is None:
            break
        if on_step is not None:
            on_step(step_index, int(next_tok), cert)
        if eos_token is not None and int(next_tok) == int(eos_token):
            break
        cur = np.concatenate([cur, np.array([next_tok], dtype="<i8")])

    leaves = [_step_hash(s.cert) for s in transcript.steps]
    if leaves:
        transcript.transcript_root = MerkleTree(leaves).root
    transcript.final_session_hash = parent
    return transcript


def verify_transcript(
    transcript: SessionTranscript,
    *,
    expected_weight_root: Optional[str] = None,
    public_key: Optional[bytes] = None,
) -> bool:
    """Independently verify the session chain *and* (if a public key is
    given) every per-step Ed25519 signature.

    Does not re-run inference — that's :class:`StreamingVerifier`. This
    function checks the transcript-level invariants only.
    """
    if expected_weight_root is not None and transcript.model_weight_root != expected_weight_root:
        return False

    parent = GENESIS_SESSION_HASH
    leaves = []
    for i, step in enumerate(transcript.steps):
        if step.step_index != i:
            return False
        if step.parent_session_hash != parent:
            return False
        # Each step's cert must agree with this transcript's weight root.
        if step.cert.weight_root != transcript.model_weight_root:
            return False
        if public_key is not None and not step.cert.verify_signature(public_key):
            return False
        elif step.cert.signature.get("algo") not in (None, "none") and public_key is None:
            # If the cert is signed, verify it against its own embedded key
            # (only valid for ed25519, where pubkey is in the cert).
            if step.cert.signature.get("algo") == "ed25519":
                if not step.cert.verify_signature():
                    return False
        step_payload = _step_payload_bytes(step.cert)
        expected_session = hash_bytes(parent + step_payload)
        if step.session_hash != expected_session:
            return False
        leaves.append(hash_bytes(step_payload))
        parent = expected_session

    if transcript.final_session_hash != parent:
        return False

    if leaves:
        recomputed_root = MerkleTree(leaves).root
        if recomputed_root != transcript.transcript_root:
            return False
    elif transcript.transcript_root != b"":
        return False

    return True


def verify_step_inclusion(
    transcript_root: bytes,
    step: StepCertificate,
    proof: MerkleProof,
) -> bool:
    """Verify that ``step`` is in the transcript whose Merkle root is
    ``transcript_root``, given an inclusion proof. The transcript itself
    need not be present — only the root."""
    if step.step_index != proof.leaf_index:
        return False
    leaf_input = hash_bytes(_step_payload_bytes(step.cert))
    return verify_merkle_proof(leaf_input, proof, transcript_root)


def save_transcript(transcript: SessionTranscript, path: str) -> None:
    with open(path, "w", encoding="ascii") as f:
        json.dump(transcript.to_dict(), f, indent=2, sort_keys=True, ensure_ascii=True)
        f.write("\n")


def load_transcript(path: str) -> SessionTranscript:
    with open(path, encoding="ascii") as f:
        return SessionTranscript.from_dict(json.load(f))
