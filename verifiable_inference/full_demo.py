"""End-to-end demo of the FULL stack.

Walks through every component:

  1. Build a multi-head transformer (using the loader's native names).
  2. Publish a registry entry signed by a publisher Ed25519 key.
  3. Run autoregressive generation under fast kernels.
  4. Sign every per-step certificate with the model's Ed25519 key.
  5. Independent verifier:
       a. Looks up weight_root in the registry.
       b. Confirms the registry entry's publisher attestation.
       c. Confirms each step certificate's pubkey matches the registry.
       d. Verifies every signature.
       e. Replays the session chain and Merkle root.
       f. Re-runs inference (full re-derivation) to confirm bit-exactness.
  6. Demonstrates selective disclosure: ship a compact transcript and
     prove a single kernel call without the rest of the trace.
  7. Demonstrates tamper-detection at every layer (signature, chain,
     merkle root, transcript link).
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from . import signatures as ed25519
from .certificate import Certificate, save_certificate
from .disclosure import (
    DisclosedKernel,
    compact_certificate,
    disclose_kernel,
    verify_disclosure,
)
from .fast_kernels import use_fast_kernels, use_reference_kernels
from .kernels import tracing
from .model import ModelConfig, init_random_weights
from .model import forward as model_forward
from .prover import Prover
from .registry import Registry, RegistryEntry, verify_certificate_against_registry
from .session import (
    SessionTranscript,
    generate as session_generate,
    verify_step_inclusion,
    verify_transcript,
)
from .trace import ExecutionTrace
from .verifier import Verifier


def _banner(msg: str) -> None:
    print()
    print("=" * 78)
    print(msg)
    print("=" * 78)


def main() -> int:
    use_fast_kernels()
    np.set_printoptions(precision=4, suppress=True)

    # ---------------------------------------------------------------- 1
    _banner("1. Build a multi-head transformer")
    cfg = ModelConfig(vocab_size=32, d_model=16, n_layers=2, max_seq_len=16, n_heads=2)
    model = init_random_weights(cfg, seed=2026)
    prover = Prover(model)
    print(f"   config       = {cfg.to_dict()}")
    print(f"   weight_root  = {prover.weight_root}")

    # ---------------------------------------------------------------- 2
    _banner("2. Publish a registry entry (publisher-signed)")
    publisher_sk, publisher_pk = ed25519.generate_keypair(seed=b"\xa1" * 32)
    model_sk, model_pk = ed25519.generate_keypair(seed=b"\xb2" * 32)
    print(f"   publisher pk = {publisher_pk.hex()}")
    print(f"   model pk     = {model_pk.hex()}")

    entry = RegistryEntry(
        weight_root=prover.weight_root,
        config=cfg.to_dict(),
        public_key=model_pk.hex(),
        name="vai-demo-tinygpt",
        version="1.0.0",
        license="Apache-2.0",
        training_data="random-init for demo",
        description="Verifiable AI inference reference model",
        model_card_url="https://example.com/vai-demo-tinygpt",
    )
    entry.attest_ed25519(publisher_sk)
    registry = Registry()
    registry.register(entry)
    print(f"   registry has {len(registry.entries)} entry; attestation verifies: {entry.verify_attestation()}")

    # ---------------------------------------------------------------- 3
    _banner("3. Autoregressive generation (5 tokens) with per-step certs")
    prompt = np.array([1, 4, 1, 5, 9, 2, 6], dtype="<i8")
    transcript = session_generate(
        prover,
        prompt,
        max_new_tokens=5,
        sign_key=model_sk,
        sign_algo="ed25519",
        include_full_trace=True,
    )
    print(f"   prompt          = {transcript.prompt_tokens}")
    print(f"   generated       = {transcript.generated_tokens}")
    print(f"   transcript_root = {transcript.transcript_root.hex()}")
    print(f"   final_session   = {transcript.final_session_hash.hex()}")
    total_kernels = sum(s.cert.n_kernels for s in transcript.steps)
    print(f"   {len(transcript.steps)} steps, {total_kernels} kernels recorded total")

    # ---------------------------------------------------------------- 4
    _banner("4. Independent verifier (registry → signatures → chain → re-run)")
    # 4a. Each step's cert must verify against the registry.
    failures = 0
    for step in transcript.steps:
        ok, reason = verify_certificate_against_registry(
            step.cert, registry, trusted_publisher_pks=[publisher_pk]
        )
        if not ok:
            print(f"   step {step.step_index} FAILED registry check: {reason}")
            failures += 1
    print(f"   registry-bound verification: {len(transcript.steps) - failures}/{len(transcript.steps)} steps pass")

    # 4b. Transcript-level chain.
    print(f"   transcript chain integrity: {verify_transcript(transcript, expected_weight_root=prover.weight_root, public_key=model_pk)}")

    # 4c. Full re-derivation on the most recent step.
    last_step = transcript.steps[-1]
    Verifier(model).verify(last_step.cert).raise_if_failed()
    print(f"   full re-derivation of last step: OK")

    # ---------------------------------------------------------------- 5
    _banner("5. Selective disclosure (compact cert + 1 kernel proof)")
    # Strip full_trace from the last step's cert.
    public_cert = compact_certificate(last_step.cert)
    print(f"   public cert size:  {len(json.dumps(public_cert.to_dict()))} bytes")
    print(f"   public cert has full_trace: {public_cert.full_trace is not None}")

    # Re-run to obtain the full trace (private to the prover).
    private_trace = ExecutionTrace()
    last_input = np.array(transcript.all_tokens[: -1 + len(transcript.prompt_tokens)], dtype="<i8")
    # Actually use the same input the last step ran on:
    if len(transcript.all_tokens) > 1:
        n_prompt = len(transcript.prompt_tokens)
        n_gen_before_last = len(transcript.steps) - 1
        last_input = np.array(
            transcript.all_tokens[: n_prompt + n_gen_before_last], dtype="<i8"
        )
        if last_input.shape[0] > cfg.max_seq_len:
            last_input = last_input[-cfg.max_seq_len :].copy()
    with tracing(private_trace):
        model_forward(last_input, model)

    # Disclose just the final kernel (the unembed linear layer).
    final_idx = len(private_trace.records) - 1
    disclosed = disclose_kernel(private_trace.records, final_idx)
    print(f"   disclosed kernel: seq={disclosed.record.seq}, op={disclosed.record.op!r}")
    print(f"   proof siblings:   {len(disclosed.proof.siblings)} levels (O(log n))")
    ok = verify_disclosure(disclosed, expected_root=bytes.fromhex(public_cert.merkle_root))
    print(f"   disclosure verifies against compact cert root: {ok}")

    # ---------------------------------------------------------------- 6
    _banner("6. Tamper detection across the whole stack")

    # 6a. Mutate one logit hash inside step 0.
    bad = SessionTranscript.from_dict(transcript.to_dict())
    bad.steps[0].cert.output_logits_hash = "0" * 64
    bad_chain = bad.to_dict()
    bad_chain["steps"][0]["cert"]["output"]["logits_hash"] = "0" * 64
    bad2 = SessionTranscript.from_dict(bad_chain)
    print(f"   mutated cert output_hash → transcript verify: {verify_transcript(bad2)} (expect False)")

    # 6b. Drop a step.
    d = transcript.to_dict()
    d["steps"] = d["steps"][:-1]
    bad3 = SessionTranscript.from_dict(d)
    print(f"   dropped final step → transcript verify:        {verify_transcript(bad3)} (expect False)")

    # 6c. Forge a registry entry with a different model pubkey.
    forge_sk, forge_pk = ed25519.generate_keypair(seed=b"\xff" * 32)
    forge_entry = RegistryEntry(
        weight_root=prover.weight_root,
        config=cfg.to_dict(),
        public_key=forge_pk.hex(),
        name="forged",
        version="1.0.0",
    )
    forge_entry.attest_ed25519(publisher_sk)  # publisher signs whatever we give them
    forged_registry = Registry()
    forged_registry.register(forge_entry)
    ok, reason = verify_certificate_against_registry(
        last_step.cert, forged_registry, trusted_publisher_pks=[publisher_pk]
    )
    print(f"   forged-registry-pubkey check:                  {ok}, reason={reason!r} (expect False)")

    # 6d. Sign with the wrong key.
    rogue_sk, _ = ed25519.generate_keypair(seed=b"\xee" * 32)
    _, rogue_cert = Prover(model).run(
        np.array([1, 2, 3, 0], dtype="<i8"),
        sign_key=rogue_sk,
        sign_algo="ed25519",
    )
    ok, reason = verify_certificate_against_registry(
        rogue_cert, registry, trusted_publisher_pks=[publisher_pk]
    )
    print(f"   rogue-signed cert vs real registry:            {ok}, reason={reason!r} (expect False)")

    # ---------------------------------------------------------------- 7
    _banner("7. DONE — every layer of the stack is operational")
    use_reference_kernels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
