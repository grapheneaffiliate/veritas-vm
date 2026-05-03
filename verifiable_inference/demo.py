"""End-to-end demo: prove + verify + show that any tampering is detected.

Run with:
    uv run python -m verifiable_inference.demo
or:
    python -m verifiable_inference.demo
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from .certificate import load_certificate, save_certificate
from .model import ModelConfig, init_random_weights
from .prover import Prover
from .verifier import VerificationError, Verifier


def _banner(msg: str) -> None:
    print()
    print("=" * 72)
    print(msg)
    print("=" * 72)


def main() -> int:
    np.set_printoptions(precision=4, suppress=True)

    _banner("1. Build a tiny deterministic transformer")
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, max_seq_len=8)
    model = init_random_weights(config, seed=42)
    print(f"   config = {config.to_dict()}")

    prover = Prover(model)
    print(f"   weight_root = {prover.weight_root}")

    _banner("2. Run inference, emit certificate")
    tokens = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype="<i8")
    print(f"   input tokens: {tokens.tolist()}")
    logits, cert = prover.run(tokens, include_full_trace=True, include_logits=False)
    print(f"   predicted next token: {cert.predicted_token}")
    print(f"   n_kernels:            {cert.n_kernels}")
    print(f"   chain_head:           {cert.chain_head}")
    print(f"   merkle_root:          {cert.merkle_root}")

    sign_key = b"demo-shared-secret-replace-with-ed25519-in-prod"
    cert.sign_hmac(sign_key, key_id="demo")
    print(f"   signed:               algo={cert.signature['algo']}")

    with tempfile.TemporaryDirectory() as td:
        cert_path = os.path.join(td, "certificate.json")
        save_certificate(cert, cert_path)
        size = os.path.getsize(cert_path)
        print(f"   certificate written: {cert_path} ({size} bytes)")

        _banner("3. Independent verifier (full re-derivation)")
        verifier = Verifier(model)
        loaded = load_certificate(cert_path)
        report = verifier.verify(loaded, sign_key=sign_key)
        report.raise_if_failed()
        print("   OK  every check passed:")
        for k, v in report.checks.items():
            print(f"        {k}: {v}")

        _banner("4. Tamper detection — flip one byte in the output")
        d = loaded.to_dict()
        old_hash = d["output"]["logits_hash"]
        d["output"]["logits_hash"] = "0" * 64
        from .certificate import Certificate

        tampered = Certificate.from_dict(d)
        try:
            verifier.verify(tampered).raise_if_failed()
            print("   ERROR: tampered certificate verified — bug!")
            return 1
        except VerificationError as e:
            print(f"   OK  rejected (code={e.code})")
            print(f"        old logits_hash = {old_hash}")
            print(f"        new logits_hash = {'0' * 64}")

        _banner("5. Tamper detection — drop the last kernel from the trace")
        d2 = loaded.to_dict()
        d2["full_trace"] = d2["full_trace"][:-1]
        d2["trace"]["n_kernels"] -= 1
        truncated = Certificate.from_dict(d2)
        try:
            verifier.verify(truncated).raise_if_failed()
            print("   ERROR: truncated certificate verified — bug!")
            return 1
        except VerificationError as e:
            print(f"   OK  rejected (code={e.code})")

        _banner("6. Tamper detection — swap one weight in the model")
        bad_model = init_random_weights(config, seed=42)
        bad_model.tok_embed[0, 0] = bad_model.tok_embed[0, 0] + 1e-9
        bad_verifier = Verifier(bad_model)
        try:
            bad_verifier.verify(loaded).raise_if_failed()
            print("   ERROR: certificate verified against tampered weights — bug!")
            return 1
        except VerificationError as e:
            print(f"   OK  rejected (code={e.code})")

        _banner("7. Determinism — re-run the prover, expect identical certificate")
        logits2, cert2 = Prover(model).run(tokens, include_full_trace=True)
        same_root = cert2.merkle_root == loaded.merkle_root
        same_chain = cert2.chain_head == loaded.chain_head
        same_output = cert2.output_logits_hash == loaded.output_logits_hash
        print(f"   merkle_root reproduced: {same_root}")
        print(f"   chain_head reproduced:  {same_chain}")
        print(f"   logits_hash reproduced: {same_output}")
        if not (same_root and same_chain and same_output):
            print("   ERROR: rerun produced a different certificate")
            return 1

    _banner("DONE — verifiable inference works end-to-end")
    print()
    print("This is the layer that has been missing from the AI stack.")
    print("Every output now carries a portable, mathematically-binding")
    print("proof of which model produced it from which input.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
