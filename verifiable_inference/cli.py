"""CLI entry point: ``python -m verifiable_inference.cli prove|verify [...]``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from .certificate import load_certificate, save_certificate
from .model import ModelConfig, init_random_weights
from .prover import Prover
from .verifier import VerificationError, Verifier, verify_certificate


def _build_demo_model(seed: int) -> tuple:
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=2, max_seq_len=8)
    return init_random_weights(config, seed=seed), config


def cmd_prove(args: argparse.Namespace) -> int:
    model, _ = _build_demo_model(args.seed)
    tokens = np.asarray(json.loads(args.tokens), dtype="<i8")
    prover = Prover(model)
    sign_key = args.sign_key.encode() if args.sign_key else None
    _, cert = prover.run(
        tokens,
        include_full_trace=not args.no_full_trace,
        include_logits=args.include_logits,
        sign_key=sign_key,
    )
    save_certificate(cert, args.output)
    print(f"wrote certificate: {args.output}")
    print(f"  weight_root:  {cert.weight_root}")
    print(f"  merkle_root:  {cert.merkle_root}")
    print(f"  predicted:    {cert.predicted_token}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cert = load_certificate(args.certificate)
    sign_key = args.sign_key.encode() if args.sign_key else None
    if args.with_weights:
        model, _ = _build_demo_model(args.seed)
        verifier = Verifier(model)
        try:
            report = verifier.verify(cert, sign_key=sign_key)
            report.raise_if_failed()
        except VerificationError as e:
            print(f"FAIL ({e.code}): {e}")
            return 1
    else:
        try:
            report = verify_certificate(cert, sign_key=sign_key)
            report.raise_if_failed()
        except VerificationError as e:
            print(f"FAIL ({e.code}): {e}")
            return 1
    print("OK  certificate verified")
    for k, v in report.checks.items():
        print(f"      {k}: {v}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="verifiable_inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prove", help="run inference, emit a certificate")
    pp.add_argument("--seed", type=int, default=42, help="weight init seed")
    pp.add_argument("--tokens", type=str, required=True, help="JSON list of int tokens")
    pp.add_argument("--output", "-o", type=str, required=True, help="cert output path")
    pp.add_argument("--no-full-trace", action="store_true", help="omit full_trace")
    pp.add_argument("--include-logits", action="store_true", help="include logits in cert")
    pp.add_argument("--sign-key", type=str, default=None, help="HMAC key for signing")
    pp.set_defaults(func=cmd_prove)

    pv = sub.add_parser("verify", help="verify a certificate")
    pv.add_argument("certificate", type=str)
    pv.add_argument(
        "--with-weights",
        action="store_true",
        help="re-run inference under the demo model (full re-derivation)",
    )
    pv.add_argument("--seed", type=int, default=42)
    pv.add_argument("--sign-key", type=str, default=None)
    pv.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
