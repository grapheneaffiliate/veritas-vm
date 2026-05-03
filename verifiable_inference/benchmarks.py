"""Benchmarks: certificate size, prove time, verify time at varying model sizes.

Run:
    uv run python -m verifiable_inference.benchmarks
"""

from __future__ import annotations

import json
import time

import numpy as np

from .fast_kernels import use_fast_kernels, use_reference_kernels
from .model import ModelConfig, init_random_weights
from .prover import Prover
from .verifier import Verifier


def _bench_one(cfg: ModelConfig, *, prompt_len: int, kernel_mode: str) -> dict:
    if kernel_mode == "fast":
        use_fast_kernels()
    else:
        use_reference_kernels()

    model = init_random_weights(cfg, seed=1)
    prover = Prover(model)
    verifier = Verifier(model)
    tokens = np.arange(prompt_len, dtype="<i8") % cfg.vocab_size

    t0 = time.perf_counter()
    _, cert = prover.run(tokens, include_full_trace=True)
    t_prove = time.perf_counter() - t0

    cert_bytes = json.dumps(cert.to_dict()).encode("utf-8")
    cert_size = len(cert_bytes)

    t0 = time.perf_counter()
    verifier.verify(cert).raise_if_failed()
    t_verify = time.perf_counter() - t0

    return {
        "kernel_mode": kernel_mode,
        "config": cfg.to_dict(),
        "prompt_len": prompt_len,
        "n_kernels": cert.n_kernels,
        "cert_bytes": cert_size,
        "prove_seconds": round(t_prove, 4),
        "verify_seconds": round(t_verify, 4),
    }


def main() -> int:
    grid = [
        # (config, prompt_len)
        (ModelConfig(vocab_size=8,  d_model=4,  n_layers=1, max_seq_len=4,  n_heads=1), 4),
        (ModelConfig(vocab_size=16, d_model=8,  n_layers=2, max_seq_len=8,  n_heads=1), 8),
        (ModelConfig(vocab_size=32, d_model=16, n_layers=2, max_seq_len=16, n_heads=2), 16),
        (ModelConfig(vocab_size=64, d_model=32, n_layers=4, max_seq_len=16, n_heads=4), 16),
        (ModelConfig(vocab_size=128, d_model=64, n_layers=4, max_seq_len=32, n_heads=8), 32),
    ]
    results = []
    print(f"{'mode':<6} {'config':<55} {'kern':>5} {'cert':>9} {'prove':>9} {'verify':>9}")
    print("-" * 100)
    for cfg, plen in grid:
        for mode in ("fast",):  # reference is too slow at the larger sizes
            try:
                r = _bench_one(cfg, prompt_len=plen, kernel_mode=mode)
            except Exception as e:
                print(f"  FAIL {cfg} {plen}: {e!r}")
                continue
            results.append(r)
            print(
                f"{r['kernel_mode']:<6} {str(r['config']):<55} "
                f"{r['n_kernels']:>5} {r['cert_bytes']:>9} "
                f"{r['prove_seconds']:>9.3f} {r['verify_seconds']:>9.3f}"
            )

    print()
    print("Summary:")
    print(f"  rows = {len(results)}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
