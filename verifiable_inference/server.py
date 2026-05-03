"""Stdlib-only HTTP server exposing the prove/verify API.

Endpoints (all JSON over POST except where noted):

  GET  /health                  -> {"ok": true, "weight_root": "..."}
  GET  /model                   -> {"config": {...}, "weight_root": "..."}
  POST /prove                   request: {"tokens": [int], "include_full_trace": bool, "include_logits": bool}
                                response: certificate dict
  POST /generate                request: {"prompt": [int], "max_new_tokens": int, "eos_token": int|null}
                                response: transcript dict
  POST /verify                  request: certificate dict
                                response: {"ok": bool, "checks": {...}, "code": str|null}
  POST /verify/full             request: certificate dict
                                response: same, but the server re-runs inference
  POST /verify/transcript       request: transcript dict
                                response: {"ok": bool}

The server signs every certificate with its Ed25519 secret key (loaded
from the ``VAI_SIGN_KEY`` env var as 64 hex chars, or generated on first
start). The corresponding public key is in /model.

This is a *demo* server — single-threaded, no auth, no rate limiting.
Production deployments should put it behind a real ASGI/WSGI runtime,
add auth, and pin the key in a KMS.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from . import signatures as ed25519
from .certificate import Certificate
from .model import ModelConfig, init_random_weights
from .prover import Prover
from .session import (
    SessionTranscript,
    generate as session_generate,
    verify_transcript,
)
from .verifier import VerificationError, Verifier, verify_certificate


def _load_or_create_key() -> tuple[bytes, bytes]:
    raw = os.environ.get("VAI_SIGN_KEY")
    if raw:
        sk = bytes.fromhex(raw)
        if len(sk) != 32:
            raise ValueError("VAI_SIGN_KEY must be 32 bytes hex (64 chars)")
    else:
        sk, _ = ed25519.generate_keypair()
    return sk, ed25519.derive_public_key(sk)


def build_demo_app(seed: int = 42) -> "AppState":
    cfg = ModelConfig(vocab_size=16, d_model=8, n_layers=2, max_seq_len=8)
    weights = init_random_weights(cfg, seed=seed)
    sk, pk = _load_or_create_key()
    return AppState(prover=Prover(weights), config=cfg, secret_key=sk, public_key=pk)


class AppState:
    def __init__(self, *, prover: Prover, config: ModelConfig, secret_key: bytes, public_key: bytes):
        self.prover = prover
        self.config = config
        self.secret_key = secret_key
        self.public_key = public_key


def _make_handler(app: AppState):
    class Handler(BaseHTTPRequestHandler):
        # Suppress default access-log noise during tests.
        def log_message(self, fmt, *args):  # noqa: ARG002
            return

        # ---- helpers ----------------------------------------------------
        def _send_json(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, sort_keys=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw)

        # ---- routing ----------------------------------------------------
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"ok": True, "weight_root": app.prover.weight_root})
            elif self.path == "/model":
                self._send_json(
                    200,
                    {
                        "config": app.config.to_dict(),
                        "weight_root": app.prover.weight_root,
                        "public_key": app.public_key.hex(),
                        "signature_algo": "ed25519",
                    },
                )
            else:
                self._send_json(404, {"error": "not_found", "path": self.path})

        def do_POST(self):  # noqa: N802
            try:
                req = self._read_json()
            except json.JSONDecodeError:
                return self._send_json(400, {"error": "invalid_json"})

            if self.path == "/prove":
                return self._handle_prove(req)
            if self.path == "/generate":
                return self._handle_generate(req)
            if self.path == "/verify":
                return self._handle_verify(req, full=False)
            if self.path == "/verify/full":
                return self._handle_verify(req, full=True)
            if self.path == "/verify/transcript":
                return self._handle_verify_transcript(req)
            return self._send_json(404, {"error": "not_found", "path": self.path})

        # ---- handlers ---------------------------------------------------
        def _handle_prove(self, req: dict[str, Any]) -> None:
            try:
                tokens = np.asarray(req["tokens"], dtype="<i8")
            except (KeyError, ValueError, TypeError) as e:
                return self._send_json(400, {"error": "bad_tokens", "detail": str(e)})
            include_full_trace = bool(req.get("include_full_trace", True))
            include_logits = bool(req.get("include_logits", False))
            _, cert = app.prover.run(
                tokens,
                include_full_trace=include_full_trace,
                include_logits=include_logits,
                sign_key=app.secret_key,
                sign_algo="ed25519",
                key_id=req.get("key_id", "server-default"),
            )
            self._send_json(200, cert.to_dict())

        def _handle_generate(self, req: dict[str, Any]) -> None:
            try:
                prompt = np.asarray(req["prompt"], dtype="<i8")
                max_new_tokens = int(req["max_new_tokens"])
            except (KeyError, ValueError, TypeError) as e:
                return self._send_json(400, {"error": "bad_request", "detail": str(e)})
            eos = req.get("eos_token")
            transcript = session_generate(
                app.prover,
                prompt,
                max_new_tokens=max_new_tokens,
                sign_key=app.secret_key,
                sign_algo="ed25519",
                eos_token=eos,
                include_full_trace=bool(req.get("include_full_trace", False)),
            )
            self._send_json(200, transcript.to_dict())

        def _handle_verify(self, req: dict[str, Any], *, full: bool) -> None:
            try:
                cert = Certificate.from_dict(req)
            except (KeyError, ValueError, TypeError) as e:
                return self._send_json(400, {"error": "bad_certificate", "detail": str(e)})
            try:
                if full:
                    report = Verifier(app.prover.model).verify(cert)
                else:
                    report = verify_certificate(cert)
                ok = report.ok
                code = report.code
            except VerificationError as e:
                ok, code, report_checks = False, e.code, {}
                self._send_json(200, {"ok": ok, "code": code, "checks": report_checks})
                return
            self._send_json(
                200,
                {"ok": ok, "code": code, "message": report.message, "checks": report.checks},
            )

        def _handle_verify_transcript(self, req: dict[str, Any]) -> None:
            try:
                t = SessionTranscript.from_dict(req)
            except (KeyError, ValueError, TypeError) as e:
                return self._send_json(400, {"error": "bad_transcript", "detail": str(e)})
            ok = verify_transcript(t)
            self._send_json(200, {"ok": ok})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, app: AppState | None = None) -> ThreadingHTTPServer:
    """Build and return an HTTP server. Caller is responsible for
    ``serve_forever()`` and ``shutdown()``."""
    if app is None:
        app = build_demo_app()
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    server.app = app  # type: ignore[attr-defined]
    return server


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    app = build_demo_app(seed=args.seed)
    server = serve(args.host, args.port, app)
    print(f"verifiable-inference server listening on {args.host}:{args.port}")
    print(f"  weight_root: {app.prover.weight_root}")
    print(f"  public_key:  {app.public_key.hex()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
