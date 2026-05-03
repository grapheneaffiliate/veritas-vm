"""End-to-end HTTP server + client tests."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from verifiable_inference import signatures as ed25519
from verifiable_inference.client import Client
from verifiable_inference.server import build_demo_app, serve
from verifiable_inference.session import verify_transcript
from verifiable_inference.verifier import Verifier


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    app = build_demo_app(seed=42)
    port = _free_port()
    srv = serve("127.0.0.1", port, app)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    # Wait for the listener to come up.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    try:
        yield (f"http://127.0.0.1:{port}", app)
    finally:
        srv.shutdown()
        thread.join(timeout=2.0)


def test_health(server):
    base_url, app = server
    c = Client(base_url)
    h = c.health()
    assert h["ok"] is True
    assert h["weight_root"] == app.prover.weight_root


def test_model_info_exposes_pubkey(server):
    base_url, app = server
    info = Client(base_url).model_info()
    assert info["weight_root"] == app.prover.weight_root
    assert info["public_key"] == app.public_key.hex()
    assert info["signature_algo"] == "ed25519"
    assert info["config"]["d_model"] > 0


def test_prove_returns_signed_certificate(server):
    base_url, app = server
    c = Client(base_url)
    cert = c.prove([3, 1, 4, 1, 5, 9, 2, 6])
    assert cert.signature["algo"] == "ed25519"
    assert bytes.fromhex(cert.signature["public_key"]) == app.public_key
    # And it independently verifies.
    Verifier(app.prover.model).verify(cert).raise_if_failed()


def test_prove_then_verify_roundtrip(server):
    base_url, _ = server
    c = Client(base_url)
    cert = c.prove([3, 1, 4, 1, 5, 9])
    result = c.verify(cert)
    assert result["ok"] is True


def test_full_verify_endpoint(server):
    base_url, _ = server
    c = Client(base_url)
    cert = c.prove([1, 2, 3])
    result = c.verify(cert, full=True)
    assert result["ok"] is True
    # Full re-derivation must include the run-time check.
    assert "rerun_merkle_root" in result["checks"]


def test_verify_rejects_tampered_cert(server):
    base_url, _ = server
    c = Client(base_url)
    cert = c.prove([1, 2, 3, 4])
    # Mutate the merkle_root client-side.
    d = cert.to_dict()
    d["trace"]["merkle_root"] = "0" * 64
    from verifiable_inference.certificate import Certificate

    bad = Certificate.from_dict(d)
    result = c.verify(bad)
    assert result["ok"] is False
    assert result["code"] in {"merkle_root_mismatch"}


def test_generate_and_verify_transcript(server):
    base_url, _ = server
    c = Client(base_url)
    transcript = c.generate([1, 2, 3], max_new_tokens=3)
    assert len(transcript.steps) == 3
    # Verify locally.
    assert verify_transcript(transcript)
    # And via server.
    result = c.verify_transcript(transcript)
    assert result["ok"] is True


def test_404_on_unknown_path(server):
    base_url, _ = server
    import urllib.error
    import urllib.request

    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(base_url + "/nope", timeout=2.0)
    assert ei.value.code == 404
