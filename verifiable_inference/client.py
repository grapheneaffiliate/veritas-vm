"""Stdlib-only HTTP client for the prove/verify API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from .certificate import Certificate
from .session import SessionTranscript


class ClientError(Exception):
    pass


class Client:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise ClientError(f"{path} -> HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")

    def _get(self, path: str) -> dict[str, Any]:
        url = self.base_url + path
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise ClientError(f"{path} -> HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")

    # --- High-level wrappers -------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def model_info(self) -> dict[str, Any]:
        return self._get("/model")

    def prove(
        self,
        tokens: list[int],
        *,
        include_full_trace: bool = True,
        include_logits: bool = False,
        key_id: Optional[str] = None,
    ) -> Certificate:
        body: dict[str, Any] = {
            "tokens": list(tokens),
            "include_full_trace": include_full_trace,
            "include_logits": include_logits,
        }
        if key_id is not None:
            body["key_id"] = key_id
        return Certificate.from_dict(self._post("/prove", body))

    def generate(
        self,
        prompt: list[int],
        *,
        max_new_tokens: int,
        eos_token: Optional[int] = None,
        include_full_trace: bool = False,
    ) -> SessionTranscript:
        body: dict[str, Any] = {
            "prompt": list(prompt),
            "max_new_tokens": int(max_new_tokens),
            "include_full_trace": include_full_trace,
        }
        if eos_token is not None:
            body["eos_token"] = int(eos_token)
        return SessionTranscript.from_dict(self._post("/generate", body))

    def verify(self, cert: Certificate, *, full: bool = False) -> dict[str, Any]:
        path = "/verify/full" if full else "/verify"
        return self._post(path, cert.to_dict())

    def verify_transcript(self, transcript: SessionTranscript) -> dict[str, Any]:
        return self._post("/verify/transcript", transcript.to_dict())
