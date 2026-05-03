"""Model registry — the directory that maps a 32-byte ``weight_root`` to
a publishable metadata document.

A registry entry is the public attestation a model owner puts on the
internet (e.g. a static JSON file at a well-known URL, a transparency
log, an immutable git tag). It pins:

  - the exact weight_root the world can verify against
  - the architecture config
  - the Ed25519 public key the prover signs certificates with
  - human-readable metadata: name, version, license, training-data
    description, model-card URL
  - an *attestation signature* by the model owner over the entry itself,
    so the entry's authenticity is independently verifiable

Verifiers run the following protocol:

  1. Receive a certificate.
  2. Look up cert.weight_root in the registry. (Out of scope here:
     in production this is a transparency log, CT-style.)
  3. Confirm the registry entry is signed by a trusted publisher.
  4. Confirm cert.signature.public_key matches the entry's public_key.
  5. Run normal certificate verification.

Step 4 is the missing piece without a registry: a malicious prover could
sign a forged certificate with their own key, and absent a registry the
verifier cannot tell the difference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from . import signatures as ed25519
from .canonical import canonical_json_bytes


@dataclass
class RegistryEntry:
    """One published row in the registry."""

    weight_root: str  # 64-char hex sha256
    config: dict
    public_key: str  # 64-char hex ed25519 pubkey
    name: str
    version: str
    license: str = "unspecified"
    training_data: str = "unspecified"
    model_card_url: Optional[str] = None
    description: str = ""
    attestation: dict = field(default_factory=lambda: {"algo": "none", "value": None})

    def attestation_payload(self) -> bytes:
        """Bytes covered by the publisher's attestation signature."""
        d = self.to_dict()
        d.pop("attestation", None)
        return canonical_json_bytes(d)

    def attest_ed25519(self, publisher_secret_key: bytes) -> bytes:
        """Sign this entry with the publisher's Ed25519 key. Returns the
        publisher's public key."""
        publisher_pk = ed25519.derive_public_key(publisher_secret_key)
        sig = ed25519.sign(self.attestation_payload(), publisher_secret_key)
        self.attestation = {
            "algo": "ed25519",
            "publisher_public_key": publisher_pk.hex(),
            "value": sig.hex(),
        }
        return publisher_pk

    def verify_attestation(self, publisher_pk: bytes | None = None) -> bool:
        algo = self.attestation.get("algo")
        if algo == "none":
            return True
        if algo != "ed25519":
            raise ValueError(f"unsupported attestation algo: {algo!r}")
        stored_pk = bytes.fromhex(self.attestation["publisher_public_key"])
        if publisher_pk is not None and publisher_pk != stored_pk:
            return False
        sig = bytes.fromhex(self.attestation["value"])
        return ed25519.verify(self.attestation_payload(), sig, stored_pk)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema": "vai-registry-entry/1.0",
            "weight_root": self.weight_root,
            "config": self.config,
            "public_key": self.public_key,
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "training_data": self.training_data,
            "description": self.description,
            "attestation": self.attestation,
        }
        if self.model_card_url is not None:
            d["model_card_url"] = self.model_card_url
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RegistryEntry":
        return RegistryEntry(
            weight_root=str(d["weight_root"]),
            config=dict(d["config"]),
            public_key=str(d["public_key"]),
            name=str(d.get("name", "")),
            version=str(d.get("version", "")),
            license=str(d.get("license", "unspecified")),
            training_data=str(d.get("training_data", "unspecified")),
            model_card_url=d.get("model_card_url"),
            description=str(d.get("description", "")),
            attestation=dict(d.get("attestation", {"algo": "none", "value": None})),
        )


@dataclass
class Registry:
    """Tiny in-memory registry. Production deployments use a transparency
    log (CT-style append-only Merkle log) or a static JSON file at a
    well-known URL. The Python object here is API-equivalent."""

    entries: dict[str, RegistryEntry] = field(default_factory=dict)

    def register(self, entry: RegistryEntry) -> None:
        if entry.weight_root in self.entries:
            raise ValueError(f"weight_root {entry.weight_root} already registered")
        self.entries[entry.weight_root] = entry

    def lookup(self, weight_root: str) -> Optional[RegistryEntry]:
        return self.entries.get(weight_root)

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": "vai-registry/1.0",
                "entries": [e.to_dict() for e in self.entries.values()],
            },
            indent=2,
            sort_keys=True,
        )

    @staticmethod
    def from_json(s: str) -> "Registry":
        d = json.loads(s)
        reg = Registry()
        for e in d["entries"]:
            reg.entries[e["weight_root"]] = RegistryEntry.from_dict(e)
        return reg


# --- Verification protocol ---------------------------------------------------


def verify_certificate_against_registry(
    cert,  # Certificate
    registry: Registry,
    *,
    trusted_publisher_pks: Optional[list[bytes]] = None,
) -> tuple[bool, str]:
    """The full real-world flow:

    1. lookup cert.weight_root in registry
    2. verify the entry's attestation — and (optionally) require it be
       signed by one of ``trusted_publisher_pks``
    3. confirm the cert's signature key matches the registry's pinned key
    4. confirm the cert's signature is valid

    Returns ``(ok, reason)``. ``reason`` is a stable code on failure.
    """
    entry = registry.lookup(cert.weight_root)
    if entry is None:
        return False, "weight_root_not_in_registry"

    if not entry.verify_attestation():
        return False, "registry_attestation_invalid"

    if trusted_publisher_pks is not None:
        algo = entry.attestation.get("algo")
        if algo != "ed25519":
            return False, "registry_unsigned_but_publishers_required"
        pk = bytes.fromhex(entry.attestation["publisher_public_key"])
        if pk not in trusted_publisher_pks:
            return False, "registry_publisher_untrusted"

    if cert.signature.get("algo") != "ed25519":
        return False, "cert_not_ed25519_signed"

    if cert.signature["public_key"] != entry.public_key:
        return False, "cert_pubkey_does_not_match_registry"

    if not cert.verify_signature():
        return False, "cert_signature_invalid"

    return True, "ok"
