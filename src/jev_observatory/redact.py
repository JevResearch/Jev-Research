"""Secret redaction applied to every persisted artifact.

The API key is read from the environment only.  It must never appear in a run
manifest, attempt ledger, raw body, exception message, analysis table or report.
Redaction is applied on write (defence in depth: a caller that accidentally
passes the key into a message still gets a safe artifact).
"""

from __future__ import annotations

import os
import re

API_KEY_ENV = "TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
LIVE_OPT_IN_ENV = "JEVO_ALLOW_LIVE"
DEFAULT_BASE_URL = "https://api.typesafe.ai"

_REDACTED = "[REDACTED]"

# Structural patterns, so that even a key we were not told about is scrubbed.
_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"']+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/\-]+=*"),
    re.compile(r"apikey_[0-9a-f_]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?)[^\s,;\"'}]+"),
]


class Redactor:
    """Scrub known secret literals plus recognised secret shapes."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        literals = [s for s in (secrets or []) if s and len(s) >= 6]
        # Longest first so a key is not partially replaced by a substring rule.
        self._literals = sorted(set(literals), key=len, reverse=True)

    @classmethod
    def from_environment(cls) -> "Redactor":
        key = os.environ.get(API_KEY_ENV)
        return cls(secrets=[key] if key else [])

    def text(self, value: str) -> str:
        if not isinstance(value, str):
            value = str(value)
        for secret in self._literals:
            value = value.replace(secret, _REDACTED)
        for pattern in _PATTERNS:
            value = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + _REDACTED, value)
        return value

    def obj(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {
                (k if not _is_secret_key(k) else _REDACTED): (
                    _REDACTED if _is_secret_key(k) else self.obj(v)
                )
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.obj(v) for v in value]
        return value

    def headers(self, headers: dict[str, str]) -> dict[str, str]:
        return {k: (_REDACTED if _is_secret_key(k) else self.text(str(v))) for k, v in headers.items()}

    def exception(self, exc: BaseException) -> dict:
        return {
            "type": type(exc).__name__,
            "message": self.text(str(exc)),
        }


def _is_secret_key(key: object) -> bool:
    """Structural secret-key detection. Deliberately excludes generic "token"
    substrings: legit contract fields like `input_tokens` must survive."""
    lowered = str(key).lower().replace("_", "").replace("-", "")
    return any(marker in lowered for marker in ("authorization", "apikey", "secret", "cookie", "credential"))


def get_api_key() -> str | None:
    """Return the process-environment key, or None. Never stored in artifacts."""
    return os.environ.get(API_KEY_ENV) or None


def base_url() -> str:
    return (os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


def live_calls_allowed() -> bool:
    """Paid live calls require an explicit opt-in (Milestone 4 gate)."""
    return os.environ.get(LIVE_OPT_IN_ENV) == "1"
