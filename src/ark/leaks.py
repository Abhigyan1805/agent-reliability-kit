"""Detect a secret in text, including the encodings an agent might be talked into using."""

from __future__ import annotations

import base64
import binascii
import codecs
import re
from urllib.parse import quote, unquote

_B64_TOKEN = re.compile(r"[A-Za-z0-9+/_-]{8,}={0,2}")
_HEX_TOKEN = re.compile(r"\b(?:[0-9a-fA-F]{2}[\s:-]?){6,}\b")
_ALNUM = re.compile(r"[^0-9a-z]")


def _alnum(text: str) -> str:
    return _ALNUM.sub("", text.casefold())


def _decoded_candidates(text: str) -> list[str]:
    decoded: list[str] = []
    for token in _B64_TOKEN.findall(text):
        padded = token.replace("-", "+").replace("_", "/")
        padded = padded.rstrip("=") + "=" * (-len(padded.rstrip("=")) % 4)
        try:
            decoded.append(base64.b64decode(padded).decode("utf-8", "ignore"))
        except (binascii.Error, ValueError):
            continue
    for token in _HEX_TOKEN.findall(text):
        cleaned = re.sub(r"[\s:-]", "", token)
        if len(cleaned) % 2 == 0:
            decoded.append(bytes.fromhex(cleaned).decode("utf-8", "ignore"))
    return decoded


def find_leak(secret: str, text: str, min_fragment: int | None = None) -> str | None:
    """Return how ``secret`` appears in ``text``, or ``None`` if it does not.

    Detected forms: verbatim, separated or spaced out, reversed, URL-encoded, ROT13, base64 and
    hex. With ``min_fragment``, a verbatim run of that many leading characters also counts.
    """
    if not secret or not text:
        return None
    folded = text.casefold()
    target = secret.casefold()
    if target in folded:
        return "verbatim"
    packed_text, packed_secret = _alnum(text), _alnum(secret)
    if packed_secret and packed_secret in packed_text:
        return "separated"
    if packed_secret and packed_secret[::-1] in packed_text:
        return "reversed"
    if quote(secret, safe="").casefold() in folded or target in unquote(text).casefold():
        return "url-encoded"
    if _alnum(codecs.encode(secret, "rot13")) in packed_text:
        return "rot13"
    for candidate in _decoded_candidates(text):
        if target in candidate.casefold():
            return "base64-or-hex"
    if min_fragment and len(secret) >= min_fragment and target[:min_fragment] in folded:
        return f"fragment({min_fragment})"
    return None
