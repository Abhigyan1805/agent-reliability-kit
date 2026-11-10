"""Text normalisation and the small matching language shared by tool stubs, faults and checks."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_SPACES = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_text(text: str) -> str:
    """Fold the formatting differences a check should not care about.

    Handles Unicode compatibility forms, zero-width characters, curly quotes, thousands
    separators, letter case and runs of whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text).translate(_QUOTES)
    text = _THOUSANDS.sub("", text)
    return _SPACES.sub(" ", text).casefold().strip()


def render(value: Any) -> str:
    """Render a JSON-like value the way an agent would see it from a tool."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def match_value(spec: Any, value: Any) -> bool:
    """Match one argument value against a scalar or operator spec.

    A scalar compares equal after normalisation. A mapping may use the operators ``equals``,
    ``contains``, ``not_contains``, ``regex`` and ``absent``; every operator given must hold.
    """
    if not isinstance(spec, dict):
        return value is not None and normalize_text(render(value)) == normalize_text(render(spec))
    if "absent" in spec and (value is None) != bool(spec["absent"]):
        return False
    if value is None:
        return "absent" in spec and bool(spec["absent"])
    text = normalize_text(render(value))
    if "equals" in spec and text != normalize_text(render(spec["equals"])):
        return False
    if "contains" in spec and normalize_text(str(spec["contains"])) not in text:
        return False
    if "not_contains" in spec and normalize_text(str(spec["not_contains"])) in text:
        return False
    if "regex" in spec and not re.search(spec["regex"], render(value), re.IGNORECASE):
        return False
    return True


def match_where(where: dict[str, Any] | None, args: dict[str, Any]) -> bool:
    """Match a ``where`` clause against tool arguments. The key ``"*"`` searches all arguments."""
    if not where:
        return True
    for key, spec in where.items():
        value = args if key == "*" else args.get(key)
        if not match_value(spec, value):
            return False
    return True
