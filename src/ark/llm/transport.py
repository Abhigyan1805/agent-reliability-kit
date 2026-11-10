"""An httpx transport that adds a response cache, pacing and backoff to any OpenAI-style client.

Doing this below the SDK gives the plain loop and LangChain identical behaviour for free.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

_RETRYABLE = {429, 500, 502, 503, 504}
_RETRY_DELAY = re.compile(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"')
_DAILY = re.compile(r"per.?day|limit:\s*0\b", re.IGNORECASE)


class InfraError(RuntimeError):
    """The model provider failed. This says nothing about the agent, so it is not scored."""


class QuotaExhausted(InfraError):
    """A long-window quota ran out. Retrying now is pointless, so the run should stop."""


class CachingTransport(httpx.BaseTransport):
    def __init__(
        self,
        cache_dir: str | Path | None,
        min_interval_s: float = 0.0,
        max_retries: int = 8,
        inner: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.inner = inner or httpx.HTTPTransport()
        self.salt = ""
        self.live_requests = 0
        self.cache_hits = 0
        self._sleep = sleep
        self._last_sent = 0.0
        self._lock = threading.Lock()
        self._provider_extras: dict[str, Any] = {}

    def _restore_extras(self, body: dict[str, Any]) -> None:
        """Re-attach provider fields (Gemini's thought signatures) that SDKs drop from history."""
        for message in body.get("messages", []):
            for call in message.get("tool_calls") or []:
                extra = self._provider_extras.get(call.get("id", ""))
                if extra and "extra_content" not in call:
                    call["extra_content"] = extra

    def _remember_extras(self, payload: bytes) -> None:
        try:
            choices = json.loads(payload).get("choices", [])
        except (json.JSONDecodeError, AttributeError):
            return
        for choice in choices:
            for call in (choice.get("message") or {}).get("tool_calls") or []:
                if call.get("extra_content"):
                    self._provider_extras[call["id"]] = call["extra_content"]

    def _cache_path(self, url: str, body: bytes) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(b"\0".join([url.encode(), body, self.salt.encode()])).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _pace(self) -> None:
        with self._lock:
            wait = self._last_sent + self.min_interval_s - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last_sent = time.monotonic()

    def _backoff(self, attempt: int, text: str = "", retry_after: str | None = None) -> float:
        match = _RETRY_DELAY.search(text)
        if match:
            return float(match.group(1)) + 1.0
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            return float(retry_after) + 1.0
        return min(90.0, 2.0 * 2**attempt) + random.uniform(0.0, 1.0)

    def _send_live(self, request: httpx.Request) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                response = self.inner.handle_request(request)
            except httpx.TransportError as exc:
                last_error = exc
                self._sleep(self._backoff(attempt))
                continue
            content = response.read()
            self.live_requests += 1
            if response.status_code not in _RETRYABLE or attempt == self.max_retries:
                if response.status_code == 429 and _DAILY.search(content.decode("utf-8", "ignore")):
                    raise QuotaExhausted(f"provider quota exhausted: {_excerpt(content)}")
                return httpx.Response(
                    response.status_code,
                    headers={"content-type": response.headers.get("content-type", "")},
                    content=content,
                )
            text = content.decode("utf-8", "ignore")
            if response.status_code == 429 and _DAILY.search(text):
                raise QuotaExhausted(f"provider quota exhausted: {_excerpt(content)}")
            self._sleep(self._backoff(attempt, text, response.headers.get("retry-after")))
        raise InfraError(f"network failure after retries: {last_error}")

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raw = request.read()
        body_bytes = raw
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            self._restore_extras(body)
            body_bytes = json.dumps(body, sort_keys=True).encode()
            request = httpx.Request(
                request.method,
                request.url,
                headers=[
                    (k, v) for k, v in request.headers.items() if k.lower() != "content-length"
                ],
                content=body_bytes,
                extensions=request.extensions,
            )
        path = self._cache_path(str(request.url), body_bytes)
        if path is not None and path.exists():
            self.cache_hits += 1
            payload = path.read_bytes()
            self._remember_extras(payload)
            return httpx.Response(
                200, headers={"content-type": "application/json"}, content=payload
            )

        response = self._send_live(request)
        if response.status_code == 200:
            self._remember_extras(response.content)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(path)
        return response

    def close(self) -> None:
        self.inner.close()


def _excerpt(content: bytes) -> str:
    return " ".join(content.decode("utf-8", "ignore").split())[:400]
