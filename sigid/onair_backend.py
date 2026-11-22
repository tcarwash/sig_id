"""Receive-only bridge for OnAir's documented loopback HTTP event feed.

This adapter consumes GET /api/v1/events, not OnAir's receiver-control or
transmitter endpoints. It never opens a radio/audio device, tunes, transmits,
or imports OnAir's GPL-licensed implementation.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import re
from threading import Event, Lock, Thread
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import (
    HTTPRedirectHandler, ProxyHandler, Request, build_opener,
)

from .decoder_backend import DecoderBackendInfo, DecoderEvent, DecoderProduct
from .modes import MODE_REGISTRY


LOGGER = logging.getLogger(__name__)
MAX_RESPONSE_BYTES = 256 * 1024
MAX_EVENT_BYTES = 16 * 1024
MAX_BATCH = 100
MAX_PENDING = 1000
MAX_DEDUPE = 2048
MAX_MODE_LENGTH = 80
MAX_EVENT_TYPE_LENGTH = 120


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # No redirect may change the loopback-only trust boundary.
        return None


def _onair_origin(base_url: str) -> str:
    """Accept exactly a local HTTP origin, never a remote server or credentials."""
    parsed = urlsplit(base_url.strip())
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("OnAir API must use http://127.0.0.1, localhost, or [::1]")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OnAir credentials belong in SIG_ID_ONAIR_TOKEN, not the URL")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("OnAir URL must be an origin without a path, query or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("OnAir API port must be valid") from exc
    if port is None or not (1 <= port <= 65535):
        raise ValueError("OnAir API URL must include a TCP port")
    return f"http://{parsed.netloc}"


def _event_mode_id(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
    # Reuse a canonical entry only when its stable ID exactly matches the
    # reported mode; do not infer protocol from a loose alias or substring.
    if key in MODE_REGISTRY:
        return key
    return f"onair-{key or 'unknown'}"


def _normalise_event(message: Any) -> DecoderEvent | None:
    if not isinstance(message, dict) or message.get("v") != 1:
        return None
    mode = message.get("mode")
    event_type = message.get("type")
    stamp = message.get("time")
    if not isinstance(mode, str) or not mode.strip() or len(mode) > MAX_MODE_LENGTH:
        return None
    if not isinstance(event_type, str) or not event_type.strip() or len(event_type) > MAX_EVENT_TYPE_LENGTH:
        return None
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        return None
    try:
        observed = datetime.fromtimestamp(stamp, timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None
    if not (2000 <= observed.year <= 2100):
        return None
    fields = message.get("fields")
    if not isinstance(fields, dict):
        fields = {}
    # Bound the memory footprint and avoid disclosing malformed/untrusted
    # provider content in backend status or oversized API responses.
    try:
        if len(fields) > 64 or len(json.dumps(fields, ensure_ascii=True)) > MAX_EVENT_BYTES:
            fields = {"truncated_by_adapter": True}
    except (TypeError, ValueError):
        fields = {"invalid_fields": True}

    freq = message.get("freqHz")
    frequency_hz = None
    if not isinstance(freq, bool) and isinstance(freq, (int, float)) and math.isfinite(freq) and 0 < freq <= 1e12:
        frequency_hz = float(freq)

    snr = fields.get("snr_db", fields.get("snr"))
    snr_db = None
    if not isinstance(snr, bool) and isinstance(snr, (int, float)) and math.isfinite(snr) and -200 <= snr <= 200:
        snr_db = float(snr)

    display_text = next(
        (value.strip() for value in
         (fields.get("text"), fields.get("message"), message.get("message"))
         if isinstance(value, str) and value.strip()),
        None,
    )
    products: list[DecoderProduct] = []
    if display_text:
        products.append(DecoderProduct("text", display_text[:8192], metadata={"origin": "OnAir"}))
    if fields and not display_text:
        products.append(DecoderProduct(
            "metadata", json.dumps(fields, ensure_ascii=True, sort_keys=True)[:MAX_EVENT_BYTES],
            media_type="application/json",
        ))

    return DecoderEvent(
        backend_id="onair",
        mode_id=_event_mode_id(mode.strip()),
        native_mode=mode.strip(),
        observed_utc=observed.isoformat(),
        products=tuple(products),
        frequency_hz=frequency_hz,
        snr_db=snr_db,
        metadata={
            "source": "onair-http-events-v1",
            "event_type": event_type,
            "event_version": 1,
            "frequency_reference": "rf",
            "truncated": bool(message.get("truncated", False)),
            "fields": fields,
        },
    )


class OnAirEventBackend:
    """Independent, opt-in, read-only poller for a separately running OnAir.

    Initial /events polling omits the cursor (OnAir starts at 'now'). Each
    subsequent request supplies the returned next sequence as 'since'.
    """

    backend_id = "onair"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        interval_seconds: float = 2.0,
        timeout_seconds: float = 1.5,
        start_worker: bool = True,
    ) -> None:
        if interval_seconds < 0.5:
            raise ValueError("OnAir polling interval must be at least 0.5 seconds")
        if not 0.2 <= timeout_seconds <= 5:
            raise ValueError("OnAir HTTP timeout must be between 0.2 and 5 seconds")
        self.origin = _onair_origin(base_url) if base_url else None
        self._token = token
        self._interval = float(interval_seconds)
        self._timeout = float(timeout_seconds)
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())
        self._lock = Lock()
        self._poll_lock = Lock()
        self._stop = Event()
        self._events: deque[DecoderEvent] = deque(maxlen=MAX_PENDING)
        self._recent: deque[str] = deque()
        self._recent_set: set[str] = set()
        self._cursor: int | None = None
        self._connected = False
        self._last_error: str | None = None
        self._gaps = 0
        self._dropped = 0
        self._worker: Thread | None = None
        if self.origin is not None and start_worker:
            self._worker = Thread(target=self._run, name="sig-id-onair-events", daemon=True)
            self._worker.start()

    def _request_events(self, cursor: int | None) -> dict[str, Any]:
        if self.origin is None:
            raise RuntimeError("OnAir API not configured")
        arguments = {"max": MAX_BATCH}
        if cursor is not None:
            arguments["since"] = cursor
        url = self.origin + "/api/v1/events?" + urlencode(arguments)
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = Request(url, headers=headers, method="GET")
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"OnAir HTTP status {response.status}")
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            raise RuntimeError(f"OnAir HTTP {exc.code}; check API enablement and token") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("OnAir API unavailable on configured loopback port") from exc
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("OnAir event response exceeded size limit")
        try:
            parsed = json.loads(body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("OnAir event response is not JSON") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
            raise ValueError("OnAir events response must contain an events array")
        if len(parsed["events"]) > MAX_BATCH:
            raise ValueError("OnAir returned more events than requested")
        next_cursor = parsed.get("next")
        if isinstance(next_cursor, bool) or not isinstance(next_cursor, int) or next_cursor < 0:
            raise ValueError("OnAir response has invalid next cursor")
        if not isinstance(parsed.get("gap", False), bool):
            raise ValueError("OnAir response has invalid gap indicator")
        return parsed

    def poll_once(self) -> int:
        """Fetch one bounded batch; useful for deterministic integration tests."""
        if self.origin is None:
            return 0
        with self._poll_lock:
            with self._lock:
                cursor = self._cursor
            try:
                result = self._request_events(cursor)
                accepted = 0
                with self._lock:
                    if result.get("gap", False):
                        self._gaps += 1
                    for raw in result["events"]:
                        try:
                            digest = hashlib.sha256(
                                json.dumps(raw, sort_keys=True, ensure_ascii=True).encode()
                            ).hexdigest()
                        except (TypeError, ValueError):
                            continue
                        if digest in self._recent_set:
                            continue
                        event = _normalise_event(raw)
                        if event is None:
                            continue
                        if len(self._recent) >= MAX_DEDUPE:
                            self._recent_set.discard(self._recent.popleft())
                        self._recent.append(digest)
                        self._recent_set.add(digest)
                        if len(self._events) >= MAX_PENDING:
                            self._dropped += 1
                        self._events.append(event)
                        accepted += 1
                    # Handles an OnAir application restart where sequence
                    # numbering resets. The prior fingerprint cache still
                    # deduplicates repeated receive events.
                    self._cursor = result["next"]
                    self._connected = True
                    self._last_error = None
                return accepted
            except (RuntimeError, ValueError) as exc:
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
                return 0

    def _run(self) -> None:
        self.poll_once()
        while not self._stop.wait(self._interval):
            self.poll_once()

    def info(self) -> DecoderBackendInfo:
        with self._lock:
            connected = self._connected
            error = self._last_error
            gaps, dropped = self._gaps, self._dropped
        if self.origin is None:
            note = "Set SIG_ID_ONAIR_URL to a local OnAir API origin and SIG_ID_ONAIR_TOKEN if required; event feed only."
        elif error:
            note = f"{error}; event feed only, no OnAir audio replay or radio control."
        elif connected:
            note = (
                f"OnAir event feed connected; {gaps} source gaps, {dropped} local queue drops. "
                "Live decoded events only; no audio-window replay, tuning or transmit."
            )
        else:
            note = "Waiting for OnAir event feed; enable OnAir's remote API locally. No audio-window replay."
        return DecoderBackendInfo(
            backend_id="onair",
            application="OnAir",
            connected=connected,
            input_kinds=("native-onair-live-events",),
            product_kinds=("text", "metadata"),
            modes=(),  # Event-only; OnAir's whole catalog is not a decode-selection menu.
            note=note,
        )

    def drain_events(self, *, limit: int = 100) -> list[DecoderEvent]:
        if not 1 <= limit <= 1000:
            raise ValueError("event limit must be between 1 and 1000")
        with self._lock:
            return [self._events.popleft() for _ in range(min(limit, len(self._events)))]

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=self._timeout + 0.5)
