"""Small, receive-only XML-RPC client for an isolated Fldigi instance."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any
from xmlrpc.client import ServerProxy, Transport


class _TimeoutTransport(Transport):
    def __init__(self, timeout: float) -> None:
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host: str) -> Any:
        connection = super().make_connection(host)
        connection.timeout = self.timeout
        return connection


@dataclass(frozen=True, slots=True)
class FldigiVersion:
    name: str
    version: str


class FldigiClient:
    """Client for read and receive-mode selection methods only.

    It intentionally exposes no transmit, tune, macro, or rig-control methods.
    Start Fldigi with an XML-RPC allow-list as an additional process boundary.
    """

    backend_name = "fldigi"

    def __init__(
        self,
        url: str = "http://127.0.0.1:7362/RPC2",
        timeout: float = 2.0,
        *,
        rpc_proxy: Any | None = None,
    ):
        self._rpc = rpc_proxy or ServerProxy(
            url,
            allow_none=True,
            use_builtin_types=True,
            transport=_TimeoutTransport(timeout),
        )
        self._lock = threading.RLock()

    def version(self) -> FldigiVersion:
        with self._lock:
            return FldigiVersion(self._rpc.fldigi.name(), self._rpc.fldigi.version())

    def modem_names(self) -> list[str]:
        with self._lock:
            return [str(name) for name in self._rpc.modem.get_names()]

    def mode_names(self) -> list[str]:
        """Return the receive-mode catalog through the generic backend contract."""
        return self.modem_names()

    def current_modem(self) -> str:
        with self._lock:
            return str(self._rpc.modem.get_name())

    def select_modem(self, name: str) -> str:
        """Select a catalog name and return the resulting current mode."""
        catalog = self.modem_names()
        if name not in catalog:
            raise ValueError(f"mode is not present in the Fldigi catalog: {name}")
        with self._lock:
            self._rpc.modem.set_by_name(name)
            return str(self._rpc.modem.get_name())

    def select_mode(self, name: str) -> str:
        """Select a receive mode through the generic backend contract."""
        return self.select_modem(name)

    def receive_text(self, *, start: int = 0, length: int = 65536) -> bytes:
        if start < 0 or length < 0:
            raise ValueError("start and length must be non-negative")
        with self._lock:
            value = self._rpc.text.get_rx(start, length)
            return value if isinstance(value, bytes) else str(value).encode("utf-8")

    def receive_text_length(self) -> int:
        with self._lock:
            return int(self._rpc.text.get_rx_length())

    def receive_wefax_file(self, *, timeout_seconds: int = 0) -> str:
        """Return the next completed WEFAX image path; zero timeout does not wait."""
        if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not 0 <= timeout_seconds <= 5:
            raise ValueError("WEFAX file wait must be an integer from 0 to 5 seconds")
        with self._lock:
            result = self._rpc.wefax.get_received_file(timeout_seconds)
            return str(result or "")


@dataclass(frozen=True, slots=True)
class ModeCapability:
    """Evidence for a mode. Catalog presence alone never means supported."""

    backend_name: str
    catalog_present: bool = False
    selectable: bool = False
    audio_routed: bool = False
    decode_validated: bool = False
    output_handled: bool = False
    generation_validated: bool = False
    automatic_id_validated: bool = False
    notes: str = "Catalog presence only; end-to-end validation pending."
