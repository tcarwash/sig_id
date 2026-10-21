"""Contracts shared by receive-only protocol decoder backends.

Backends expose protocol decoding as text or other typed receive products. The
contract deliberately says nothing about modulation identification or raw symbol
recovery; those are separate analysis paths.
"""

from __future__ import annotations

from typing import Protocol


class TextReceiveBackend(Protocol):
    """Minimal interface needed by the bounded live mode-trial runner."""

    backend_name: str

    def mode_names(self) -> list[str]: ...

    def select_mode(self, name: str) -> str: ...

    def receive_text(self, *, start: int = 0, length: int = 65536) -> bytes: ...

    def receive_text_length(self) -> int: ...
