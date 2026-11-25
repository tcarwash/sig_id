"""Receive-only browser audio bridge for WebSDR, KiwiSDR and OpenWebRX pages.

Audio is isolated in a per-session PipeWire/PulseAudio null sink. Only the
dedicated browser process session is routed to it; unrelated desktop audio is
never redirected. The remapped monitor appears as an ordinary PortAudio input.
"""
from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4


DEFAULT_RECEIVERS = [
    {
        "id": "northern-utah",
        "name": "Northern Utah WebSDR",
        "url": "https://www.sdrutah.org/",
        "kind": "WebSDR",
        "description": "Known WebSDR starting point; tune in its browser window.",
    },
    {
        "id": "twente",
        "name": "University of Twente WebSDR",
        "url": "http://websdr.ewi.utwente.nl:8901/",
        "kind": "WebSDR",
        "description": "Classic multi-user HF WebSDR.",
    },
    {
        "id": "websdr-directory",
        "name": "WebSDR.org directory",
        "url": "https://www.websdr.org/",
        "kind": "Directory",
        "description": "Choose a receiver; audio from its browser tab is captured.",
    },
    {
        "id": "kiwisdr-map",
        "name": "KiwiSDR receiver map",
        "url": "https://rx.linkfanel.net/",
        "kind": "Directory",
        "description": "Choose a KiwiSDR; audio from its browser tab is captured.",
    },
]


def validate_receiver_url(value: str, *, allow_private: bool = False) -> str:
    """Accept browser receiver URLs, not arbitrary local-file/browser schemes."""
    url = str(value or "").strip()
    if not url or len(url) > 2048:
        raise ValueError("receiver URL must contain 1–2048 characters")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port  # force malformed ports to fail validation
    except ValueError as exc:
        raise ValueError("receiver URL has an invalid host or port") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ValueError("receiver URL must be an http:// or https:// web address")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("receiver URLs must not contain embedded credentials")
    host = hostname.rstrip(".").casefold()
    if not allow_private:
        if host == "localhost" or host.endswith((".localhost", ".local")):
            raise ValueError("local receiver URLs are disabled; set SIG_ID_WEBSDR_ALLOW_PRIVATE=1 to allow LAN receivers")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("private or local receiver addresses are disabled; set SIG_ID_WEBSDR_ALLOW_PRIVATE=1 to allow them")
    return url


def _desktop_environment() -> dict[str, str]:
    """Recover the logged-in user's audio/Wayland session when launched by a service."""
    env = {key: value for key, value in os.environ.items()}
    runtime = Path(env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    if runtime.is_dir():
        env.setdefault("XDG_RUNTIME_DIR", str(runtime))
    pulse_socket = runtime / "pulse" / "native"
    if "PULSE_SERVER" not in env and pulse_socket.exists():
        env["PULSE_SERVER"] = f"unix:{pulse_socket}"

    if not env.get("WAYLAND_DISPLAY") and not env.get("DISPLAY"):
        wayland = sorted(path for path in runtime.glob("wayland-*") if path.is_socket())
        if wayland:
            env["WAYLAND_DISPLAY"] = wayland[0].name
        else:
            # Xwayland is common on desktop Linux. Its command line carries
            # both the display number and the session-specific auth file.
            for proc in Path("/proc").glob("[0-9]*"):
                try:
                    if proc.stat().st_uid != os.getuid():
                        continue
                    args = (proc / "cmdline").read_bytes().decode(errors="ignore").split("\0")
                    if not args or "Xwayland" not in Path(args[0]).name:
                        continue
                    display = next((item for item in args[1:] if re.fullmatch(r":\d+(?:\.\d+)?", item)), None)
                    if display:
                        env["DISPLAY"] = display
                        if "-auth" in args:
                            index = args.index("-auth")
                            if index + 1 < len(args):
                                env["XAUTHORITY"] = args[index + 1]
                        break
                except (OSError, ValueError):
                    continue
    return env


class WebSDRLoopback:
    """Create a virtual capture input and launch one isolated receiver browser."""

    def __init__(self, *, profile_dir: str | Path, browser: str | None = None):
        self.env = _desktop_environment()
        self.pactl = shutil.which("pactl")
        if not self.pactl:
            raise RuntimeError("Automatic WebSDR audio needs pactl and a running PipeWire/PulseAudio server.")
        candidates = (browser,) if browser else (
            self.env.get("SIG_ID_WEBSDR_BROWSER"),
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "brave-browser",
        )
        self.browser = next(
            (resolved for name in candidates if name and (resolved := shutil.which(name))),
            None,
        )
        if not self.browser:
            raise RuntimeError("Chrome/Chromium was not found; install it or set SIG_ID_WEBSDR_BROWSER.")
        if not self.env.get("WAYLAND_DISPLAY") and not self.env.get("DISPLAY"):
            raise RuntimeError("No desktop display is available to open the receiver. Run Signal ID in the logged-in desktop session.")
        self.profile_dir = Path(profile_dir).expanduser().resolve()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        suffix = uuid4().hex[:10]
        self.sink = f"sigid_websdr_{suffix}"
        self.source = f"{self.sink}_rx"
        self.device_name = self.source
        self.device_label = f"Signal-ID-WebSDR-RX-{suffix}"
        self.modules: list[str] = []
        self.process: subprocess.Popen | None = None
        self.browser_log = None
        self.stop_event = threading.Event()
        self.watcher: threading.Thread | None = None
        self.routing_errors = 0
        self._closed = False
        try:
            self.modules.append(self._pulse(
                "load-module", "module-null-sink",
                f"sink_name={self.sink}", "rate=48000", "channels=2",
                f"sink_properties=device.description={self.sink}",
            ).strip())
            self.modules.append(self._pulse(
                "load-module", "module-remap-source",
                f"master={self.sink}.monitor", f"source_name={self.source}",
                "rate=48000", "channels=1", "channel_map=mono",
                f"source_properties=device.description={self.device_label}",
            ).strip())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                sources = json.loads(self._pulse("--format=json", "list", "sources"))
                if any(item.get("name") == self.source for item in sources):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("The virtual WebSDR capture input did not become available.")
        except Exception:
            self.close()
            raise

    def _pulse(self, *args: str) -> str:
        try:
            result = subprocess.run(
                [self.pactl, *args], env=self.env, capture_output=True,
                text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("PipeWire/PulseAudio command failed; check the desktop audio session.") from exc
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"PipeWire/PulseAudio operation failed: {detail[-300:]}")
        return result.stdout

    def launch(self, url: str) -> None:
        if self.process is not None:
            raise RuntimeError("The WebSDR browser is already running.")
        self.browser_log = (self.profile_dir / "browser.log").open("w", encoding="utf-8")
        env = dict(self.env)
        env["PULSE_SINK"] = self.sink
        env["PULSE_PROP"] = f"application.id={self.sink}"
        command = [
            self.browser,
            f"--user-data-dir={self.profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--new-window",
            "--autoplay-policy=no-user-gesture-required",
            url,
        ]
        try:
            self.process = subprocess.Popen(
                command, env=env, stdout=self.browser_log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            time.sleep(0.2)
            if self.process.poll() is not None:
                raise RuntimeError("The WebSDR browser exited during startup; check its browser.log.")
            self.watcher = threading.Thread(
                target=self._route, name="sig-id-websdr-routing", daemon=True,
            )
            self.watcher.start()
        except Exception:
            self.close()
            raise

    def _route(self) -> None:
        while not self.stop_event.wait(0.25):
            if self.process is None or self.process.poll() is not None:
                return
            try:
                self.route_once()
            except (RuntimeError, ValueError, OSError, StopIteration, subprocess.TimeoutExpired):
                self.routing_errors += 1

    def route_once(self) -> None:
        """Route only streams belonging to this isolated browser process session."""
        sinks = json.loads(self._pulse("--format=json", "list", "sinks"))
        target = next(item["index"] for item in sinks if item.get("name") == self.sink)
        streams = json.loads(self._pulse("--format=json", "list", "sink-inputs"))
        for stream in streams:
            properties = stream.get("properties", {})
            owned = properties.get("application.id") == self.sink
            if not owned and self.process is not None:
                try:
                    pid = int(properties.get("application.process.id", "-1"))
                    owned = os.getsid(pid) == self.process.pid
                except (OSError, ValueError, ProcessLookupError):
                    owned = False
            if owned and stream.get("sink") != target:
                self._pulse("move-sink-input", str(stream["index"]), self.sink)

    def status(self) -> dict[str, Any]:
        running = bool(self.process is not None and self.process.poll() is None)
        return {
            "browser_running": running,
            "state": "connected" if running else "browser_closed",
            "device_name": self.device_name,
            "device_label": self.device_label,
            "audio_source_name": self.source,
            "routing_errors": self.routing_errors,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        if self.watcher is not None:
            self.watcher.join(timeout=6)
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.browser_log is not None:
            self.browser_log.close()
            self.browser_log = None
        for module in reversed(self.modules):
            try:
                self._pulse("unload-module", module)
            except (RuntimeError, subprocess.TimeoutExpired):
                self.routing_errors += 1
        self.modules.clear()


class WebSDRSessionManager:
    """Own at most one isolated WebSDR browser/audio route per Signal ID process."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        loopback_factory: Any = WebSDRLoopback,
        browser: str | None = None,
        allow_private: bool | None = None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.profile_root = self.data_dir / "profiles"
        self.loopback_factory = loopback_factory
        self.browser = browser
        self.allow_private = (
            os.environ.get("SIG_ID_WEBSDR_ALLOW_PRIVATE", "").strip() == "1"
            if allow_private is None else bool(allow_private)
        )
        self._lock = threading.RLock()
        self._loopback: WebSDRLoopback | Any | None = None
        self._url: str | None = None
        self._name: str | None = None
        self._profile: Path | None = None

    def connect(self, url: str, name: str = "WebSDR receiver") -> dict[str, Any]:
        normalized_url = validate_receiver_url(url, allow_private=self.allow_private)
        label = " ".join(str(name or "WebSDR receiver").split())[:100] or "WebSDR receiver"
        with self._lock:
            if self._loopback is not None:
                status = self._loopback.status()
                if status.get("browser_running"):
                    raise RuntimeError("A WebSDR browser is already connected; disconnect it before switching receivers.")
                self._close_locked()
            session_id = uuid4().hex[:12]
            profile = self.profile_root / session_id
            loopback = self.loopback_factory(profile_dir=profile, browser=self.browser)
            try:
                loopback.launch(normalized_url)
            except Exception:
                loopback.close()
                raise
            self._loopback = loopback
            self._url = normalized_url
            self._name = label
            self._profile = profile
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._loopback is None:
                return {"active": False, "state": "disconnected"}
            result = self._loopback.status()
            return {
                "active": True,
                "state": result.get("state", "connected"),
                "browser_running": bool(result.get("browser_running")),
                "name": self._name,
                "url": self._url,
                "device_name": result.get("device_name"),
                "device_label": result.get("device_label"),
                "audio_source_name": result.get("audio_source_name"),
                "routing_errors": int(result.get("routing_errors", 0)),
            }

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            self._close_locked()
            return {"active": False, "state": "disconnected"}

    def _close_locked(self) -> None:
        if self._loopback is not None:
            self._loopback.close()
        self._loopback = None
        self._url = None
        self._name = None
        self._profile = None

    def close(self) -> None:
        self.disconnect()
