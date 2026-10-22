"""RF-free, end-to-end Fldigi audio loopback tests for PSK, RTTY and THOR.

Run explicitly with SIGID_TEST_FLDIGI_AUDIO=1. It needs fldigi, xvfb-run,
pactl, a configured ~/.fldigi profile, and a working PulseAudio/PipeWire server.
The temporary null sink and copied profiles prevent audio or PTT from reaching a radio.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import shutil
import signal
import subprocess
import tempfile
import time
from xmlrpc.client import ServerProxy

import httpx
import pytest

from sigid.api import create_app
from sigid.fldigi import FldigiClient
from sigid.fldigi_replay import IsolatedFldigiReplay


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=check, timeout=5)


def _isolated_profile(root: Path, role: str, sink: str, port: int) -> subprocess.Popen[bytes]:
    source = Path.home() / ".fldigi"
    if not (source / "fldigi_def.xml").is_file():
        raise RuntimeError("a configured ~/.fldigi/fldigi_def.xml profile is required")
    config = root / role / "config"
    home = root / role / "home"
    shutil.copytree(source, config)
    home.mkdir()

    # Fldigi's preferences file is XML-like but contains unescaped text, so patch
    # only known scalar tags. Never edit the operator's profile.
    prefs = config / "fldigi_def.xml"
    text = prefs.read_text(errors="replace")
    disabled = {
        "AUDIOIO": "2",  # PulseAudio/PipeWire compatibility layer
        "CHKUSEHAMLIBIS": "0",
        "CHKUSERIGCATIS": "0",
        "CHKUSEXMLRPCIS": "0",
        "PTTKEYLINE": "0",
        "HAMLIBPTTONDATA": "0",
        "HAMLIBCMDPTT": "0",
        "RIGCATCMDPTT": "0",
        "FLRIG_FSK": "0",
        "PTTRIGHTCHANNEL": "0",
        "RTSPTT": "0",
        "DTRPTT": "0",
        "ENABLE_GPIO_PTT": "0",
        "USEPPORTPTT": "0",
        "USEUHROUTERPTT": "0",
        "CMEDIA-PTT": "0",
        "KMLSERVER_ENABLED": "0",
        "PSKREP_AUTOSTART": "0",
        "USEPSKREP": "0",
        "PSKREPAUTO": "0",
        "PSKREPLOG": "0",
        "PSKMAILS2NREPORT": "0",
        "AUTO_TALK": "0",
        "ENABLE_UDP_LOGGING": "0",
        "CONNECT_TO_MACLOGGER": "0",
        "CONNECT_TO_N3FJP": "0",
        "CONNECT_TO_FDSERVER": "0",
        "DXC_AUTO_CONNECT": "0",
        "TCP_UDP_AUTO_CONNECT": "0",
    }
    for tag, value in disabled.items():
        text, count = re.subn(fr"(<{re.escape(tag)}>).*?(</{re.escape(tag)}>)", rf"\g<1>{value}\g<2>", text, count=1)
        if count != 1:
            raise RuntimeError(f"could not force {tag} off in the isolated Fldigi profile")
    prefs.write_text(text)

    environment = os.environ.copy()
    environment["PULSE_SOURCE"] = f"{sink}.monitor"
    environment["PULSE_SINK"] = sink
    return subprocess.Popen(
        [
            "xvfb-run", "-a", "fldigi",
            "--home-dir", str(home),
            "--config-dir", str(config),
            "--xmlrpc-server-address", "127.0.0.1",
            "--xmlrpc-server-port", str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        start_new_session=True,
    )


def _connect(url: str, process: subprocess.Popen[bytes]) -> tuple[FldigiClient, ServerProxy]:
    client = FldigiClient(url, timeout=0.4)
    for _ in range(70):
        if process.poll() is not None:
            raise RuntimeError("isolated Fldigi process exited during startup")
        try:
            client.version()
            proxy = ServerProxy(url, allow_none=True, use_builtin_types=True)
            return client, proxy
        except Exception:
            time.sleep(0.15)
    raise RuntimeError("isolated Fldigi XML-RPC server did not start")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=4)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


@pytest.mark.parametrize(
    "mode",
    ["BPSK31", "BPSK63", "BPSK125", "BPSK250", "QPSK31", "QPSK63", "QPSK125", "QPSK250", "QPSK500",
     "8PSK125", "8PSK250", "8PSK500", "RTTY", "THOR8", "THOR16", "MFSK16", "MFSK32",
     "MT63-500S", "MT63-1KS"],
)
@pytest.mark.skipif(
    os.environ.get("SIGID_TEST_FLDIGI_AUDIO") != "1",
    reason="set SIGID_TEST_FLDIGI_AUDIO=1 for isolated Fldigi audio loopback",
)
def test_fldigi_payload_reaches_workbench_api_through_virtual_audio(tmp_path, mode):
    for executable in ("fldigi", "xvfb-run", "pactl"):
        if shutil.which(executable) is None:
            pytest.skip(f"{executable} is required for the Fldigi audio integration test")

    sink = f"sigid_test_{os.getpid()}"
    try:
        module_id = _run(
            "pactl", "load-module", "module-null-sink",
            f"sink_name={sink}", "rate=48000", "channels=2",
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"could not create a temporary PulseAudio null sink: {exc}")

    processes: list[subprocess.Popen[bytes]] = []
    try:
        try:
            receiver_port = _free_port()
            sender_port = _free_port()
            receiver_process = _isolated_profile(tmp_path, "rx", sink, receiver_port)
            processes.append(receiver_process)
            sender_process = _isolated_profile(tmp_path, "tx", sink, sender_port)
            processes.append(sender_process)
        except RuntimeError as exc:
            pytest.skip(str(exc))

        receiver, receiver_rpc = _connect(f"http://127.0.0.1:{receiver_port}/RPC2", receiver_process)
        sender, sender_rpc = _connect(f"http://127.0.0.1:{sender_port}/RPC2", sender_process)
        app = create_app(tmp_path / "app-data", fldigi_client=receiver)
        payload = "SIGID LOOPBACK COMPLETE 123"
        wire_text = f"{payload} {payload}" if mode.startswith(("QPSK", "8PSK")) else payload

        async def verify_workbench_decode() -> None:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                status = (await client.get("/api/fldigi/status")).json()
                assert status["connected"] is True
                assert status["mode_count"] > 100
                modes = (await client.get("/api/fldigi/modes")).json()
                assert mode in modes
                assert (await client.post("/api/fldigi/mode", json={"name": mode})).status_code == 200

                def generate_rf_free_audio() -> None:
                    receiver_rpc.text.clear_rx()
                    sender_rpc.text.clear_tx()
                    sender.select_modem(mode)
                    sender_rpc.modem.set_carrier(1000)
                    receiver_rpc.modem.set_carrier(1000)
                    sender_rpc.text.add_tx(wire_text)
                    sender_rpc.main.tx()

                await asyncio.to_thread(generate_rf_free_audio)

                deadline = time.monotonic() + 75
                while time.monotonic() < deadline:
                    output = await client.get(
                        "/api/fldigi/output", params={"start": 0, "limit": 65536}
                    )
                    assert output.status_code == 200, output.text
                    received_text = output.json()["text"]
                    if payload in received_text:
                        return
                    await asyncio.sleep(0.25)
                pytest.fail(
                    f"full known {mode} payload did not appear through the workbench API; "
                    f"received {received_text[-1000:]!r}"
                )

        asyncio.run(verify_workbench_decode())
    finally:
        for process in processes:
            _stop_process(process)
        _run("pactl", "unload-module", module_id, check=False)


@pytest.mark.skipif(
    os.environ.get("SIGID_TEST_FLDIGI_REGION_REPLAY") != "1",
    reason="set SIGID_TEST_FLDIGI_REGION_REPLAY=1 for an RF-free saved-region decode",
)
@pytest.mark.parametrize(
    "mode",
    ["BPSK31", "BPSK63", "BPSK125", "BPSK250", "QPSK31", "QPSK63", "QPSK125", "QPSK250", "QPSK500",
     "8PSK125", "8PSK250", "8PSK500", "RTTY", "THOR8", "THOR16", "MFSK16", "MFSK32",
     "MT63-500S", "MT63-1KS"],
)
def test_isolated_region_replay_decodes_a_known_fldigi_transmission(tmp_path, mode):
    """Transmit inside a null sink, save that audio, then decode it via the production replay worker."""
    for executable in ("fldigi", "xvfb-run", "pactl", "paplay", "parecord"):
        if shutil.which(executable) is None:
            pytest.skip(f"{executable} is required for the isolated region replay test")

    sink = f"sigid_region_{os.getpid()}"
    try:
        module_id = _run(
            "pactl", "load-module", "module-null-sink",
            f"sink_name={sink}", "rate=48000", "channels=2",
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"could not create an isolated PulseAudio null sink: {exc}")

    transmitter = None
    recorder = None
    replay = IsolatedFldigiReplay(decode_timeout=25)
    try:
        try:
            port = _free_port()
            transmitter = _isolated_profile(tmp_path, "tx", sink, port)
        except RuntimeError as exc:
            pytest.skip(str(exc))
        sender, sender_rpc = _connect(f"http://127.0.0.1:{port}/RPC2", transmitter)
        payload = "SIGID LOOPBACK COMPLETE 123"
        wire_text = f"{payload} {payload}" if mode.startswith(("QPSK", "8PSK")) else payload
        sender.select_modem(mode)
        sender.set_carrier(1000)
        audio_path = tmp_path / "known-transmission.wav"
        recorder = subprocess.Popen(
            ["parecord", f"--device={sink}.monitor", "--file-format=wav",
             "--rate=48000", "--channels=2", str(audio_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True,
        )
        time.sleep(0.5)
        sender_rpc.text.clear_tx()
        sender_rpc.text.add_tx(wire_text)
        sender_rpc.main.tx()
        if mode.startswith(("MT63", "THOR")):
            dwell = 40
        elif mode.startswith(("QPSK", "8PSK")):
            dwell = 24
        elif mode.startswith("MFSK"):
            dwell = 20
        else:
            dwell = 14
        time.sleep(dwell)
        sender_rpc.main.rx()
        time.sleep(0.3)
        _stop_process(recorder)
        recorder = None
        assert audio_path.stat().st_size > 100_000, "virtual sink recorded no transmission audio"

        result = replay.decode(audio_path, mode=mode, carrier_hz=1000)
        assert result["status"] == "completed", result
        assert payload in result["text"], result["text"]
        assert result["backend_version"]
    finally:
        if recorder is not None:
            _stop_process(recorder)
        if transmitter is not None:
            _stop_process(transmitter)
        replay.close()
        _run("pactl", "unload-module", module_id, check=False)


@pytest.mark.skipif(
    os.environ.get("SIGID_TEST_FLDIGI_CATALOG") != "1",
    reason="set SIGID_TEST_FLDIGI_CATALOG=1 to audit selection of the installed catalog",
)
def test_every_installed_catalog_entry_is_selectable_in_isolated_fldigi(tmp_path):
    for executable in ("fldigi", "xvfb-run", "pactl"):
        if shutil.which(executable) is None:
            pytest.skip(f"{executable} is required for the Fldigi catalog audit")

    sink = f"sigid_catalog_{os.getpid()}"
    try:
        module_id = _run(
            "pactl", "load-module", "module-null-sink",
            f"sink_name={sink}", "rate=48000", "channels=2",
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"could not create an isolated PulseAudio null sink: {exc}")

    process = None
    try:
        try:
            port = _free_port()
            process = _isolated_profile(tmp_path, "catalog", sink, port)
        except RuntimeError as exc:
            pytest.skip(str(exc))
        client, rpc = _connect(f"http://127.0.0.1:{port}/RPC2", process)
        original = client.current_modem()
        modes = client.modem_names()
        failures = []
        results = []
        for mode in modes:
            try:
                selected = client.select_mode(mode)
                started = time.monotonic()
                while selected != mode and time.monotonic() - started < 1.0:
                    time.sleep(0.05)
                    selected = client.current_modem()
                if selected != mode:
                    result = {"mode": mode, "selectable": False, "selected": selected, "reason": "backend reported a different current mode"}
                    failures.append(result)
                else:
                    result = {"mode": mode, "selectable": True, "selection_latency_ms": round((time.monotonic() - started) * 1000)}
            except Exception as exc:
                result = {"mode": mode, "selectable": False, "reason": f"{type(exc).__name__}: {exc}"}
                failures.append(result)
            results.append(result)
        failed_names = {result["mode"] for result in failures}
        id_candidates = {name: [] for name in failed_names}
        max_mode_id = int(rpc.modem.get_max_id())
        for mode_id in range(max_mode_id + 1):
            try:
                rpc.modem.set_by_id(mode_id)
                selected = client.current_modem()
            except Exception:
                continue
            if selected in id_candidates:
                id_candidates[selected].append(mode_id)
        report = {
            "backend": "fldigi",
            "version": client.version().version,
            "tested_at_utc": datetime.now(timezone.utc).isoformat(),
            "catalog_count": len(modes),
            "selectable_count": len(modes) - len(failures),
            "selection_failures": failures,
            "id_candidates_after_name_selection_failure": id_candidates,
            "results": results,
            "original_mode": original,
            "scope": "mode selection only; audio routing, decode, generation, and specialized output are not tested",
        }
        (tmp_path / "fldigi-catalog-audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, sort_keys=True))
        assert report["catalog_count"] == len(modes)
        assert report["selectable_count"] + len(failures) == len(modes)
        assert [result["mode"] for result in results] == modes
        if client.version().version == "4.2.13":
            assert failed_names == {"OFDM500F", "OFDM750F", "OFDM3500", "RSID_EOT"}
    finally:
        if process is not None:
            _stop_process(process)
        _run("pactl", "unload-module", module_id, check=False)
