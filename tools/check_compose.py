"""Isolated Linux Compose build, persistence, UI and graceful-capture smoke checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import tarfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from sigid.synthetic import generate_bpsk_wav

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def http(url, data=None, content_type="application/json"):
    payload = json.dumps(data).encode() if isinstance(data, dict) else data
    request = Request(url, data=payload, headers={"Content-Type": content_type})
    with urlopen(request, timeout=15) as response:
        raw = response.read()
        return (
            json.loads(raw)
            if "application/json" in response.headers.get("Content-Type", "")
            else raw
        )


def ready(base):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            if http(base + "/api/health")["status"] == "ok":
                return
        except (URLError, OSError):
            pass
        time.sleep(0.2)
    raise RuntimeError("isolated container did not become ready")


def check(build=True):
    project = "sigid-smoke-" + uuid4().hex[:8]
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "SIG_ID_HOST": "127.0.0.1",
        "SIG_ID_PORT": str(port),
        "SIG_ID_FLDIGI_URL": "http://127.0.0.1:1/RPC2",
        "SIG_ID_MODEL_PATH": "",
    }
    compose = ["docker", "compose", "-p", project]
    capture_name = project + "-synthetic-input"
    restore_project = project + "-restore"

    def run(command, timeout=90, check=True, extra_env=None):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**env, **(extra_env or {})},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode:
            raise RuntimeError(
                f"Command failed: {command}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )
        return result.stdout.strip()

    try:
        command = [*compose, "up", "-d", "--wait", "--wait-timeout", "60"]
        if build:
            command.append("--build")
        run(command, timeout=900)
        ready(base)
        identifier = run([*compose, "ps", "-q", "workbench"])
        container = json.loads(run(["docker", "inspect", identifier]))[0]
        assert container["HostConfig"]["ReadonlyRootfs"] is True
        runtime = json.loads(
            run(
                [
                    *compose,
                    "exec",
                    "-T",
                    "workbench",
                    "python",
                    "-c",
                    "import os,json,platform,importlib.metadata as m,numpy as np; from sigid.classifier import extract_features; features=extract_features(np.sin(2*np.pi*1500*np.arange(6000)/12000),12000); assert features.shape==(266,) and np.isfinite(features).all(); print(json.dumps({'uid':os.geteuid(),'python':platform.python_version(),'numpy':m.version('numpy'),'scipy':m.version('scipy'),'fastapi':m.version('fastapi'),'model_feature_count':len(features)}))",
                ]
            )
        )
        assert runtime["uid"] == 10001
        run([*compose, "exec", "-T", "workbench", "pip", "check"])
        with tempfile.TemporaryDirectory(prefix="sigid-compose-fixture-") as directory:
            wav = Path(directory) / "source.wav"
            generate_bpsk_wav(wav, "00110110" * 32)
            audio = wav.read_bytes()
            boundary = uuid4().hex
            body = (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="compose-bpsk.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                + audio
                + f"\r\n--{boundary}--\r\n".encode()
            )
            capture = http(
                base + "/api/captures",
                body,
                f"multipart/form-data; boundary={boundary}",
            )
            assert capture["sha256"] == hashlib.sha256(audio).hexdigest()
            regions = f"{base}/api/captures/{capture['id']}/regions"
            label = http(
                regions,
                {
                    "sample_start": 0,
                    "sample_count": capture["sample_count"],
                    "family": "PSK",
                },
            )
            revised = http(
                regions + "/" + label["id"] + "/labels",
                {
                    "expected_revision": 1,
                    "family": "PSK",
                    "method": "binary phase-shift keying",
                    "notes": "Compose persistence check",
                },
            )
            assert revised["revision"] == 2
            demod = http(
                f"{base}/api/captures/{capture['id']}/demod/bpsk",
                {
                    "sample_start": 0,
                    "sample_count": capture["sample_count"],
                    "symbol_rate": 100,
                    "carrier_hz": 1500,
                },
            )
            assert demod["symbol_count"] == 256
            run([*compose, "restart", "workbench"])
            ready(base)
            restored = http(regions)
            assert len(restored) == 1 and restored[0]["revision"] == 2
            assert http(f"{base}/api/captures/{capture['id']}/audio") == audio

        # Test process termination during recording through a paced synthetic device,
        # mounted only in this disposable check, never through a host audio device.
        capture_port = free_port()
        capture_base = f"http://127.0.0.1:{capture_port}"
        run(
            [
                *compose,
                "run",
                "-d",
                "--no-deps",
                "--name",
                capture_name,
                "-e",
                f"SIG_ID_PORT={capture_port}",
                "-v",
                f"{ROOT / 'tests'}:/fixture-tests:ro",
                "workbench",
                "python",
                "-m",
                "uvicorn",
                "--app-dir",
                "/fixture-tests",
                "ui_server:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(capture_port),
            ]
        )
        ready(capture_base)
        devices = http(capture_base + "/api/audio/devices")
        assert devices and all(row["name"] == "Synthetic UI input" for row in devices)
        started = http(capture_base + "/api/live/start", {"sample_rate": 48000})
        deadline = time.monotonic() + 5
        while (
            http(capture_base + "/api/live")["frames_written"] < 4800
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        run(["docker", "stop", "--time", "20", capture_name])
        saved = [
            item
            for item in http(base + "/api/captures")
            if started["session_id"] in item["audio_path"]
        ]
        assert len(saved) == 1 and saved[0]["sample_count"] >= 4800
        assert saved[0]["source_kind"] == "live"
        with tempfile.TemporaryDirectory(prefix="sigid-compose-backup-") as directory:
            backup = Path(directory)
            run([*compose, "stop", "workbench"])
            run([*compose, "cp", "workbench:/data/.", str(backup)])
            assert (backup / "dataset.sqlite3").is_file()
            run([*compose, "start", "workbench"])
            ready(base)
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as bundle:
                bundle.add(backup, arcname=".")
            restore_port = free_port()
            restore_env = {**env, "SIG_ID_PORT": str(restore_port)}
            restore_cli = ["docker", "compose", "-p", restore_project]
            extracted = subprocess.run(
                [
                    *restore_cli,
                    "run",
                    "--rm",
                    "-T",
                    "--no-deps",
                    "workbench",
                    "tar",
                    "--no-same-owner",
                    "-xf",
                    "-",
                    "-C",
                    "/data",
                ],
                input=archive.getvalue(),
                capture_output=True,
                cwd=ROOT,
                env=restore_env,
                timeout=60,
            )
            if extracted.returncode:
                raise RuntimeError(extracted.stderr.decode(errors="replace"))
            run(
                [*restore_cli, "up", "-d", "--wait", "--wait-timeout", "60"],
                extra_env=restore_env,
            )
            restore_base = f"http://127.0.0.1:{restore_port}"
            restored_regions = http(
                f"{restore_base}/api/captures/{capture['id']}/regions"
            )
            assert len(restored_regions) == 1 and restored_regions[0]["revision"] == 2
            assert http(f"{restore_base}/api/captures/{capture['id']}/audio") == audio
            assert any(
                item["id"] == saved[0]["id"]
                for item in http(restore_base + "/api/captures")
            )
        browser = "not run (set SIGID_PLAYWRIGHT_MODULE to enable)"
        if os.environ.get("SIGID_PLAYWRIGHT_MODULE"):
            browser_env = {**env, "SIGID_UI_URL": base, "SIGID_UI_LIVE": "0"}
            result = subprocess.run(
                ["node", "tests/ui_smoke.mjs"],
                cwd=ROOT,
                env=browser_env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode:
                raise RuntimeError(result.stdout + "\n" + result.stderr)
            browser = json.loads(result.stdout.strip().splitlines()[-1])
        return {
            "schema_version": 1,
            "scope": "isolated Linux Docker Compose and synthetic browser/capture checks; no hardware or RF validation",
            "image_id": container["Image"],
            "runtime": runtime,
            "read_only_rootfs": True,
            "capture_survived_restart": True,
            "annotation_revision_survived_restart": 2,
            "replayed_audio_sha256": capture["sha256"],
            "raw_bpsk_symbols": demod["symbol_count"],
            "active_recording_finalized_on_container_stop": True,
            "backup_restore_verified": True,
            "finalized_live_sample_count": saved[0]["sample_count"],
            "browser": browser,
        }
    finally:
        run(
            [
                "docker",
                "compose",
                "-p",
                restore_project,
                "down",
                "--volumes",
                "--remove-orphans",
            ],
            check=False,
        )
        run(["docker", "rm", "-f", capture_name], check=False)
        run([*compose, "down", "--volumes", "--remove-orphans"], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    report = check(build=not args.no_build)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
