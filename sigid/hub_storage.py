"""Audio asset storage for the contribution hub.

Hub metadata remains in SQLite. Audio defaults to the local assets directory;
S3-compatible storage is opt-in and stores digest-addressed objects.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote, unquote, urlsplit

from fastapi.responses import FileResponse, StreamingResponse


class HubAudioStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.assets = self.root / "assets"
        backend = os.environ.get("SIG_ID_HUB_STORAGE_BACKEND", "local").strip().lower()
        if backend not in {"local", "s3"}:
            raise ValueError("SIG_ID_HUB_STORAGE_BACKEND must be 'local' or 's3'")
        self.backend = backend
        self.bucket = os.environ.get("SIG_ID_HUB_S3_BUCKET", "").strip()
        self.prefix = os.environ.get("SIG_ID_HUB_S3_PREFIX", "sig-id-hub").strip("/")
        self.client = None
        self.server_side_encryption = os.environ.get("SIG_ID_HUB_S3_SERVER_SIDE_ENCRYPTION", "").strip()
        self.kms_key_id = os.environ.get("SIG_ID_HUB_S3_KMS_KEY_ID", "").strip()
        if backend == "local":
            self.assets.mkdir(parents=True, exist_ok=True)
            return
        if not self.bucket:
            raise ValueError("SIG_ID_HUB_S3_BUCKET is required when S3 storage is enabled")
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("S3 audio storage requires the optional hub-s3 install extra") from exc
        addressing_style = os.environ.get("SIG_ID_HUB_S3_ADDRESSING_STYLE", "auto").strip().lower()
        if addressing_style not in {"auto", "path", "virtual"}:
            raise ValueError("SIG_ID_HUB_S3_ADDRESSING_STYLE must be auto, path, or virtual")
        client_args = {
            "service_name": "s3",
            "endpoint_url": os.environ.get("SIG_ID_HUB_S3_ENDPOINT_URL") or None,
            "region_name": os.environ.get("SIG_ID_HUB_S3_REGION") or None,
            "config": Config(s3={"addressing_style": addressing_style}),
        }
        access_key = os.environ.get("SIG_ID_HUB_S3_ACCESS_KEY_ID")
        secret_key = os.environ.get("SIG_ID_HUB_S3_SECRET_ACCESS_KEY")
        session_token = os.environ.get("SIG_ID_HUB_S3_SESSION_TOKEN")
        if bool(access_key) != bool(secret_key):
            raise ValueError("both SIG_ID_HUB_S3_ACCESS_KEY_ID and SIG_ID_HUB_S3_SECRET_ACCESS_KEY are required when using explicit credentials")
        if access_key:
            client_args.update(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
            if session_token:
                client_args["aws_session_token"] = session_token
        self.client = boto3.client(**client_args)

    @property
    def description(self) -> str:
        return "s3" if self.backend == "s3" else "local"

    def store(self, local_path: str | Path, digest: str) -> str:
        """Persist an uploaded WAV and return its stable database locator."""
        path = Path(local_path)
        if self.backend == "local":
            destination = self.assets / f"{digest}.wav"
            if destination.exists():
                path.unlink(missing_ok=True)
            else:
                path.replace(destination)
            return str(destination)
        key = self._key(digest)
        extra_args: dict[str, Any] = {"ContentType": "audio/wav", "Metadata": {"sha256": digest}}
        if self.server_side_encryption:
            extra_args["ServerSideEncryption"] = self.server_side_encryption
        if self.kms_key_id:
            extra_args["SSEKMSKeyId"] = self.kms_key_id
            extra_args["ServerSideEncryption"] = "aws:kms"
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra_args)
        return f"s3://{self.bucket}/{quote(key, safe='/')}"

    def response(self, locator: str, *, filename: str):
        if not locator.startswith("s3://"):
            return FileResponse(locator, media_type="audio/wav", filename=filename)
        bucket, key = self._parse_locator(locator)
        try:
            result = self.client.get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise FileNotFoundError(f"Hub audio object is unavailable: {exc}") from exc
        body = result["Body"]

        def chunks() -> Iterator[bytes]:
            try:
                while chunk := body.read(1024 * 1024):
                    yield chunk
            finally:
                body.close()

        headers = {"Content-Disposition": f'inline; filename="{filename}"'}
        if result.get("ContentLength") is not None:
            headers["Content-Length"] = str(result["ContentLength"])
        return StreamingResponse(chunks(), media_type="audio/wav", headers=headers)

    def _key(self, digest: str) -> str:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("audio digest must be a lowercase SHA-256 hex string")
        return f"{self.prefix + '/' if self.prefix else ''}assets/{digest}.wav"

    def _parse_locator(self, locator: str) -> tuple[str, str]:
        parsed = urlsplit(locator)
        bucket = parsed.netloc
        key = unquote(parsed.path.lstrip("/"))
        if bucket != self.bucket or not key or not key.startswith(f"{self.prefix + '/' if self.prefix else ''}assets/"):
            raise ValueError("audio object locator is outside the configured Hub bucket and prefix")
        return bucket, key
