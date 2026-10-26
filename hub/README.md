# Signal ID contribution hub

The hub is a separate Docker Compose project for opt-in WAV contributions and peer
review. It is a first implementation slice, not a public multi-tenant service.
Do not expose it directly to the internet without TLS/reverse-proxy setup, stronger
identity management, rate limiting, backups, and an operational review.

## Start locally

Create distinct random secrets for each collector and reviewer. The local `.env`
file is ignored by Git; do not commit it.

```sh
cd hub
cat > .env <<'EOF'
SIG_ID_HUB_UPLOAD_TOKEN=collector-a=replace-with-a-long-random-secret,collector-b=another-long-random-secret
SIG_ID_HUB_REVIEWERS=reviewer-a=reviewer-secret-one,reviewer-b=reviewer-secret-two
EOF
docker compose up --build -d
docker compose ps
```

Open <http://127.0.0.1:8020/review>. Upload credentials use `identity=secret`
entries, allowing the service to prevent contributors from reviewing their own
submissions. Reviewer entries use the same `identity=secret` form. Keep identities
and tokens unique; this initial setup has no account management or token rotation UI.
The service listens on loopback by default. Set `SIG_ID_HUB_BIND=0.0.0.0` only behind
an appropriately configured trusted TLS proxy/firewall. Use a strong random
`SIG_ID_HUB_PORT` if 8020 is occupied.

Hub SQLite metadata and lossless WAV assets persist in the `hub_data` named volume.
Back up the whole volume while the service is stopped. `docker compose down --volumes`
deletes the hub data. The hub currently accepts only uncompressed PCM WAV, with a
64 MiB per-file limit. Annotation sample/frequency bounds and manifest dimensions
are validated against the WAV. Uploads are immutable and deduplicated by contributor
identity plus audio hash.

## S3-compatible audio storage

Local-volume storage remains the default. To store WAV assets in AWS S3 or an
S3-compatible service such as MinIO, create the private bucket first, then add these
settings to the ignored `hub/.env` file:

```dotenv
SIG_ID_HUB_INSTALL_S3=1
SIG_ID_HUB_STORAGE_BACKEND=s3
SIG_ID_HUB_S3_BUCKET=your-private-bucket
SIG_ID_HUB_S3_REGION=us-east-1
SIG_ID_HUB_S3_PREFIX=sig-id-hub
# For MinIO or another compatible endpoint:
# SIG_ID_HUB_S3_ENDPOINT_URL=https://minio.example.net
# SIG_ID_HUB_S3_ADDRESSING_STYLE=path
# For S3 keys, or leave unset to use the runtime's AWS credential chain:
# SIG_ID_HUB_S3_ACCESS_KEY_ID=...
# SIG_ID_HUB_S3_SECRET_ACCESS_KEY=...
# Optional temporary credential and encryption settings:
# SIG_ID_HUB_S3_SESSION_TOKEN=...
# SIG_ID_HUB_S3_SERVER_SIDE_ENCRYPTION=AES256
# SIG_ID_HUB_S3_KMS_KEY_ID=...
```

Rebuild and start the Hub with `docker compose up --build -d`. WAV objects are stored
under `<prefix>/assets/<sha256>.wav`; the service streams them back through the
authenticated Hub audio endpoint. Configure the bucket with private access, encryption
and an appropriate retention policy. The Hub still stores its SQLite metadata in the
`hub_data` volume, so back up that volume as well as the bucket. Use bucket versioning
if you need protection from accidental object deletion. The S3 option supports one
configured bucket per Hub instance; it does not move existing WAVs automatically.

## Review

Authenticated reviewers can listen to each submitted recording and inspect its
sample/frequency bounds and optional label. `looks_right` from two distinct reviewers
on the same immutable annotation revision marks extent `extent_verified`. A correction,
multiple-signal report, false detection, or abstention blocks this simple consensus;
those cases currently require operator follow-up. Recording completeness remains a
separate `completeness` field and is not inferred from extent votes. This implementation
does not yet provide waterfall rendering, correction/adjudication UI, scene-coverage
review, curated training exports, or hub-to-collector sync.

## HTTP contribution contract

`POST /api/submissions` is multipart with:

- `audio`: PCM WAV upload.
- `manifest`: JSON string, `schema_version: 1`, audio `sample_rate`, `channels`,
  `sample_count`, and a non-empty `annotations` list. Each annotation has `sample_start`,
  `sample_count`, optional paired `frequency_low_hz`/`frequency_high_hz`, optional
  identity fields, and optional `completeness` (`complete`, `left_censored`,
  `right_censored`, `both_censored`, or `unknown`).

Use `Authorization: Bearer <collector-secret>`. Reviewers use authenticated
`GET /api/review/queue`, `GET /api/submissions/{id}/audio`, and
`POST /api/submissions/{id}/annotations/{annotation_id}/reviews` with JSON
`{"revision": 1, "decision": "looks_right"}` (or `adjust_bounds`,
`multiple_signals`, `not_signal`, `unsure`). Audio downloads also require the reviewer
bearer token. Tokens are compared server-side and never placed in URLs.

## Limitations before real deployment

The local workbench does not yet have a submit/outbox UI or hub-sync client; this
contract currently requires a caller to construct the multipart request. Reviewers
can record decisions but adjustments are not yet promoted to new annotation revisions,
and disputed items have no adjudication workflow. Consensus here concerns extent only;
it must not be treated as protocol truth or training admission. Review queue and media
access are token-authenticated but lack fine-grained organization/project ACLs.
