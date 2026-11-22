# OnAir decoder event bridge

Signal ID can consume decoded-message notifications from a **separately running OnAir** instance through its documented local HTTP API. This is not a port of OnAir's decoders and it does not incorporate OnAir's GPL-licensed source code.

## What is integrated

- An opt-in `OnAirEventBackend` polls **GET `/api/v1/events?since=N&max=100`** about every two seconds. The first poll omits `since`, so collection begins with new events.
- Normalized decoded events appear in **Decode → Native decoder applications** and through `GET /api/decoder-backends/onair/events`.
- Every event retains the native mode name, event type, UTC timestamp, **absolute RF frequency**, fields, optional SNR and OnAir provenance. Unrecognized modes remain unrecognized; we do not invent a protocol identity.
- Bounded polling, cursor recovery, duplicate suppression, and queue/drop reporting protect the application from unbounded traffic and repeats.
- The bridge cannot tune, transmit, start or stop OnAir; it issues **GET requests only**, never opening radio or audio devices.

**Not yet integrated:** selected-WAV replay into OnAir, automatic association of OnAir messages with Signal ID CV tracks, or the OnAir decoder engines themselves. The current OnAir remote API does not advertise a selected-audio decode endpoint. Correlating its RF-referenced events with audio-frequency CV boxes will require known radio tuning, synchronized capture timestamps and passband geometry.

## Connect a local OnAir instance

1. Install and launch OnAir separately. Under **View → Data feeds → Remote control API**, enable the local server and copy its bearer token. The example port in OnAir's documentation is `8765`.
2. Configure the Signal ID server with `SIG_ID_ONAIR_URL=http://127.0.0.1:8765` and `SIG_ID_ONAIR_TOKEN` set to the copied token. For Docker Compose, use the local `.env` file; the variables are passed through `compose.yaml`. For native development, export them in the shell before running `sig-id-serve`.
3. Restart Signal ID **only when it is not capturing**. Open **Decode → Native decoder applications**. Once the event feed responds, OnAir is shown as **ready — live decode events only**.
4. Receive a mode in OnAir. Newly decoded events should appear with an `ONAIR` source label. A successful API connection does **not** mean a signal has decoded.

Keep the token local and out of source control, public logs and chats. If your OnAir API uses a different port, change the URL accordingly. Do not set an OnAir URL if you are not running OnAir; the backend will remain inactive.

## Event handling and validation

- The URL must be an explicit loopback HTTP origin with a port (`127.0.0.1`, `localhost`, `[::1]`); redirects and HTTP proxies are disabled so credentials are never sent off-host.
- Requests are limited to 100 events / 256 KiB, with a 1.5-second timeout, a 1000-event local queue and duplicate filtering.
- The backend reads OnAir's `gap` flag and includes source gaps and local queue drops in its status rather than silently promising a complete history.
- `GET /api/decoder-backends/onair/events` **drains the event queue**. Another client polling it can consume messages before the browser; an empty response can also mean OnAir has not decoded anything.
- Events are tentative external-decoder evidence and do not automatically confirm Signal ID model labels.

Tests in `tests/test_onair_backend.py` use a local simulated OnAir HTTP server to verify token authentication, cursor advancement and restart, duplicate filtering, loopback/redirect policy, error reporting, and Signal ID API output. They do not validate actual OnAir RF decoding.

Protocol reference: [OnAir Remote Control API](https://github.com/adswill/OnAir/blob/HEAD/docs/REMOTE-API.md).
