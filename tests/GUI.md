# Browser workflow verification

Run an isolated workbench so the smoke test's generated recordings and annotations
stay out of the operator dataset:

```sh
SIG_ID_DATA_DIR=/tmp/sigid-ui-validation .venv/bin/python -m uvicorn \
  --app-dir tests ui_server:create_app --factory --host 127.0.0.1 --port 8011
```

In another terminal, install Playwright outside the project and run the workflow:

```sh
npm install --prefix /tmp/sigid-ui-tools playwright
/tmp/sigid-ui-tools/node_modules/.bin/playwright install chromium
SIGID_UI_URL=http://127.0.0.1:8011 \
SIGID_UI_LIVE=1 \
SIGID_PLAYWRIGHT_MODULE=/tmp/sigid-ui-tools/node_modules/playwright/index.mjs \
node tests/ui_smoke.mjs
node --test tests/test_*.mjs
```

The browser test uses actual workbench upload, spectrum, labeling and raw-demodulation
APIs. It generates fixtures and verifies all five JSON exports, source switching,
annotation selection, synthetic import, explicit mode activation and responsive layouts.
The optional live checks use the paced synthetic audio device in `ui_server.py` to
verify 48/96/192 kHz capture, fixed startup history, freeze/resume and stopping/reopening.
The test verifies the synthetic device identity before starting capture.
It also verifies editing a saved annotation creates a second revision on the same
region, and an intentionally delayed raw result cannot be exported as a new source.
It simulates a failed Stop response after the input has stopped, checks that Start
becomes usable again, and starts and stops another synthetic recording.
Decoder requests are intercepted so the test does not change a running Fldigi instance.
It prints a temporary artifact directory containing desktop/mobile screenshots and
exported results. This is browser and synthetic validation; it does not test hardware
capture, radio routing or independent protocol interoperability.

The incremental display encoder can be benchmarked separately:

```sh
.venv/bin/python tools/benchmark_live_spectrum.py --output /tmp/sigid-spectrum-benchmark.json
```

The committed encoder report measures 200 ms increments at 48/96/192 kHz on
`tyler-ldesktop`: 20–22 kB per batch and median encoding times of 1.1/2.0/3.7 ms.
This excludes network, browser rendering and hardware capture.

## Collect/review and output ownership regressions

Use a fresh isolated synthetic server and temporary tooling, never the operator
dataset. `TMPDIR` may point at an agent scratch directory. Run from the repo root:

```sh
export SIGID_UI_DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sigid-ux-data-XXXXXX")"
export SIGID_UI_URL=http://127.0.0.1:8019
SIG_ID_DATA_DIR="$SIGID_UI_DATA_DIR" .venv/bin/python -m uvicorn \
  --app-dir tests ui_server:create_app --factory --host 127.0.0.1 --port 8019
```

In another terminal, preserve the same `SIGID_UI_DATA_DIR` and `SIGID_UI_URL`, and
set `SIGID_PLAYWRIGHT_MODULE` to the installed Playwright `index.mjs` path:

```sh
node tests/ui_review_flow.mjs
node tests/ui_decode_context.mjs
node tests/ui_collection_flow.mjs
node tests/ui_resize_handles.mjs
node tests/ui_live_review_switch.mjs
node tests/ui_investigate_workbench.mjs
.venv/bin/python -m pytest tests/test_collect_decode_api.py tests/test_collect_review_contract.py
.venv/bin/python -m pytest tests/test_attempt_history_api.py
node --test tests/test_decode_context.mjs
```

The review test uses real annotation/coverage APIs with a controlled queue fixture,
including save-without-completing, coverage-failure retry, next/defer navigation,
crossing-window validation and dirty-draft preservation. The output test intercepts
decoder transport and injects late completions without operating decoder apps.
The collection test seeds a synthetic persisted decoder run into the explicitly
configured temporary database and exercises real collection/queue/review/label APIs.
It verifies collection stays pending, suggestions only edit drafts, and saving
labels does not assert window coverage. Tests print screenshot artifact directories.
The live/review switch test requires the synthetic audio device, verifies frames keep
arriving while a saved clip replaces the waterfall, and checks Return to live restores
the prior selection and session state. It deliberately does not assert that the short
analysis ring pins that sample interval. The attempt-history API test verifies the
existing source-scoped attempt endpoints contain enough evidence for comparison and
records the not-yet-persisted auto-identify setting as a limitation.
The Investigate workbench test verifies the single top-level tab, Decoder/Raw
symbols local selection, selection preservation, local Fldigi Parameters disclosure,
and side-by-side comparison of exact persisted decoder attempts at desktop/mobile sizes.

The existing broad `ui_smoke.mjs` currently stops on its old mode-search `<option>`
assumption at line 62, also reproduced against clean HEAD. The new focused tests
do not replace the rest of that suite; updating the stale suite remains separate.
