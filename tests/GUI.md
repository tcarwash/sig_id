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
