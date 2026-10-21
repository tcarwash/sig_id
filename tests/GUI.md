# Browser workflow verification

Run an isolated workbench so the smoke test's generated recordings and annotations
stay out of the operator dataset:

```sh
SIG_ID_DATA_DIR=/tmp/sigid-ui-validation .venv/bin/sig-id-serve --port 8011
```

In another terminal, install Playwright outside the project and run the workflow:

```sh
npm install --prefix /tmp/sigid-ui-tools playwright
/tmp/sigid-ui-tools/node_modules/.bin/playwright install chromium
SIGID_UI_URL=http://127.0.0.1:8011 \
SIGID_PLAYWRIGHT_MODULE=/tmp/sigid-ui-tools/node_modules/playwright/index.mjs \
node tests/ui_smoke.mjs
node --test tests/test_*.mjs
```

The browser test uses actual workbench upload, spectrum, labeling and raw-demodulation
APIs. It generates fixtures and verifies all five JSON exports, source switching,
annotation selection, synthetic import, explicit mode activation and responsive layouts.
Decoder requests are intercepted so the test does not change a running Fldigi instance.
It prints a temporary artifact directory containing desktop/mobile screenshots and
exported results. This is browser and synthetic validation; it does not test hardware
capture, radio routing or independent protocol interoperability.
