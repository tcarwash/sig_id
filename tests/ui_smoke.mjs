/** Run against an isolated SIGID_UI_URL server. See README's UI verification section. */
import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
const { chromium } = await import(
  process.env.SIGID_PLAYWRIGHT_MODULE || "playwright"
);
const root = await mkdtemp(join(tmpdir(), "sigid-ui-smoke-"));
execFileSync(process.env.SIGID_PYTHON || ".venv/bin/python", [
  "-c",
  `
from pathlib import Path
from sigid.synthetic import generate_2fsk_wav, generate_mfsk_wav, generate_ook_wav, generate_bpsk_wav, generate_qpsk_wav
p=Path(${JSON.stringify(root)})
bits='00110110'*32
generate_2fsk_wav(p/'2fsk.wav',bits,sample_rate=12000,symbol_rate=100,tone_low_hz=1000,tone_high_hz=1200)
generate_mfsk_wav(p/'mfsk.wav',[0,1,3,2]*64,tone_count=4,symbol_rate=50,base_tone_hz=1000,tone_spacing_hz=200)
for name,fn in [('ook',generate_ook_wav),('bpsk',generate_bpsk_wav),('qpsk',generate_qpsk_wav)]: fn(p/(name+'.wav'),bits)
`,
]);
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  acceptDownloads: true,
});
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
let modeChanges = 0;
let cleanupLive = false;
let lastCaptureId = null;
// Exercise controls without changing the operator's Fldigi instance.
await page.route("**/api/fldigi/**", async (route) => {
  const path = new URL(route.request().url()).pathname;
  let body = {};
  if (path.endsWith("/status"))
    body = {
      connected: true,
      name: "Test decoder",
      version: "1",
      current_mode: "BPSK31",
      mode_count: 2,
      receive_text_length: 0,
      wefax_output_available: false,
    };
  else if (path.endsWith("/modes")) body = ["BPSK31", "RTTY"];
  else if (path.endsWith("/mode")) {
    modeChanges++;
    body = { current_mode: "RTTY", output_offset: 0 };
  } else if (path.endsWith("/output")) body = { end: 0, text: "" };
  else if (path.endsWith("/products")) body = [];
  await route.fulfill({ json: body });
});
try {
  await page.goto(process.env.SIGID_UI_URL || "http://127.0.0.1:8011");
  for (const mode of ["2fsk", "mfsk", "ook", "bpsk", "qpsk"]) {
    await page.locator("#open-recording").click();
    await page.locator("#wav-file").setInputFiles(join(root, `${mode}.wav`));
    await page.locator("#upload-form button").click();
    await page.locator("#library-dialog").waitFor({ state: "hidden" });
    await page.locator("#workspace").waitFor({ state: "visible" });
    await page.waitForFunction(
      (name) => document.getElementById("capture-title").textContent === name,
      `${mode}.wav`,
    );
    assert.ok(await page.locator("#error").isHidden());
    await page.locator("#show-signals").click();
    await page.locator('[data-inspector-tab="raw"]').click();
    await page.locator("#raw-tool").selectOption(mode);
    const submit = mode === "2fsk" ? "demod-run" : `${mode}-run`;
    await page.locator(`#${submit}`).click();
    const exportId = `demod-export-${mode}`;
    await page.waitForFunction(
      (id) => !document.getElementById(id).disabled,
      exportId,
    );
    const pending = page.waitForEvent("download");
    await page.locator(`#${exportId}`).click();
    const download = await pending,
      destination = join(root, `${mode}-result.json`);
    await download.saveAs(destination);
    const result = JSON.parse(await readFile(destination, "utf8"));
    assert.equal(result.protocol, null);
    assert.equal(result.modulation_hypothesis, mode);
    lastCaptureId = result.source_capture.capture_id;
    assert.ok(result.result.symbol_count > 0);
    assert.equal(await page.locator(".raw-tool:visible").count(), 1);
    assert.equal(await page.locator("[data-raw-result]:visible").count(), 1);
  }
  // Manual labeling and direct saved-annotation selection.
  await page.locator("#close-inspector").click();
  const box = await page.locator("#overlay").boundingBox();
  await page.mouse.move(box.x + box.width * 0.25, box.y + box.height * 0.2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.75, {
    steps: 5,
  });
  await page.mouse.up();
  await page.locator("#family").fill("PSK");
  await page.locator("#save-region").click();
  await page.locator(".signal-tag.saved").first().waitFor();
  await page.locator(".signal-tag.saved").first().click();
  assert.equal(await page.locator("#family").inputValue(), "PSK");
  assert.equal(
    await page.locator("#save-region").textContent(),
    "Save revision",
  );
  await page.locator("#method").fill("quadrature phase-shift keying");
  await page.locator("#save-region").click();
  await page.waitForFunction(
    () =>
      document.getElementById("save-region").textContent ===
      "Save region label",
  );
  const labels = await (
    await page.request.get(
      new URL(`/api/captures/${lastCaptureId}/regions`, page.url()).href,
    )
  ).json();
  assert.equal(labels.length, 1);
  assert.equal(labels[0].revision, 2);
  await page.locator(".signal-tag.saved").first().click();
  await page.locator("#label-history summary").click();
  await page.waitForFunction(
    () =>
      document.getElementById("label-history-list").querySelectorAll("p")
        .length === 2,
  );
  assert.equal(
    modeChanges,
    0,
    "signal selection must not switch decoder modes",
  );
  await page.locator('[data-inspector-tab="decode"]').click();
  await page.locator("#fldigi-mode").selectOption("RTTY");
  assert.equal(modeChanges, 0, "mode choice requires explicit action");
  await page.locator("#fldigi-select").click();
  await page.waitForFunction(() =>
    document
      .getElementById("fldigi-status")
      .textContent.includes("Selected RTTY"),
  );
  assert.equal(modeChanges, 1);
  let replayRequest = null;
  await page.route("**/api/captures/*/decode", async (route) => {
    replayRequest = JSON.parse(route.request().postData());
    await route.fulfill({ json: { run_id: "region-run-01", status: "queued", mode: "RTTY" }, status: 202 });
  });
  await page.route("**/api/captures/*/decode-runs", async (route) => {
    const captureId = new URL(route.request().url()).pathname.split("/")[3];
    await route.fulfill({ json: [{
      id: "region-run-01", capture_id: captureId, track_id: "saved-region",
      sample_start: 1, sample_count: 100, sample_rate: 12000,
      current_mode: "RTTY", status: "completed", note: "Isolated replay finished.",
      output_text: "REGION DECODED", output_scope: "isolated_region_replay",
      association: "exact_capture_window",
    }] });
  });
  await page.locator("#decode-selection").click();
  await page.waitForFunction(() => document.getElementById("replay-status").textContent.includes("COMPLETED"));
  assert.ok(replayRequest.sample_start >= 0 && replayRequest.sample_count > 0);
  assert.equal(replayRequest.mode, "RTTY");
  assert.match(await page.locator("#decoded-text").textContent(), /REGION DECODED/);
  await page.locator("#tag-as-mode").click();
  await page.waitForFunction(() => document.getElementById("save-region").textContent === "Save region label");
  const taggedRegions = await (await page.request.get(
    new URL(`/api/captures/${lastCaptureId}/regions`, page.url()).href,
  )).json();
  assert.equal(taggedRegions[0].variant, "RTTY");
  // A result returning after a source change must not be exported as the new source.
  await page.locator('[data-inspector-tab="raw"]').click();
  await page.locator("#raw-tool").selectOption("qpsk");
  await page.route("**/demod/qpsk", async (route) => {
    const response = await route.fetch();
    await new Promise((resolve) => setTimeout(resolve, 1500));
    await route.fulfill({ response });
  });
  const pendingAnalysis = page.waitForRequest((request) =>
    request.url().endsWith("/demod/qpsk"),
  );
  await page.locator("#qpsk-run").click();
  await pendingAnalysis;
  await page.locator("#show-library").click();
  await page.locator("#synthetic-wav").setInputFiles(join(root, "2fsk.wav"));
  await page
    .locator("#synthetic-manifest")
    .setInputFiles(join(root, "2fsk.wav.json"));
  await page.locator("#synthetic-form button").click();
  await page.locator("#library-dialog").waitFor({ state: "hidden" });
  await page.locator(".signal-tag.saved").first().waitFor();
  assert.ok(await page.locator("#error").isHidden());
  await page.waitForTimeout(1700);
  assert.ok(await page.locator("#demod-export-qpsk").isDisabled());
  assert.equal(
    await page.locator("#decoded-text").textContent(),
    "Decoded text will appear here.",
  );
  assert.equal(
    await page.locator("#qpsk-output").textContent(),
    "Select a saved capture or region to analyze.",
  );
  await page.route("**/api/captures/*/decode-runs", async route => {
    const captureId=new URL(route.request().url()).pathname.split("/")[3];
    await route.fulfill({json:[{
      id:"saved-run-01", capture_id:captureId, track_id:"synthetic:track-1",
      sample_start:1200, sample_count:4800, sample_rate:12000,
      current_mode:"RTTY", status:"tentative_output", note:"Review this shared-input output.",
      output_text:"TEST SAVED TRACK RUN", output_scope:"shared_live_input",
      association:"track_window_tentative"
    }]});
  });
  await page.locator("#show-library").click();
  await page.locator("#capture-list .capture-row button").first().click();
  await page.locator("#library-dialog").waitFor({state:"hidden"});
  await page.locator("#show-signals").click();
  await page.locator('[data-inspector-tab="decode"]').click();
  await page.locator("#decode-runs .decode-run").waitFor();
  await page.locator("#decode-runs .decode-run").click();
  assert.match(await page.locator("#decoded-text").textContent(), /TEST SAVED TRACK RUN/);
  assert.match(await page.locator("#output-context").textContent(), /shared_live_input/);
  if (process.env.SIGID_UI_LIVE === "1") {
    const devices = await (
      await page.request.get(
        `${process.env.SIGID_UI_URL || "http://127.0.0.1:8011"}/api/audio/devices`,
      )
    ).json();
    assert.ok(
      devices.every((device) => device.name === "Synthetic UI input"),
      "Live UI tests require tests/ui_server.py, never a radio device",
    );
    cleanupLive = true;
    for (const rate of [48000, 96000, 192000]) {
      await page.locator("#show-settings").click();
      await page.locator("#audio-rate").selectOption(String(rate));
      await page.locator("#settings-dialog [data-close-dialog]").click();
      await page.locator("#live-start").click();
      await page.waitForFunction(() =>
        document
          .getElementById("waterfall-state")
          .textContent.includes("30 s display"),
      );
      if (rate === 48000) {
        let liveReplayRequest = null;
        let liveReplaySession = null;
        await page.route("**/api/live/analysis*", async (route) => {
          const status = await (await page.request.get(new URL("/api/live", page.url()).href)).json();
          liveReplaySession = status.session_id;
          const count = Math.max(1, Math.min(rate, status.frames_written));
          const start = status.frames_written - count;
          await route.fulfill({ json: {
            active: true, warming_up: false, session_id: liveReplaySession,
            spectrum: null,
            tracks: [{
              id: "track-001", track_id: `${liveReplaySession}:ui-track`,
              sample_start: start, sample_count: count,
              time_start_seconds: start / rate, time_end_seconds: status.frames_written / rate,
              frequency_low_hz: 950, frequency_high_hz: 1250, center_hz: 1100,
              bandwidth_hz: 300, peak_snr_db: 22, drift_hz_per_second: 0,
              tone_centers_hz: [1000, 1200], tone_count_candidate: 2,
              modulation_hypothesis: "FSK-like", protocol: null,
              track_observations: 4, track_state: "persistent", occupancy: 0.9,
            }],
            automatic_trial: null, analysis: { threshold_db: 18, method: "UI fixture" },
          } });
        });
        await page.route("**/api/live/sessions/*/decode", async (route) => {
          liveReplayRequest = JSON.parse(route.request().postData());
          await route.fulfill({ status: 202, json: { run_id: "live-replay-01", status: "queued", mode: "RTTY" } });
        });
        await page.route("**/api/live/sessions/*/decode-runs", async (route) => {
          await route.fulfill({ json: [{
            id: "live-replay-01", track_id: `${liveReplaySession}:ui-track`,
            sample_start: liveReplayRequest?.sample_start ?? 0,
            sample_count: liveReplayRequest?.sample_count ?? 1,
            sample_rate: rate, current_mode: "RTTY", status: "completed",
            note: "Isolated live track replay finished.", output_text: "LIVE TRACK TEXT",
            output_scope: "isolated_live_window_replay", association: "exact_live_track_window",
          }] });
        });
        await page.waitForTimeout(1100);
        await page.locator("#show-signals").click();
        await page.locator("#track-list").getByRole("button", { name: "Review region" }).first().click();
        await page.locator('[data-inspector-tab="decode"]').click();
        await page.locator("#fldigi-mode").selectOption("RTTY");
        await page.locator("#decode-selection").click();
        await page.waitForFunction(() => document.getElementById("replay-status").textContent.includes("COMPLETED"));
        assert.equal(liveReplayRequest.mode, "RTTY");
        assert.ok(liveReplayRequest.sample_count > 0);
        assert.match(await page.locator("#decoded-text").textContent(), /LIVE TRACK TEXT/);
        assert.equal(liveReplaySession, (await page.request.get(new URL("/api/live", page.url()).href).then(response => response.json())).session_id);
      }
      assert.equal(await page.locator("#time-start").textContent(), "Now");
      assert.equal(await page.locator("#time-end").textContent(), "−30 s");
      assert.equal(
        await page.locator("#waterfall").getAttribute("height"),
        String(Math.ceil((30 * rate) / 512)),
      );
      await page.locator("#freeze-view").click();
      const frozen = await page
        .locator("#waterfall")
        .evaluate((canvas) => canvas.toDataURL());
      const before = await (
        await page.request.get(new URL("/api/live", page.url()).href)
      ).json();
      await page.waitForTimeout(700);
      const after = await (
        await page.request.get(new URL("/api/live", page.url()).href)
      ).json();
      assert.ok(
        after.frames_written > before.frames_written,
        "recording continues while frozen",
      );
      assert.equal(
        await page
          .locator("#waterfall")
          .evaluate((canvas) => canvas.toDataURL()),
        frozen,
      );
      await page.locator("#freeze-view").click();
      await page.waitForFunction(
        (old) => document.getElementById("waterfall").toDataURL() !== old,
        frozen,
      );
      if (rate === 48000)
        await page.screenshot({
          path: join(root, "live-30-second-history.png"),
        });
      await page.locator("#live-stop").click();
      await page.waitForFunction(() =>
        document
          .getElementById("live-status")
          .textContent.includes("Saved live capture"),
      );
      assert.ok(await page.locator("#live-start").isEnabled());
      assert.ok(await page.locator("#freeze-view").isDisabled());
    }
    let failedStop = false;
    await page.route("**/api/live/stop", async route => {
      if (failedStop) { await route.continue(); return; }
      failedStop = true;
      await route.fetch();
      await route.fulfill({status:500,json:{detail:"Simulated metadata failure; audio retained in the capture directory"}});
    });
    await page.locator("#live-start").click();
    await page.waitForFunction(()=>document.getElementById("waterfall-state").textContent.includes("30 s display"));
    await page.locator("#live-stop").click();
    await page.waitForFunction(()=>document.getElementById("live-status").textContent.includes("Input stopped"));
    assert.ok(await page.locator("#live-start").isEnabled());
    assert.ok(await page.locator("#empty-state").isVisible());
    await page.locator("#live-start").click();
    await page.waitForFunction(()=>document.getElementById("waterfall-state").textContent.includes("30 s display"));
    await page.locator("#live-stop").click();
    await page.waitForFunction(()=>document.getElementById("live-status").textContent.includes("Saved live capture"));
  }
  await page.keyboard.press("Escape");
  assert.ok(await page.locator("#inspector").isHidden());
  await page.screenshot({ path: join(root, "desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  );
  await page.locator("#show-signals").click();
  await page.screenshot({ path: join(root, "mobile.png") });
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      result: "passed",
      artifacts: root,
      raw_exports: 5,
      mode_changes: modeChanges,
    }),
  );
} finally {
  if (cleanupLive)
    await page.request.post(new URL("/api/live/stop", page.url()).href);
  await browser.close();
}
