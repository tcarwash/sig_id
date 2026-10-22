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
generate_2fsk_wav(p/'2fsk.wav','0110000101111110'+bits,sample_rate=12000,symbol_rate=100,tone_low_hz=1000,tone_high_hz=1200)
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
let taggedCaptureId = null;
let useSavedRunFixture = false;
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
  await page.locator("#show-signals").click();
  await page.locator('[data-inspector-tab="label"]').click();
  await page.locator("#mode-search").fill("PSK31");
  await page.waitForFunction(() => document.querySelector('#mode-results option[value="BPSK31"]') && document.querySelector('#mode-results option[value="QPSK31"]'));
  await page.locator("#mode-search").evaluate((input) => {
    input.value = "BPSK31";
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.waitForFunction(() => document.getElementById("variant").value === "BPSK31");
  assert.deepEqual(await Promise.all(["family", "method", "protocol", "variant"].map((id) => page.locator(`#${id}`).inputValue())), ["PSK", "BPSK", "PSK31", "BPSK31"]);
  const outputDock = page.locator("#output-dock");
  const initialDockHeight = await outputDock.evaluate((dock) => dock.getBoundingClientRect().height);
  const resizeHandle = await page.locator("#output-resize").boundingBox();
  await page.mouse.move(resizeHandle.x + resizeHandle.width / 2, resizeHandle.y + resizeHandle.height / 2);
  await page.mouse.down();
  await page.mouse.move(resizeHandle.x + resizeHandle.width / 2, resizeHandle.y - 70, { steps: 4 });
  await page.mouse.up();
  const enlargedDockHeight = await outputDock.evaluate((dock) => dock.getBoundingClientRect().height);
  assert.ok(enlargedDockHeight >= initialDockHeight + 60, "dragging the divider enlarges the output pane");
  assert.equal(await page.evaluate(() => Number(localStorage.getItem("sig-id.output-dock-height"))), Math.round(enlargedDockHeight));
  await page.locator("#output-resize").focus();
  await page.keyboard.press("Home");
  for (let step = 0; step < 4; step += 1) await page.keyboard.press("ArrowUp");
  const keyboardDockHeight = await outputDock.evaluate((dock) => dock.getBoundingClientRect().height);
  assert.ok(keyboardDockHeight > 160, "keyboard resizing restores a useful pane size");
  await page.reload();
  assert.equal(await outputDock.evaluate((dock) => Math.round(dock.getBoundingClientRect().height)), Math.round(keyboardDockHeight), "the pane height survives a reload");
  await page.locator("#toggle-output").click();
  assert.equal(await outputDock.evaluate((dock) => dock.getBoundingClientRect().height), 40, "the resized pane still collapses");
  await page.locator("#toggle-output").click();
  assert.equal(await outputDock.evaluate((dock) => Math.round(dock.getBoundingClientRect().height)), Math.round(keyboardDockHeight), "expanding restores the chosen size");
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
    if (mode === "mfsk") assert.match(await page.locator("[data-raw-result]:visible .mode-shortlist").first().textContent(), /THOR8.*THOR16.*MFSK16.*MFSK32/s);
    if (mode === "bpsk") assert.match(await page.locator("[data-raw-result]:visible .mode-shortlist").first().textContent(), /BPSK31.*BPSK63/s);
    if (mode === "qpsk") assert.match(await page.locator("[data-raw-result]:visible .mode-shortlist").first().textContent(), /QPSK31.*QPSK63.*QPSK125.*QPSK250.*QPSK500/s);
    if (mode === "2fsk") {
      assert.ok(await page.locator(".bit-ascii").allTextContents().then((items) => items.includes("a")));
      assert.match(await page.locator(".bit-byte-header").textContent(), /Bit offset.*Bits.*MSB hex.*MSB ASCII.*LSB hex.*LSB ASCII/);
      const alignment = page.locator("[aria-label='Byte alignment offset']");
      await alignment.selectOption("3");
      assert.equal(await page.locator(".bit-byte:not(.bit-byte-header) .bit-offset").first().textContent(), "000003");
      assert.match(await page.locator(".bit-metrics").textContent(), /Byte alignment 3.*skip 3 leading bits/);
      await alignment.selectOption("0");
      const transform = page.locator("[aria-label='Bitstream interpretation transform']");
      await transform.selectOption("differential-xor");
      assert.equal(await page.locator("[aria-label='Differential initial state']").isEnabled(), true);
      assert.match(await page.locator(".bit-metrics").textContent(), /bit entropy.*byte entropy/);
      await transform.selectOption("raw");
      const asciiRows = await page.locator(".bit-ascii").allTextContents();
      const asciiIndex = asciiRows.indexOf("a");
      assert.equal(await page.locator(".bit-byte").nth(asciiIndex + 1).locator(".bit-hex").first().textContent(), "61");
      assert.match(await page.locator(".bit-clues").textContent(), /HDLC-like flag/);
      assert.match(await page.locator("[data-raw-result]:visible .mode-shortlist").first().textContent(), /RTTY/);
      await page.locator(".bit-value").nth(0).click();
      await page.locator(".bit-value").nth(1).click();
      await page.locator(".bit-tag-form input[aria-label='Operator interpretation']").fill("likely header");
      await page.locator(".bit-tag-form button").click();
      assert.match(await page.locator(".bit-annotation").first().textContent(), /likely header/);
      assert.ok(await page.locator(".bit-byte[style*='--tag-color']").count() > 0, "saved tags highlight their source byte rows");
      await page.locator(".bit-stencil-form input[aria-label='Stencil name']").fill("Saved preamble");
      await page.locator(".bit-stencil-form input[aria-label='Stencil field']").fill("preamble");
      await page.locator(".bit-stencil-form input[aria-label='Possible protocols']").fill("Demo protocol");
      await page.locator(".bit-stencil-form button").click();
      await page.waitForFunction(() => document.querySelector(".bit-stencil-status")?.textContent.includes("Saved for future"));
    }
    const pending = page.waitForEvent("download");
    await page.locator(`#${exportId}`).click();
    const download = await pending,
      destination = join(root, `${mode}-result.json`);
    await download.saveAs(destination);
    const result = JSON.parse(await readFile(destination, "utf8"));
    assert.equal(result.protocol, null);
    if (mode === "2fsk") {
      taggedCaptureId = result.source_capture.capture_id;
      assert.equal(result.bit_annotations[0].label, "likely header");
      assert.deepEqual(result.bit_annotations[0].settings.interpretation, {
        transform: "raw", initial_state: 0, byte_alignment: 0, byte_order: "both",
      });
      const savedAnnotations = await (await page.request.get(new URL(`/api/captures/${taggedCaptureId}/bit-annotations`, page.url()).href)).json();
      assert.deepEqual(savedAnnotations[0].settings.interpretation, {
        transform: "raw", initial_state: 0, byte_alignment: 0, byte_order: "both",
      });
    }
    assert.equal(result.modulation_hypothesis, mode);
    lastCaptureId = result.source_capture.capture_id;
    assert.ok(result.result.symbol_count > 0);
    assert.equal(await page.locator(".raw-tool:visible").count(), 1);
    assert.equal(await page.locator("[data-raw-result]:visible").count(), 1);
  }
  await page.locator("#show-library").click();
  await page.locator(".capture-row").filter({ hasText: "2fsk.wav" }).first().getByRole("button", { name: "Open" }).click();
  await page.locator("#library-dialog").waitFor({ state: "hidden" });
  await page.waitForFunction(() => document.getElementById("saved-bit-annotations").textContent.includes("likely header"));
  lastCaptureId = taggedCaptureId;
  // Manual labeling and direct saved-annotation selection.
  await page.locator("#show-signals").click();
  await page.locator('[data-inspector-tab="raw"]').click();
  await page.locator("#raw-tool").selectOption("2fsk");
  await page.locator("#demod-run").click();
  await page.waitForFunction(() => [...document.querySelectorAll(".bit-clues")].some(node => node.textContent.includes("Saved preamble")));
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
    const run = useSavedRunFixture ? {
      id: "saved-run-01", capture_id: captureId, track_id: "synthetic:track-1",
      sample_start: 1200, sample_count: 4800, sample_rate: 12000,
      current_mode: "RTTY", status: "tentative_output", note: "Review this shared-input output.",
      output_text: "TEST SAVED TRACK RUN", output_scope: "shared_live_input",
      association: "track_window_tentative",
    } : {
      id: "region-run-01", capture_id: captureId, track_id: "saved-region",
      sample_start: 1, sample_count: 100, sample_rate: 12000,
      current_mode: "RTTY", status: "completed", note: "Isolated replay finished.",
      output_text: "REGION DECODED", output_scope: "isolated_region_replay",
      association: "exact_capture_window", carrier_hz: 1808,
    };
    await route.fulfill({ json: [run] });
  });
  assert.equal(await page.locator("#decode-selection").textContent(), "Decode selected region");
  const carrierInput = page.locator("#fldigi-carrier-hz");
  assert.ok(Number(await carrierInput.inputValue()) > 0, "the selected band center initializes the audio carrier field");
  await carrierInput.fill("1808");
  const expectedPlaybackStart = labels[0].sample_start / 12000;
  await page.locator("#play-selection").click();
  await page.waitForFunction(() => document.getElementById("audio-region-status").textContent.startsWith("Playing selected audio"));
  await page.waitForFunction((expectedStart) => document.getElementById("audio-player").currentTime >= expectedStart, expectedPlaybackStart, { timeout: 3000 });
  const playedStart = await page.locator("#audio-player").evaluate((audio) => audio.currentTime);
  assert.ok(playedStart >= expectedPlaybackStart, "selected-region replay seeks to its sample-indexed start");
  await page.locator("#play-selection").click();
  assert.equal(await page.locator("#play-selection").textContent(), "Replay selected region");
  assert.match(await page.locator("#audio-region-status").textContent(), /stopped/i);
  await page.locator("#decode-selection").click();
  await page.waitForFunction(() => document.getElementById("replay-status").textContent.includes("COMPLETED"));
  assert.ok(replayRequest.sample_start >= 0 && replayRequest.sample_count > 0);
  assert.equal(replayRequest.mode, "RTTY");
  assert.equal(replayRequest.carrier_hz, 1808);
  assert.match(await page.locator(".decode-run").textContent(), /carrier 1808 Hz/);
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
  useSavedRunFixture = true;
  await page.locator("#show-library").click();
  await page.locator("#capture-list .capture-row").filter({ hasText: "qpsk.wav" }).first().getByRole("button", { name: "Open" }).click();
  await page.locator("#library-dialog").waitFor({state:"hidden"});
  await page.locator("#show-library").click();
  await page.locator("#capture-list .capture-row").filter({ hasText: "2fsk.wav" }).first().getByRole("button", { name: "Open" }).click();
  await page.locator("#library-dialog").waitFor({state:"hidden"});
  await page.locator("#show-signals").click();
  await page.locator('[data-inspector-tab="decode"]').click();
  await page.locator("#decode-runs .decode-run").waitFor();
  await page.locator("#decode-runs .decode-run").click();
  assert.match(await page.locator("#decoded-text").textContent(), /TEST SAVED TRACK RUN/);
  assert.match(await page.locator("#output-context").textContent(), /shared_live_input/);
  const proposalResponse = await page.request.post(
    new URL(`/api/captures/${taggedCaptureId}/regions`, page.url()).href,
    { data: { sample_start: 2000, sample_count: 2000, frequency_low_hz: 800, frequency_high_hz: 1400, mode_id: "bpsk31", origin: "detector_proposal" } },
  );
  assert.equal(proposalResponse.status(), 201);
  const proposal = await proposalResponse.json();
  await page.locator("#show-library").click();
  await page.locator("#capture-list .capture-row").filter({ hasText: "2fsk.wav" }).first().getByRole("button", { name: "Open" }).click();
  await page.locator("#library-dialog").waitFor({ state: "hidden" });
  await page.locator("#show-signals").click();
  await page.locator(".region-shortcut").filter({ hasText: "BPSK31" }).click();
  assert.ok(await page.locator("#feedback-panel").isVisible());
  await page.locator("#mode-search").fill("RTTY");
  await page.waitForFunction(() => document.querySelector('#mode-results option[value="RTTY"]'));
  await page.locator("#mode-search").evaluate((input) => {
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  assert.match(await page.locator("#mode-resolution").textContent(), /RTTY.*FSK.*2-FSK/);
  await page.locator("#feedback-wrong").click();
  await page.waitForFunction(() => document.getElementById("feedback-status").textContent.includes("incorrect identity recorded"));
  assert.equal((await (await page.request.get(new URL(`/api/captures/${taggedCaptureId}/regions`, page.url()).href)).json()).find((region) => region.id === proposal.id).variant, "RTTY");
  await page.locator("#edit-region-bounds").click();
  const geometryBox = await page.locator("#overlay").boundingBox();
  await page.mouse.move(geometryBox.x + geometryBox.width * 0.35, geometryBox.y + geometryBox.height * 0.3);
  await page.mouse.down();
  await page.mouse.move(geometryBox.x + geometryBox.width * 0.55, geometryBox.y + geometryBox.height * 0.55, { steps: 4 });
  await page.mouse.up();
  await page.locator("#save-region-bounds").click();
  await page.waitForFunction(() => document.getElementById("geometry-status").textContent.includes("geometry revision 2"));
  const geometryHistory = await (await page.request.get(new URL(`/api/captures/${taggedCaptureId}/regions/${proposal.id}/geometry`, page.url()).href)).json();
  assert.equal(geometryHistory.length, 2);
  assert.notEqual(geometryHistory[0].sample_start, geometryHistory[1].sample_start);
  const captureInfo = await (await page.request.get(new URL(`/api/captures/${taggedCaptureId}`, page.url()).href)).json();
  const splitParentResponse = await page.request.post(
    new URL(`/api/captures/${taggedCaptureId}/regions`, page.url()).href,
    { data: { sample_start: 0, sample_count: captureInfo.sample_count, frequency_low_hz: 100, frequency_high_hz: 2900, mode_id: "bpsk31", origin: "detector_proposal" } },
  );
  assert.equal(splitParentResponse.status(), 201);
  const splitParent = await splitParentResponse.json();
  await page.locator("#show-library").click();
  await page.locator(`[data-capture-id="${taggedCaptureId}"] button`).click();
  await page.locator("#library-dialog").waitFor({ state: "hidden" });
  await page.locator("#show-library").click();
  await page.locator(`#region-list [data-region-id="${splitParent.id}"] button`).click();
  await page.locator("#library-dialog").waitFor({ state: "hidden" });
  await page.locator("#start-split").click();
  const splitBox = await page.locator("#overlay").boundingBox();
  for (const [x0, x1, y0, y1] of [[0.20, 0.32, 0.40, 0.47], [0.62, 0.75, 0.55, 0.62]]) {
    await page.mouse.move(splitBox.x + splitBox.width * x0, splitBox.y + splitBox.height * y0);
    await page.mouse.down();
    await page.mouse.move(splitBox.x + splitBox.width * x1, splitBox.y + splitBox.height * y1, { steps: 4 });
    await page.mouse.up();
  }
  await page.locator("#save-split").click();
  await page.waitForFunction(() => document.getElementById("split-status").textContent.includes("Split recorded"));
  const lineage = await (await page.request.get(new URL(`/api/captures/${taggedCaptureId}/regions/${splitParent.id}/lineage`, page.url()).href)).json();
  const splitChildren = lineage[0].result_region_ids;
  const afterSplit = await (await page.request.get(new URL(`/api/captures/${taggedCaptureId}/regions`, page.url()).href)).json();
  assert.equal(afterSplit.find((region) => region.id === splitParent.id).lineage_state, "superseded");
  await page.locator("#show-library").click();
  for (const childId of splitChildren) await page.locator(`[data-region-merge-id="${childId}"]`).check();
  await page.locator('#library-dialog [data-close-dialog]').click();
  await page.locator("#merge-regions").click();
  await page.waitForFunction(() => document.getElementById("merge-status").textContent.includes("Merge recorded"));
  const capturesForRegistry = await (await page.request.get(new URL("/api/captures", page.url()).href)).json();
  const independentCapture = capturesForRegistry.find((capture) => capture.id !== taggedCaptureId && capture.sha256 !== captureInfo.sha256);
  assert.ok(independentCapture, "mode proposals need a second independent audio recording");
  await page.request.post(new URL(`/api/captures/${independentCapture.id}/regions`, page.url()).href, {
    data: { sample_start: 0, sample_count: Math.min(2000, independentCapture.sample_count) },
  });
  await page.locator("#mode-proposal-details > summary").click();
  await page.locator("#mode-proposal-capture-2").selectOption(independentCapture.id);
  await page.waitForFunction(() => document.querySelectorAll("#mode-proposal-region-2 option").length > 1);
  await page.locator("#mode-proposal-region-2").selectOption({ index: 1 });
  for (const [id, value] of [
    ["mode-proposal-id", "smoke-custom-multitone"], ["mode-proposal-name", "Smoke Custom Multitone"],
    ["mode-proposal-family", "MFSK"], ["mode-proposal-method", "multi-tone FSK"],
    ["mode-proposal-protocol", "Smoke Protocol"], ["mode-proposal-variant", "X1"],
    ["mode-proposal-aliases", "SCM1, smoke multitone"],
  ]) await page.locator(`#${id}`).fill(value);
  await page.locator("#mode-proposal-rationale").fill("This is a repeatable user-observed mode with a stable waveform and independently collected evidence from two different audio recordings for initial local registry review.");
  await page.locator("#mode-proposal-distinguishing").fill("Its observed multi-tone spacing and transition sequence distinguish it from the existing supported MFSK catalog entries in both attached independent source recordings.");
  await page.locator("#mode-proposal-observations-1").fill("The selected source has regular multi-tone transitions over this exact sample range; these are not explained by the neighboring known modes.");
  await page.locator("#mode-proposal-observations-2").fill("The independent recording reproduces the tone spacing and cadence while using a different source waveform hash and capture lineage.");
  await page.locator("#mode-proposal-form button[type=submit]").click();
  await page.waitForFunction(() => document.getElementById("mode-proposal-status").textContent.includes("pending proposal"));
  await page.locator("#mode-proposal-review-note").fill("Reviewed the two independent source regions and the stated distinguishing tone evidence; local confirmation only, pending independent community review.");
  const proposalCard = page.locator("#mode-proposal-list article").filter({ hasText: "Smoke Custom Multitone" });
  await proposalCard.getByRole("button", { name: "Accept locally" }).click();
  await page.waitForFunction(() => document.getElementById("mode-proposal-status").textContent.includes("not peer verification"));
  const customSearch = await (await page.request.get(new URL("/api/modes?q=SCM1", page.url()).href)).json();
  assert.equal(customSearch.modes[0].id, "smoke-custom-multitone");
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
        let liveSweepRequest = null;
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
        await page.route("**/api/live/sessions/*/decode-trials", async (route) => {
          liveSweepRequest = JSON.parse(route.request().postData());
          await route.fulfill({ status: 202, json: {
            run_id: "live-trial-01", session_id: liveReplaySession,
            track_id: liveSweepRequest.track_id, status: "running",
            modes: ["RTTY", "THOR16", "BPSK31"], current_mode: "RTTY",
            note: "Isolated sweep queued.",
          } });
        });
        await page.route("**/api/live/sessions/*/decode-runs", async (route) => {
          await route.fulfill({ json: [{
            id: "live-replay-01", track_id: `${liveReplaySession}:ui-track`,
            sample_start: liveReplayRequest?.sample_start ?? 0,
            sample_count: liveReplayRequest?.sample_count ?? 1,
            sample_rate: rate, current_mode: "RTTY", status: "completed",
            attempted_modes: ["RTTY"], modes: ["RTTY"],
            note: "Isolated live track replay finished.", output_text: "LIVE TRACK TEXT",
            output_scope: "isolated_live_window_replay", association: "exact_live_track_window",
          }, {
            id: "live-trial-01", track_id: liveSweepRequest?.track_id || `${liveReplaySession}:ui-track`,
            sample_start: liveSweepRequest?.sample_start ?? 0,
            sample_count: liveSweepRequest?.sample_count ?? 1,
            sample_rate: rate, current_mode: "RTTY", status: "tentative_output",
            attempted_modes: ["RTTY"], modes: ["RTTY", "THOR16", "BPSK31"],
            note: "RTTY recovered text; review it.", output_text: "ISOLATED SWEEP TEXT",
            output_scope: "isolated_live_mode_sweep", association: "exact_live_track_window",
          }] });
        });
        await page.waitForTimeout(1100);
        await page.locator("#show-signals").click();
        await page.locator("#track-list").getByRole("button", { name: "Review region" }).first().click();
        await page.locator('[data-inspector-tab="decode"]').click();
        await page.locator("#fldigi-mode").selectOption("RTTY");
        assert.equal(await page.locator("#play-selection").isDisabled(), true);
        assert.equal(await page.locator("#decode-selection").textContent(), "Decode live track window");
        await page.locator("#decode-selection").click();
        await page.waitForFunction(() => document.getElementById("replay-status").textContent.includes("COMPLETED"));
        assert.equal(liveReplayRequest.mode, "RTTY");
        assert.ok(liveReplayRequest.sample_count > 0);
        assert.match(await page.locator("#decoded-text").textContent(), /LIVE TRACK TEXT/);
        assert.equal(liveReplaySession, (await page.request.get(new URL("/api/live", page.url()).href).then(response => response.json())).session_id);
        await page.locator('[data-inspector-tab="signals"]').click();
        await page.locator("#track-list").getByRole("button", { name: "Try tested modes" }).first().click();
        await page.waitForFunction(() => document.getElementById("fldigi-status").textContent.includes("tentative_output"));
        assert.equal(liveSweepRequest.track_id, `${liveReplaySession}:ui-track`);
        assert.equal(liveSweepRequest.modulation_hypothesis, "FSK-like");
        assert.match(await page.locator("#decoded-text").textContent(), /ISOLATED SWEEP TEXT/);
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
  const mobileDockHeight = await outputDock.evaluate((dock) => dock.getBoundingClientRect().height);
  await page.locator("#output-resize").focus();
  await page.keyboard.press("ArrowUp");
  assert.ok(await outputDock.evaluate((dock) => dock.getBoundingClientRect().height) > mobileDockHeight, "the output pane is keyboard-resizable on mobile widths");
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
