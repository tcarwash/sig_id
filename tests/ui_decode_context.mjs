/** Focused output ownership regression. Decoder transport is simulated; no radio/decoder is touched. */
import assert from "node:assert/strict";

const { chromium } = await import(process.env.SIGID_PLAYWRIGHT_MODULE || "playwright");
const browser = await chromium.launch({headless: true});
const page = await browser.newPage({viewport: {width: 1440, height: 900}});
const errors = [];

page.on("pageerror", error => errors.push(error.message));
await page.route("**/app.js", async route => {
  const response = await route.fetch();
  await route.fulfill({response, body: `${await response.text()}\nglobalThis.__uxTest = {state, pollSavedDecodeRun, pollLiveDecodeRun, renderDecodeRuns, showOutput, pollFldigiOutput, watchDecoderTrial, setFldigiOffset(value) { fldigiOffset = value; }};`});
});
await page.route("**/api/fldigi/**", route => {
  const path = new URL(route.request().url()).pathname;
  return route.fulfill({json: path.endsWith("/modes") ? [] : path.endsWith("/products") ? [] : {connected: false, text: "", end: 0}});
});
await page.route("**/api/decoder-backends/**", route => route.fulfill({json: []}));
let run = {id: "run-a", capture_id: "capture-a", status: "completed", current_mode: "BPSK31", attempted_modes: ["BPSK31"],
  sample_start: 10, sample_count: 100, sample_rate: 12000, note: "Test result", output_text: "OLD SIGNAL OUTPUT", output_scope: "isolated_region_replay", association: "exact_capture_window"};
await page.route("**/api/captures/capture-a/decode-runs", route => route.fulfill({json: [run]}));
await page.route("**/api/live/sessions/session-a/decode-runs", route => route.fulfill({json: [run]}));
try {
  await page.goto(process.env.SIGID_UI_URL || "http://127.0.0.1:8018");
  await page.waitForFunction(() => Boolean(globalThis.__uxTest));
  for (const live of [false, true]) {
    for (const scenario of ["other-selection", "other-attempt", "other-output", "same-context"]) {
      const result = await page.evaluate(async ({live, scenario}) => {
        const {state, pollSavedDecodeRun, pollLiveDecodeRun} = globalThis.__uxTest;
        const selection = {sample_start: 10, sample_count: 100, frequency_low_hz: 900, frequency_high_hz: 1300};
        Object.assign(state, {liveView: live, liveSessionId: live ? "session-a" : null,
          capture: live ? null : {id: "capture-a"}, selection, regions: [], frozen: true, outputTab: "text"});
        const owner = {selectionKey: JSON.stringify([live ? "live" : "capture", live ? "session-a" : "capture-a", 10, 100, 900, 1300]), runId: "run-a"};
        state.focusedDecodeAttempt = owner;
        document.getElementById("decoded-text").textContent = "KEEP CURRENT OUTPUT";
        if (scenario === "other-selection") state.selection = {...selection, frequency_low_hz: 1000};
        if (scenario === "other-attempt") state.focusedDecodeAttempt = {...owner, runId: "run-b"};
        if (scenario === "other-output") state.outputTab = "raw";
        if (live) await pollLiveDecodeRun("session-a", "run-a", owner);
        else await pollSavedDecodeRun("capture-a", "run-a", owner);
        return {text: document.getElementById("decoded-text").textContent, outputTab: state.outputTab,
          retainedRuns: state.decodeRuns.map(run => run.id)};
      }, {live, scenario});
      assert.equal(result.text, scenario === "same-context" ? "OLD SIGNAL OUTPUT" : "KEEP CURRENT OUTPUT", `${live ? "live" : "saved"}: ${scenario}`);
      assert.equal(result.outputTab, scenario === "other-output" ? "raw" : "text");
      assert.deepEqual(result.retainedRuns, ["run-a"], "background result remains inspectable");
    }
  }
  await page.route("**/api/fldigi/output?*", route => route.fulfill({json:{end:12,text:"SHARED TEXT"}}));
  await page.evaluate(async () => {
    const {state, pollFldigiOutput, setFldigiOffset} = globalThis.__uxTest;
    setFldigiOffset(0);
    state.focusedDecodeAttempt = {runId:"run-a",selectionKey:"selected"};
    document.getElementById("decoded-text").textContent = "ISOLATED ATTEMPT";
    await pollFldigiOutput();
  });
  assert.equal(await page.locator("#decoded-text").textContent(), "ISOLATED ATTEMPT", "shared receiver output never contaminates a selected isolated attempt");
  let finishNative;
  const nativeReady = new Promise(resolve => { finishNative = resolve; });
  let releaseNative;
  const nativeGate = new Promise(resolve => { releaseNative = resolve; });
  await page.route("**/api/captures/capture-a/decoder-backends/test/decode", async route => {
    finishNative();
    await nativeGate;
    await route.fulfill({json:{status:"completed",application:"Test native",native_mode:"FT8",note:"Done",products:[{kind:"text",content:"LATE NATIVE OUTPUT"}]}});
  });
  await page.evaluate(() => {
    const {state} = globalThis.__uxTest;
    Object.assign(state,{capture:{id:"capture-a"},liveView:false,selection:{sample_start:10,sample_count:100},regions:[],outputTab:"text"});
    const mode = document.getElementById("native-decoder-mode");
    mode.add(new Option("Test", "test|ft8")); mode.value="test|ft8";
    const button = document.getElementById("native-decode-selection"); button.disabled=false; button.click();
  });
  await nativeReady;
  await page.evaluate(() => {
    globalThis.__uxTest.state.selection = {sample_start:200,sample_count:100};
    document.getElementById("decoded-text").textContent = "KEEP NEW SIGNAL";
  });
  releaseNative();
  await page.waitForResponse("**/api/captures/capture-a/decoder-backends/test/decode");
  await page.waitForTimeout(150);
  assert.equal(await page.locator("#decoded-text").textContent(),"KEEP NEW SIGNAL","native result also respects selection ownership");
  await page.evaluate(() => {
    const {state,watchDecoderTrial} = globalThis.__uxTest;
    state.selectedTrack = {track_id:"new-track"};
    watchDecoderTrial({session_id:"session-a",track_id:"old-track",run_id:"auto-old",status:"queued",current_mode:"RTTY",note:"Automatic attempt"});
  });
  assert.equal(await page.locator("#decoded-text").textContent(),"KEEP NEW SIGNAL","automatic trials on another track never steal output");
  const collectionRequests = [];
  await page.route("**/decode-runs/*/collect", async route => {
    collectionRequests.push({path: new URL(route.request().url()).pathname, body: route.request().postDataJSON()});
    await route.fulfill({status: 201, json: {capture_id: "collected-a", run_id: "run-a", message: "Collected for review"}});
  });
  for (const live of [false, true]) {
    await page.evaluate(({run, live}) => {
      const {state, renderDecodeRuns} = globalThis.__uxTest;
      Object.assign(state, {liveView: live, liveSessionId: live ? "session-a" : null,
        capture: live ? null : {id: "capture-a"}, decodeRuns: [run], inspectorMode: "classify"});
      document.getElementById("inspector").classList.remove("hidden");
      document.querySelector('[data-investigation-pane="decode"]').hidden = false;
      renderDecodeRuns();
    }, {run, live});
    await page.getByRole("button", {name: "Collect example", exact: true}).click({timeout: 4000});
    await page.waitForFunction(() => document.getElementById("collection-status").textContent.includes("Collected"));
    assert.equal(await page.locator("#review-queue-dialog").isVisible(), false, "collection does not open review");
  }
  assert.deepEqual(collectionRequests.map(item => item.path), [
    "/api/captures/capture-a/decode-runs/run-a/collect",
    "/api/live/sessions/session-a/decode-runs/run-a/collect",
  ]);
  assert.deepEqual(collectionRequests.map(item => item.body), [{notes: ""}, {notes: ""}], "run identity, not current dropdown or label drafts, owns collection");
  assert.deepEqual(errors, []);
  console.log("PASS: live/saved results retain source ownership, focused attempt, and output view");
} finally { await browser.close(); }
